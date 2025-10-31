#!/usr/bin/env python
# scripts/finetune_hdi.py
from __future__ import annotations
import os, sys, json, random, argparse
from pathlib import Path

import numpy as np
try:
    import yaml
except Exception as e:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from e

import torch
from torch.utils.data import DataLoader, random_split
from meta_hdi.data.loaders import load_graph, load_path_items, PathDataset, collate_paths
from meta_hdi.models.meta_hdi import MetaHDI
from meta_hdi.train.loop import train_epoch, eval_epoch

def set_seed(seed: int = 42):
    import numpy as _np, random as _rand, torch as _torch
    _rand.seed(seed); _np.random.seed(seed); _torch.manual_seed(seed); _torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")

    out_dir = Path(cfg.get("out_dir", "outputs/hdi"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("train", {}).get("seed", 42)
    set_seed(seed)

    # ----- Load data
    graph_pkl = cfg["data"]["graph"]
    hdi_pkl   = cfg["data"]["hdi_paths"]

    graph = load_graph(graph_pkl)
    items = load_path_items(hdi_pkl)

    max_len = int(cfg["model"].get("max_len", 8))
    num_types = int(cfg["model"].get("num_types", 7))

    dataset = PathDataset(items, max_len=max_len)
    num_labels = int(cfg["model"].get("num_labels", dataset.num_labels))
    if num_labels != dataset.num_labels:
        print(f"[warn] Overriding num_labels ({num_labels}) -> {dataset.num_labels} from data")
        num_labels = dataset.num_labels

    # train/val split (90/10)
    N = len(dataset)
    n_val = max(1, int(0.1 * N))
    n_train = N - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    collate = lambda batch: collate_paths(batch, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"].get("batch_size", 32)),
                              shuffle=True, num_workers=0, collate_fn=collate, pin_memory=("cuda" in device_str))
    val_loader   = DataLoader(val_ds, batch_size=int(cfg["train"].get("batch_size", 32)),
                              shuffle=False, num_workers=0, collate_fn=collate, pin_memory=("cuda" in device_str))

    # ----- Build model
    num_nodes = int(graph["num_nodes"])
    edges = graph.get("edges", None)
    model = MetaHDI(
        num_nodes=num_nodes,
        num_types=num_types,
        num_labels=num_labels,
        emb_dim=int(cfg["model"].get("emb_dim", 128)),
        type_emb_dim= int(cfg["model"].get("type_emb_dim", 8)),
        edges=edges,
        lstm_hidden=int(cfg["model"].get("lstm_hidden", 128)),
        freeze_gcn=bool(cfg["model"].get("freeze_gcn", False)),
    ).to(device)

    # optional: initialize node embeddings if provided
    if "pretrained_node_emb" in graph and graph["pretrained_node_emb"] is not None:
        try:
            emb = torch.as_tensor(graph["pretrained_node_emb"], dtype=torch.float32, device=device)
            model.load_pretrained_node_emb(emb)
            print("[info] Loaded pretrained_node_emb from graph pickle")
        except Exception as e:
            print(f"[warn] Failed to load pretrained_node_emb: {e}")

    # ----- Load DDI-pretrained weights
    pre = cfg.get("pretrained", {})
    if pre.get("path"):
        sd = torch.load(pre["path"], map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[info] Loaded pretrained: {pre['path']} (strict=False)")
        print(f"[info] missing={len(missing)}, unexpected={len(unexpected)}")

    # Optional: freeze backbone
    if pre.get("freeze_backbone", False):
        for n,p in model.named_parameters():
            if n.startswith(("gcn.", "lstm.", "proj")):
                p.requires_grad = False
        print("[info] Backbone frozen (gcn/lstm/proj)")

    # ----- Optimizer
    lr = float(cfg["train"].get("lr", 5e-4))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best = {"epoch": -1, "auroc_macro": -1.0}
    log = []

    epochs = int(cfg["train"].get("epochs", 20))
    for ep in range(1, epochs+1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val = eval_epoch(model, val_loader, device)
        val["train_loss"] = tr_loss
        val["epoch"] = ep
        log.append(val)
        print(f"[{ep:03d}] train_loss={tr_loss:.4f} val_loss={val['val_loss']:.4f} "
              f"AUROC={val['auroc_macro']:.4f} F1(μ)={val['f1_micro']:.4f} F1(M)={val['f1_macro']:.4f}")

        if val["auroc_macro"] > best["auroc_macro"]:
            best = {"epoch": ep, **val}
            torch.save(model.state_dict(), out_dir / "hdi_finetuned.pt")
            print(f"[info] Saved best to {out_dir / 'hdi_finetuned.pt'}")

    with open(out_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, "history": log}, f, ensure_ascii=False, indent=2)

    print("[done] best:", best)

if __name__ == "__main__":
    main()
