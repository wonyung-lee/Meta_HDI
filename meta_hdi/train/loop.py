# meta_hdi/train/loop.py
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
from .metrics import compute_macro_auroc, compute_f1

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k,v in batch.items()}
        out = model(batch)
        loss = F.binary_cross_entropy_with_logits(out["logits"], batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu().item())
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, device) -> Dict[str, float]:
    model.eval()
    y_true_list, y_prob_list = [], []
    total_loss = 0.0
    for batch in loader:
        batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k,v in batch.items()}
        out = model(batch)
        logits = out["logits"]
        total_loss += float(F.binary_cross_entropy_with_logits(logits, batch["label"]).cpu().item())
        prob = torch.sigmoid(logits).cpu().numpy()
        y_prob_list.append(prob)
        y_true_list.append(batch["label"].cpu().numpy())
    auroc = compute_macro_auroc(y_true_list, y_prob_list)
    f1s = compute_f1(y_true_list, y_prob_list)
    return {"val_loss": total_loss / max(1, len(loader)), "auroc_macro": auroc, **f1s}
