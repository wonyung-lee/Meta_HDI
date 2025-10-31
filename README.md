# Meta-HDI

Meta-HDI project for identifying herb-drug interactions to demonstrate two stages:

1) **DDI pretraining** (drug–drug interaction knowledge)
2) **HDI finetuning** (herb–drug interactions with compound-level interpretation)

---

## What’s in this repo (intended layout)

```
meta-hdi/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ train_ddi.yaml
│  └─ finetune_hdi.yaml
├─ meta_hdi/
│  ├─ data/
│  │  └─ loaders.py            # graph & path-dict loaders, padding collate
│  ├─ models/
│  │  └─ meta_hdi.py           # GCN/LSTM/attention backbone (training only)
│  └─ train/
│     ├─ loop.py               # training & evaluation loops
│     └─ metrics.py            # AUROC/F1 for multilabel
└─ scripts/
   ├─ train_ddi.py             # pretrain entry point
   └─ finetune_hdi.py          # finetune entry point
```
---

## Installation

We recommend Python **3.10** (tested with 3.10.11). From a clean environment:
> If you plan to use GPU, install the appropriate **PyTorch** build for your CUDA version from https://pytorch.org/get-started/locally/ and then install the rest from `requirements.txt` (remove the torch line first, or install torch again with the CUDA wheel).

---

## Configuration (`configs/*.yaml`)

Two minimal YAMLs control paths and hyperparameters.

**`configs/train_ddi.yaml`**
```yaml
device: "cuda:0"        # or "cpu"
out_dir: "outputs/ddi"
data:
  graph: "D:/meta-hdi/data/graph_HDI_start_1_26394.pkl"
  ddi_paths: "D:/meta-hdi/data/DDI_path_dict_120129_onlypos_path_unspecified.pkl"
model:
  emb_dim: 128
  max_len: 8
  num_types: 7
  num_labels: 8          # DDI multi-label size
train:
  epochs: 10
  batch_size: 64
  lr: 7e-4
  seed: 42
```

**`configs/finetune_hdi.yaml`**
```yaml
device: "cuda:0"
out_dir: "outputs/hdi"
data:
  graph: "D:/meta-hdi/data/graph_HDI_start_1_26394.pkl"
  hdi_paths: "D:/meta-hdi/data/KMSDR_ing_com_path_dict_unspecified_240811.pkl"
model:
  emb_dim: 128
  max_len: 8
  num_types: 7
  num_labels: 3          # e.g., increase / decrease / excretion
pretrained:
  path: "outputs/ddi/ddi_pretrained.pt"
  freeze_backbone: true
train:
  epochs: 20
  batch_size: 32
  lr: 5e-4
  seed: 42
```

**Key fields**
- `data.graph`: graph pickle (dict with `num_nodes`, `edges`, optional `pretrained_node_emb`)
- `data.ddi_paths`/`data.hdi_paths`: path-dict pickles (list of items with `paths`, `types`, `label`)
- `model.max_len`: path padding length (e.g., 8)
- `model.num_types`: number of node types (herb/compound/protein/drug/etc.)
- `model.num_labels`: number of multilabel outputs for the task

---

## Quickstart

```bash
# 1) DDI pretraining
python scripts/train_ddi.py --config configs/train_ddi.yaml

# 2) HDI finetuning (loads pretrained weights)
python scripts/finetune_hdi.py --config configs/finetune_hdi.yaml
```

During finetuning, you can **freeze** some backbone modules via config (e.g., GCN/LSTM) and update only attention/classifier layers.

---

## Data & Compliance

- This repository does **not** ship proprietary or restricted databases (DrugBank, TTD, etc.).
- If you use such sources, ensure you have the appropriate **licenses/permissions** and build the required pickles locally.
- Clinical/IRB datasets are **not** included in this training-only release.

---

## License

Choose a license appropriate for your institution and intended reuse. Common choices:
- **Apache-2.0** (recommended for open research with patent grant)
- MIT

Add `LICENSE` accordingly.
