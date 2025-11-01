# Meta-HDI

Meta-HDI project for identifying herb–drug interactions in two stages:

1) **DDI pretraining** (drug–drug interaction knowledge)  
2) **HDI finetuning** (herb–drug interactions with compound-level interpretation)

---

## 📦 Artifacts (download & place under `data/`)

Large files are hosted externally due to GitHub size limits.

**Download:** https://drive.google.com/file/d/134L6bnYsrcsAbSDVOuuxzAoAgJBhQ4t4/view?usp=sharing

**Expected contents after extract / download (put in `data/`):**
- `graph_HDI_start_1_26394.pkl`
- `DDI_path_dict_120129_onlypos_path_unspecified_type_final.pkl`
- `model_embedding_generator_64_120129_deepDDI_multi_label.pth`  ← optional init for node embeddings
- `model_DDI_predictor_120129_deepDDI_multi_label.pth`           ← DDI-pretrained predictor weights

> You can also run DDI pretraining yourself (scripts below) and ignore the `.pth` files.

---

## 📁 Repo layout

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

## 🚀 Installation

We recommend Python **3.10** (tested with 3.10.11). From a clean environment:

> If you plan to use GPU, install the appropriate **PyTorch** build for your CUDA version from https://pytorch.org/get-started/locally/ and then install the rest from `requirements.txt` (remove the `torch==...` line first, or reinstall torch with the CUDA wheel).

```bash
# conda
conda create -n meta-hdi python=3.10 -y
conda activate meta-hdi

# deps
pip install -r requirements.txt
```

---

## ⚙️ Configuration (`configs/*.yaml`)

Two minimal YAMLs control paths and hyperparameters. **Edit the paths** to match your local `data/`.

**`configs/train_ddi.yaml`**
```yaml
device: "cuda:0"        # or "cpu"
out_dir: "outputs/ddi"
data:
  graph: "D:/meta-hdi/data/graph_HDI_start_1_26394.pkl"
  ddi_paths: "D:/meta-hdi/data/DDI_path_dict_120129_onlypos_path_unspecified_type_final.pkl"

model:
  emb_dim: 128
  max_len: 8
  num_types: 7
  num_labels: 8          # DDI multi-label size

# optional: initialize from provided embedding/predictor checkpoints
init:
  embedding: "D:/meta-hdi/data/model_embedding_generator_64_120129_deepDDI_multi_label.pth"
  predictor: ""          # leave empty to train from scratch

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
  hdi_paths: "D:/meta-hdi/data/KMSDR_ing_com_path_dict_unspecified_240811.pkl"  # provide your HDI paths

model:
  emb_dim: 128
  max_len: 8
  num_types: 7
  num_labels: 3          # e.g., [increase, decrease, excretion]

# use DDI-pretrained predictor (from artifacts) for better finetuning
pretrained:
  path: "D:/meta-hdi/data/model_DDI_predictor_120129_deepDDI_multi_label.pth"
  freeze_backbone: true

train:
  epochs: 20
  batch_size: 32
  lr: 5e-4
  seed: 42
```

**Key fields**
- `data.graph`: graph pickle (dict with `num_nodes`, `edges`, optional `pretrained_node_emb`)
- `data.ddi_paths` / `data.hdi_paths`: path-dict pickles (list of items with `paths`, `types`, `label`)
- `model.max_len`: path padding length (e.g., 8)
- `model.num_types`: number of node types (herb/compound/protein/drug/etc.)
- `model.num_labels`: number of multilabel outputs (DDI vs HDI)

---

## ▶️ Quickstart

```bash
# 1) DDI pretraining (optional if using provided predictor .pth)
python scripts/train_ddi.py --config configs/train_ddi.yaml

# 2) HDI finetuning (loads DDI-pretrained weights if configured)
python scripts/finetune_hdi.py --config configs/finetune_hdi.yaml
```

During finetuning, you can **freeze** backbone modules via config (e.g., GCN/LSTM) and update only attention/classifier.

---
