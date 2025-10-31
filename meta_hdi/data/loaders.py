# meta_hdi/data/loaders.py
# Utilities to load graph & path pickles and to collate batches with padding.
from __future__ import annotations
import pickle
from typing import Any, Dict, List, Tuple
import torch
from torch.utils.data import Dataset

PADDING_ID = 0  # node index padding

class PathDataset(Dataset):
    """Dataset holding pre-extracted paths for (herb, drug) or (drug, drug) pairs.
    Each item is a dict with:
      - 'paths': List[List[int]]  (node indices per path)
      - 'types': List[List[int]]  (node-type ids per node in each path)
      - 'label': List[int] or List[float] (multilabel target)
    """
    def __init__(self, items: List[Dict[str, Any]], max_len: int = 8):
        self.items = items
        self.max_len = max_len

        # basic validation
        if len(items) == 0:
            raise ValueError("Empty items for PathDataset")
        # infer label size
        first_lbl = items[0].get("label", None)
        if first_lbl is None:
            raise ValueError("Each item must have a 'label' key")
        self.num_labels = len(first_lbl) if hasattr(first_lbl, "__len__") else 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

def _pad(seq: List[int], L: int, pad: int) -> List[int]:
    if len(seq) >= L:
        return seq[:L]
    return seq + [pad] * (L - len(seq))

def collate_paths(batch: List[Dict[str, Any]], max_len: int = 8, pad: int = PADDING_ID) -> Dict[str, torch.Tensor]:
    """Pad a batch of path dictionaries to tensors.

    Returns:
      {
        'paths': LongTensor [B, Lp, max_len],
        'types': LongTensor [B, Lp, max_len],
        'mask':  BoolTensor [B, Lp, max_len],  # True where valid node
        'path_mask': BoolTensor [B, Lp],       # True where path has at least one valid node
        'label': FloatTensor [B, C]
      }
    """
    B = len(batch)
    # determine max number of paths in this batch
    Lp = max(len(x["paths"]) for x in batch)
    # label size
    C = len(batch[0]["label"]) if hasattr(batch[0]["label"], "__len__") else 1

    Xp = torch.full((B, Lp, max_len), pad, dtype=torch.long)
    Xt = torch.zeros((B, Lp, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        paths = item["paths"]
        types = item["types"]
        assert len(paths) == len(types), "paths/types length mismatch"
        for j in range(len(paths)):
            Xp[i, j] = torch.tensor(_pad(paths[j], max_len, pad), dtype=torch.long)
            Xt[i, j] = torch.tensor(_pad(types[j], max_len, 0), dtype=torch.long)

    mask = (Xp != pad)  # [B, Lp, max_len]
    # a path exists if any node is valid
    path_mask = mask.any(dim=-1)  # [B, Lp]

    # labels
    y = torch.zeros((B, C), dtype=torch.float32)
    for i, item in enumerate(batch):
        lab = item["label"]
        if hasattr(lab, "__len__"):
            y[i, :len(lab)] = torch.tensor(lab, dtype=torch.float32)
        else:
            y[i, 0] = float(lab)

    return {"paths": Xp, "types": Xt, "mask": mask, "path_mask": path_mask, "label": y}

def load_graph(graph_pkl: str) -> Dict[str, Any]:
    """Load a graph pickle expected to contain:
      {
        'num_nodes': int,
        'edges': List[Tuple[int,int]],
        'node_types': Optional[List[int]],
        'pretrained_node_emb': Optional[np.ndarray or list],
      }
    """
    with open(graph_pkl, "rb") as f:
        g = pickle.load(f)
    if not isinstance(g, dict):
        raise ValueError("Graph pickle must be a dict-like object.")
    if "num_nodes" not in g:
        raise KeyError("Graph dict missing 'num_nodes'.")
    if "edges" not in g:
        raise KeyError("Graph dict missing 'edges'.")
    return g

def load_path_items(path_pkl: str) -> List[Dict[str, Any]]:
    with open(path_pkl, "rb") as f:
        items = pickle.load(f)
    if not isinstance(items, list):
        raise ValueError("Path pickle must be a list of items.")
    return items
