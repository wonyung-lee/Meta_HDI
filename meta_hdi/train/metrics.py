# meta_hdi/train/metrics.py
from __future__ import annotations
import numpy as np
from typing import List, Dict
from sklearn.metrics import roc_auc_score, f1_score

def _stack(y_true_list: List[np.ndarray], y_prob_list: List[np.ndarray]):
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    return y_true, y_prob

def compute_macro_auroc(y_true_list: List[np.ndarray], y_prob_list: List[np.ndarray]) -> float:
    y_true, y_prob = _stack(y_true_list, y_prob_list)
    C = y_true.shape[1]
    scores = []
    for c in range(C):
        if len(np.unique(y_true[:, c])) < 2:
            continue  # skip degenerate columns
        try:
            scores.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0

def compute_f1(y_true_list: List[np.ndarray], y_prob_list: List[np.ndarray], thr: float = 0.5) -> Dict[str,float]:
    y_true, y_prob = _stack(y_true_list, y_prob_list)
    y_pred = (y_prob >= thr).astype(int)
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
