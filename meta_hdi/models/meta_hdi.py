# meta_hdi/models/meta_hdi.py
# Minimal Meta-HDI model: GCN (optional) → BiLSTM → node-level attention → path-level attention → classifier
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_norm_adj(num_nodes: int, edges: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build symmetric normalized adjacency (A_hat = D^{-1/2}(A+I)D^{-1/2}) as a sparse tensor.
    edges: LongTensor [2, E] with (src, dst)
    """
    # add self loops
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long).unsqueeze(0).repeat(2,1)
    ei = torch.cat([edges, self_loops], dim=1)  # [2, E+N]

    # degrees
    ones = torch.ones(ei.size(1), device=device)
    deg = torch.zeros(num_nodes, device=device).scatter_add_(0, ei[0], ones)
    deg = deg + torch.zeros(num_nodes, device=device).scatter_add_(0, ei[1], ones) - ones[:num_nodes]
    deg = deg.clamp(min=1.0)
    deg_inv_sqrt = torch.pow(deg, -0.5)

    # weights per edge
    src, dst = ei[0], ei[1]
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    A = torch.sparse_coo_tensor(ei, norm, size=(num_nodes, num_nodes), device=device)
    return A.coalesce()

class SimpleGCN(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, edges: Optional[list] = None, n_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb = nn.Embedding(num_nodes, emb_dim)
        self.layers = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])
        self._edges_list = edges if edges is not None else []
        self.register_buffer("_adj_idx", torch.empty(2,0, dtype=torch.long), persistent=False)
        self.register_buffer("_adj_val", torch.empty(0), persistent=False)

    def _ensure_adj(self, device: torch.device):
        if self._edges_list is None or len(self._edges_list) == 0:
            return None
        if self._adj_idx.numel() == 0:
            ei = torch.tensor(self._edges_list, dtype=torch.long, device=device).t()  # [2, E]
            A = build_norm_adj(self.num_nodes, ei, device=device)
            self._adj_idx = A.indices()
            self._adj_val = A.values()

    def forward(self) -> torch.Tensor:
        X = self.emb.weight  # [N, D]
        if self._edges_list is None or len(self._edges_list) == 0:
            return X
        self._ensure_adj(self.emb.weight.device)
        A = torch.sparse_coo_tensor(self._adj_idx, self._adj_val, size=(self.num_nodes, self.num_nodes), device=self.emb.weight.device)
        H = X
        for lin in self.layers:
            H = torch.sparse.mm(A, H)
            H = lin(H)
            H = F.relu(H, inplace=True)
        return H

class TypeEmbedding(nn.Module):
    def __init__(self, num_types: int, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(num_types, dim)
    def forward(self, type_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(type_idx)

class IngredientBias(nn.Module):
    """Adds extra attention logit to nodes of a target type (e.g., compound)."""
    def __init__(self, dim: int, target_type_id: int = 1):
        super().__init__()
        self.lin = nn.Linear(dim, 1, bias=False)
        self.target_type_id = target_type_id
    def forward(self, H: torch.Tensor, type_idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        bias = self.lin(H).squeeze(-1)  # [B*Lp, L]
        mask = (type_idx == self.target_type_id)
        return logits + bias * mask

class PathAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
    def forward(self, paths: torch.Tensor, path_mask: torch.Tensor):
        scores = self.w(paths).squeeze(-1)  # [B, Lp]
        scores = scores.masked_fill(~path_mask, -1e9)
        beta = torch.softmax(scores, dim=-1)
        pair = torch.bmm(beta.unsqueeze(1), paths).squeeze(1)  # [B, D]
        return pair, beta

class MetaHDI(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_types: int,
        num_labels: int,
        emb_dim: int = 128,
        type_emb_dim: int = 8,
        edges: Optional[list] = None,
        use_compound_bias: bool = True,
        compound_type_id: int = 1,
        lstm_hidden: int = 128,
        freeze_gcn: bool = False,
    ):
        super().__init__()
        self.gcn = SimpleGCN(num_nodes=num_nodes, emb_dim=emb_dim, edges=edges, n_layers=2)
        self.type_emb = TypeEmbedding(num_types=num_types, dim=type_emb_dim)

        self.lstm = nn.LSTM(
            input_size=emb_dim + type_emb_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * lstm_hidden, lstm_hidden)

        self.node_w = nn.Linear(lstm_hidden, 1, bias=False)  # node-level attention logits
        self.use_compound_bias = use_compound_bias
        if use_compound_bias:
            self.compound_bias = IngredientBias(lstm_hidden, target_type_id=compound_type_id)

        self.path_attn = PathAttention(lstm_hidden)
        self.cls = nn.Linear(lstm_hidden, num_labels)

        if freeze_gcn:
            for p in self.gcn.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def load_pretrained_node_emb(self, weight: torch.Tensor):
        if weight.shape != self.gcn.emb.weight.shape:
            raise ValueError(f"pretrained_node_emb shape {weight.shape} != {self.gcn.emb.weight.shape}")
        self.gcn.emb.weight.copy_(weight)

    def forward(self, batch: dict) -> dict:
        node_mat = self.gcn()  # [N, D]

        X_paths = batch["paths"]      # [B, Lp, L]
        T_paths = batch["types"]      # [B, Lp, L]
        mask = batch["mask"].to(dtype=torch.bool)       # [B, Lp, L]
        path_mask = batch["path_mask"].to(dtype=torch.bool)  # [B, Lp]

        B, Lp, L = X_paths.shape
        D = node_mat.size(1)

        X_flat = X_paths.view(-1, L)     # [B*Lp, L]
        T_flat = T_paths.view(-1, L)     # [B*Lp, L]
        M_flat = mask.view(-1, L)        # [B*Lp, L]

        H_nodes = node_mat.index_select(0, X_flat.view(-1)).view(B*Lp, L, D)
        H_types = self.type_emb(T_flat)
        H = torch.cat([H_nodes, H_types], dim=-1)  # [B*Lp, L, D+Te]

        H_lstm, _ = self.lstm(H)                  # [B*Lp, L, 2H]
        H_lstm = self.proj(H_lstm)                # [B*Lp, L, H]

        logits = self.node_w(H_lstm).squeeze(-1)  # [B*Lp, L]
        if hasattr(self, "compound_bias") and self.use_compound_bias:
            logits = self.compound_bias(H_lstm, T_flat, logits)
        logits = logits.masked_fill(~M_flat, -1e9)
        alpha = torch.softmax(logits, dim=-1)     # [B*Lp, L]
        path_vec = torch.bmm(alpha.unsqueeze(1), H_lstm).squeeze(1)  # [B*Lp, H]
        path_vec = path_vec.view(B, Lp, -1)       # [B, Lp, H]

        pair_vec, beta = self.path_attn(path_vec, path_mask)         # [B, H], [B, Lp]
        out_logits = self.cls(pair_vec)                               # [B, C]

        return {
            "logits": out_logits,
            "alpha_node": alpha.view(B, Lp, L),
            "beta_path": beta,
        }
