"""Drug-protein fusion modules.

* ``concat``          - mean-pooled atom embeddings || protein vector,
* ``cross_attention`` - the protein vector queries the atoms with multi-head attention
  (the "multi-head attention layer" of the thesis architecture figure); its output is
  concatenated with the mean-pooled drug and the protein vector,
* ``add``             - element-wise sum (the fusion used in one of the original notebooks).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch


class ConcatFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_dim = 2 * dim

    def forward(self, h: torch.Tensor, batch: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.cat([global_mean_pool(h, batch, size=p.size(0)), p], dim=-1)


class AddFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_dim = dim

    def forward(self, h: torch.Tensor, batch: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(h, batch, size=p.size(0)) + p


class CrossAttentionFusion(nn.Module):
    """Protein-conditioned attention pooling over atoms."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = 3 * dim

    def forward(self, h: torch.Tensor, batch: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        dense, mask = to_dense_batch(h, batch, batch_size=p.size(0))  # [B, Nmax, dim], [B, Nmax]
        query = p.unsqueeze(1)  # [B, 1, dim]
        attended, _ = self.attn(query, dense, dense, key_padding_mask=~mask)
        attended = self.norm(attended.squeeze(1) + p)
        pooled = global_mean_pool(h, batch, size=p.size(0))
        return torch.cat([attended, pooled, p], dim=-1)


def build_fusion(kind: str, dim: int, num_heads: int = 4, dropout: float = 0.1) -> nn.Module:
    if kind == "concat":
        return ConcatFusion(dim)
    if kind == "add":
        return AddFusion(dim)
    if kind == "cross_attention":
        return CrossAttentionFusion(dim, num_heads, dropout)
    raise ValueError(f"unknown fusion '{kind}'")
