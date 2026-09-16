"""Edge-aware graph transformer for molecular graphs (pure PyTorch / PyG, no DGL).

The layer follows the *Graph Transformer with edge features* of Dwivedi & Bresson (2021) as
adapted by GraphormerDTI: attention is restricted to bonded atom pairs (plus self loops),
attention logits are biased by a learned projection of the bond features, and the bond
representation is updated alongside the atom representation. Graphormer's *centrality
(degree) encoding* is added to the input atom embedding.

Each component is switchable so it can be ablated:

* ``use_edge_features`` - bond-feature bias in attention and edge-channel updates,
* ``use_degree_encoding`` - centrality encoding,
* ``residual`` / ``norm`` - skip connections and Batch/Layer normalisation,
* ``num_layers`` / ``num_heads`` - depth and heads.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, scatter, softmax

from dti_gt.data.featurize import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS, MAX_DEGREE, SELF_LOOP_BOND_TYPE


class CategoricalEncoder(nn.Module):
    """Sum of per-column embeddings for integer feature matrices (OGB ``AtomEncoder`` style)."""

    def __init__(self, feature_dims: List[int], dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, dim) for n in feature_dims])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x[:, i])
        return out


def make_norm(kind: str, dim: int) -> nn.Module:
    """``'batch'`` -> BatchNorm1d, ``'layer'`` -> LayerNorm, ``'none'`` -> Identity."""
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind in ("none", None, False):
        return nn.Identity()
    raise ValueError(f"unknown norm '{kind}'")


class EdgeAwareMultiHeadAttention(nn.Module):
    """Sparse multi-head attention over graph edges with an additive edge-feature bias."""

    def __init__(self, dim: int, num_heads: int, use_edge_features: bool, use_bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_edge_features = use_edge_features
        self.Q = nn.Linear(dim, dim, bias=use_bias)
        self.K = nn.Linear(dim, dim, bias=use_bias)
        self.V = nn.Linear(dim, dim, bias=use_bias)
        self.E = nn.Linear(dim, dim, bias=use_bias) if use_edge_features else None

    def forward(self, h: torch.Tensor, e: Optional[torch.Tensor], edge_index: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        src, dst = edge_index
        n = h.size(0)
        q = self.Q(h).view(n, self.num_heads, self.head_dim)
        k = self.K(h).view(n, self.num_heads, self.head_dim)
        v = self.V(h).view(n, self.num_heads, self.head_dim)

        # implicit attention: element-wise K_src * Q_dst  (GraphormerDTI's src_dot_dst)
        score = k[src] * q[dst] / math.sqrt(self.head_dim)  # [E, H, d]
        if self.use_edge_features and e is not None:
            score = score + self.E(e).view(-1, self.num_heads, self.head_dim)  # explicit edge bias
        e_out = score.reshape(-1, self.num_heads * self.head_dim) if self.use_edge_features else None

        logits = score.sum(-1).clamp(-5.0, 5.0)  # [E, H]
        attn = softmax(logits, dst, num_nodes=n)  # per-destination softmax, per head
        msg = v[src] * attn.unsqueeze(-1)  # [E, H, d]
        h_out = scatter(msg, dst, dim=0, dim_size=n, reduce="sum")
        return h_out.reshape(n, self.num_heads * self.head_dim), e_out


class GraphTransformerLayer(nn.Module):
    """One transformer block: attention -> O -> (residual, norm) -> FFN -> (residual, norm)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        residual: bool = True,
        norm: str = "batch",
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        self.use_edge_features = use_edge_features
        self.residual = residual
        self.dropout = dropout
        self.attention = EdgeAwareMultiHeadAttention(dim, num_heads, use_edge_features)
        self.O_h = nn.Linear(dim, dim)
        self.norm1_h = make_norm(norm, dim)
        self.ffn_h = nn.Sequential(nn.Linear(dim, dim * ffn_multiplier), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * ffn_multiplier, dim))
        self.norm2_h = make_norm(norm, dim)
        if use_edge_features:
            self.O_e = nn.Linear(dim, dim)
            self.norm1_e = make_norm(norm, dim)
            self.ffn_e = nn.Sequential(nn.Linear(dim, dim * ffn_multiplier), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * ffn_multiplier, dim))
            self.norm2_e = make_norm(norm, dim)

    def forward(self, h: torch.Tensor, e: Optional[torch.Tensor], edge_index: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h_in, e_in = h, e
        h_attn, e_attn = self.attention(h, e, edge_index)

        h = self.O_h(F.dropout(h_attn, self.dropout, self.training))
        if self.residual:
            h = h_in + h
        h = self.norm1_h(h)
        h2 = self.ffn_h(h)
        h = self.norm2_h(h + h2 if self.residual else h2)

        if self.use_edge_features and e is not None:
            e = self.O_e(F.dropout(e_attn, self.dropout, self.training))
            if self.residual:
                e = e_in + e
            e = self.norm1_e(e)
            e2 = self.ffn_e(e)
            e = self.norm2_e(e + e2 if self.residual else e2)
        return h, e


class GraphTransformerEncoder(nn.Module):
    """Stack of :class:`GraphTransformerLayer` producing per-atom embeddings.

    Args:
        dim: hidden size.
        num_layers: number of transformer blocks.
        num_heads: attention heads per block.
        dropout: dropout inside blocks.
        input_dropout: dropout on the input embedding.
        use_edge_features: enable bond-feature bias / edge channel.
        use_degree_encoding: add Graphormer centrality (degree) embedding to atoms.
        residual: skip connections.
        norm: ``'batch'``, ``'layer'`` or ``'none'``.
        self_loops: add self loops (bond type 0) so each atom attends to itself.
    """

    def __init__(
        self,
        dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
        use_edge_features: bool = True,
        use_degree_encoding: bool = True,
        residual: bool = True,
        norm: str = "batch",
        self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_edge_features = use_edge_features
        self.use_degree_encoding = use_degree_encoding
        self.self_loops = self_loops
        self.atom_encoder = CategoricalEncoder(ATOM_FEATURE_DIMS, dim)
        self.degree_encoder = nn.Embedding(MAX_DEGREE + 1, dim) if use_degree_encoding else None
        self.bond_encoder = CategoricalEncoder(BOND_FEATURE_DIMS, dim) if use_edge_features else None
        self.input_dropout = nn.Dropout(input_dropout)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(dim, num_heads, dropout, use_edge_features, residual, norm) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if self.self_loops:
            loop_attr = torch.zeros(1, edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device)
            loop_attr[0, 0] = SELF_LOOP_BOND_TYPE
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=loop_attr.squeeze(0), num_nodes=n)
        h = self.atom_encoder(x)
        if self.degree_encoder is not None:
            h = h + self.degree_encoder(degree.clamp(max=MAX_DEGREE))
        h = self.input_dropout(h)
        e = self.bond_encoder(edge_attr) if self.bond_encoder is not None else None
        for layer in self.layers:
            h, e = layer(h, e, edge_index)
        return h
