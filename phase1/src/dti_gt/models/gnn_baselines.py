"""Message-passing baselines (GCN / GAT / GIN) with the same input encoding as the transformer.

The GCN backbone corresponds to what the *original* research notebook actually trained
(``12 x GCNConv`` followed by a global attention read-out); it is kept as an ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINEConv

from dti_gt.data.featurize import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS, MAX_DEGREE
from dti_gt.models.graph_transformer import CategoricalEncoder, make_norm


class GNNEncoder(nn.Module):
    """Stack of GCN / GAT / GINE layers producing per-atom embeddings (``[N, dim]``)."""

    def __init__(
        self,
        kind: str = "gcn",
        dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_degree_encoding: bool = True,
        residual: bool = True,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.kind = kind
        self.residual = residual
        self.dropout = dropout
        self.atom_encoder = CategoricalEncoder(ATOM_FEATURE_DIMS, dim)
        self.degree_encoder = nn.Embedding(MAX_DEGREE + 1, dim) if use_degree_encoding else None
        self.bond_encoder = CategoricalEncoder(BOND_FEATURE_DIMS, dim) if kind == "gine" else None
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if kind == "gcn":
                conv = GCNConv(dim, dim)
            elif kind == "gat":
                conv = GATConv(dim, dim // num_heads, heads=num_heads, dropout=dropout)
            elif kind == "gine":
                conv = GINEConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)), edge_dim=dim)
            else:
                raise ValueError(f"unknown GNN kind '{kind}'")
            self.convs.append(conv)
            self.norms.append(make_norm(norm, dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        h = self.atom_encoder(x)
        if self.degree_encoder is not None:
            h = h + self.degree_encoder(degree.clamp(max=MAX_DEGREE))
        e = self.bond_encoder(edge_attr) if self.bond_encoder is not None else None
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, e) if self.kind == "gine" else conv(h, edge_index)
            h = F.dropout(F.relu(norm(h)), self.dropout, self.training)
            if self.residual:
                h = h + h_in
        return h
