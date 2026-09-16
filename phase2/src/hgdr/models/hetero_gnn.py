"""Heterogeneous / homogeneous GNN encoders over the medical-entity graph.

``HeteroGNN`` is a relation-aware encoder: one ``SAGEConv`` (or
``GATConv``) per edge type inside PyG's ``HeteroConv``, summed per
destination node type, with residual connections and LayerNorm.  It is the
cleaned-up successor of the ``LightGCNHetero`` class in the original
notebooks (which used ``HeteroConv(SAGEConv)`` with learned root weights).

``HomoGNN`` is the ablation counterpart: all node types are stacked into a
single index space and all edge types are merged into one edge set, so
type information is discarded ("homogeneous graph" ablation).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

EdgeKey = Tuple[str, str, str]


def _make_conv(kind: str, dim: int, heads: int, dropout: float):
    if kind == "sage":
        return SAGEConv((dim, dim), dim, aggr="mean")
    if kind == "gat":
        return GATConv((dim, dim), dim // heads, heads=heads, dropout=dropout, add_self_loops=False)
    raise ValueError(f"unknown conv kind '{kind}'")


class HeteroGNN(nn.Module):
    """Relation-aware message passing over ``x_dict`` / ``edge_index_dict``."""

    def __init__(self, node_types: List[str], edge_types: List[EdgeKey], dim: int, num_layers: int = 2,
                 conv: str = "sage", heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({et: _make_conv(conv, dim, heads, dropout) for et in self.edge_types},
                                         aggr="sum"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(dim) for nt in self.node_types}))

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[EdgeKey, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = dict(x_dict)
        active = {k: v for k, v in edge_index_dict.items() if k in self.edge_types and v.numel() > 0}
        for conv, norms in zip(self.convs, self.norms):
            out = conv(h, active) if active else {}
            new = {}
            for nt, x in h.items():
                m = out.get(nt)
                if m is None:
                    new[nt] = x
                else:
                    new[nt] = norms[nt](x + F.dropout(F.relu(m), self.dropout, self.training))
            h = new
        return h


class HomoGNN(nn.Module):
    """Type-agnostic GNN used for the *homogeneous graph* ablation."""

    def __init__(self, node_types: List[str], dim: int, num_layers: int = 2, conv: str = "sage",
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.node_types = list(node_types)
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv == "sage":
                self.convs.append(SAGEConv(dim, dim, aggr="mean"))
            else:
                self.convs.append(GATConv(dim, dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(dim))

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[EdgeKey, torch.Tensor]) -> Dict[str, torch.Tensor]:
        offsets: Dict[str, int] = {}
        xs = []
        off = 0
        for nt in self.node_types:
            offsets[nt] = off
            xs.append(x_dict[nt])
            off += x_dict[nt].shape[0]
        x = torch.cat(xs, 0)
        eis = []
        for (s, _, d), ei in edge_index_dict.items():
            if ei.numel() == 0 or s not in offsets or d not in offsets:
                continue
            eis.append(torch.stack([ei[0] + offsets[s], ei[1] + offsets[d]], 0))
        edge_index = torch.cat(eis, 1) if eis else torch.zeros((2, 0), dtype=torch.long, device=x.device)
        h = x
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index)
            h = norm(h + F.dropout(F.relu(m), self.dropout, self.training))
        out = {}
        for nt in self.node_types:
            n = x_dict[nt].shape[0]
            out[nt] = h[offsets[nt]:offsets[nt] + n]
        return out
