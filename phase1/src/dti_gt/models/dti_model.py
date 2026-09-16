"""End-to-end drug-target affinity model: drug encoder + protein encoder + fusion + MLP head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from dti_gt.models.fusion import build_fusion
from dti_gt.models.gnn_baselines import GNNEncoder
from dti_gt.models.graph_transformer import GraphTransformerEncoder
from dti_gt.models.protein_encoder import build_protein_encoder


class DTIModel(nn.Module):
    """Predict a scalar affinity from a batched molecular graph and a protein tensor."""

    def __init__(self, drug_encoder: nn.Module, protein_encoder: nn.Module, fusion: nn.Module, head_hidden: int, head_layers: int, dropout: float) -> None:
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.fusion = fusion
        layers, d_in = [], fusion.out_dim
        for _ in range(head_layers):
            layers += [nn.Linear(d_in, head_hidden), nn.LeakyReLU(), nn.Dropout(dropout)]
            d_in = head_hidden
        layers.append(nn.Linear(d_in, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, graphs: Batch, protein: torch.Tensor) -> torch.Tensor:
        h = self.drug_encoder(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.degree)
        p = self.protein_encoder(protein)
        z = self.fusion(h, graphs.batch, p)
        return self.head(z).squeeze(-1)


def build_drug_encoder(mcfg: dict) -> nn.Module:
    """Instantiate the drug encoder from the ``model`` section (``backbone: transformer|gcn|gat|gine``)."""
    backbone = mcfg.get("backbone", "transformer")
    common = dict(
        dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        num_heads=int(mcfg["num_heads"]),
        dropout=float(mcfg.get("dropout", 0.1)),
        use_degree_encoding=bool(mcfg.get("use_degree_encoding", True)),
        residual=bool(mcfg.get("residual", True)),
        norm=str(mcfg.get("norm", "batch")),
    )
    if backbone == "transformer":
        return GraphTransformerEncoder(
            input_dropout=float(mcfg.get("input_dropout", 0.0)),
            use_edge_features=bool(mcfg.get("use_edge_features", True)),
            self_loops=bool(mcfg.get("self_loops", True)),
            **common,
        )
    return GNNEncoder(kind=backbone, **common)


def build_model(cfg: Dict, protein_input_dim: int) -> DTIModel:
    """Build the full model from a resolved config."""
    mcfg, pcfg = cfg["model"], cfg["protein"]
    dim = int(mcfg["hidden_dim"])
    drug_encoder = build_drug_encoder(mcfg)
    protein_encoder = build_protein_encoder(pcfg, protein_input_dim, dim)
    fusion = build_fusion(mcfg.get("fusion", "cross_attention"), dim, int(mcfg.get("fusion_heads", 4)), float(mcfg.get("dropout", 0.1)))
    return DTIModel(drug_encoder, protein_encoder, fusion, int(mcfg.get("head_hidden", 256)), int(mcfg.get("head_layers", 3)), float(mcfg.get("head_dropout", 0.1)))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
