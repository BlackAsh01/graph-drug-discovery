"""Build a model from the ``model:`` section of a config."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..data.graph import HeteroGraph
from .baselines import MultiHotBaseline, NearestVisitBaseline
from .recommender import HGDRecommender


def build_model(cfg: Dict[str, Any], graph: HeteroGraph, drug_prior: Optional[torch.Tensor] = None) -> nn.Module:
    """Instantiate the model named by ``cfg['model']['name']``.

    Supported names: ``hgdr`` (main model), ``mlp``, ``lr`` (multi-hot
    baselines) and ``nearest`` (copy previous visit).
    """
    m = cfg["model"]
    name = m.get("name", "hgdr").lower()
    n_diag, n_proc, n_drug = graph.num_nodes["diag"], graph.num_nodes.get("proc", 0), graph.num_nodes["drug"]
    if name == "hgdr":
        return HGDRecommender(
            graph,
            dim=int(m.get("hidden_dim", 64)),
            gnn_layers=int(m.get("gnn_layers", 2)),
            gnn_type=m.get("gnn_type", "hetero"),
            conv=m.get("conv", "sage"),
            heads=int(m.get("heads", 4)),
            use_mol=bool(m.get("use_mol", True)),
            mol_layers=int(m.get("mol_layers", 2)),
            mol_conv=m.get("mol_conv", "gat"),
            use_proc=bool(m.get("use_proc", True)),
            use_history=bool(m.get("use_history", True)),
            use_ddi_edges=bool(m.get("use_ddi_edges", True)),
            use_coprescription_edges=bool(m.get("use_coprescription_edges", True)),
            dropout=float(m.get("dropout", 0.2)),
        )
    if name in ("mlp", "lr"):
        return MultiHotBaseline(
            n_diag, n_proc, n_drug,
            hidden_dim=int(m.get("hidden_dim", 256)) if name == "mlp" else 0,
            dropout=float(m.get("dropout", 0.3)),
            use_proc=bool(m.get("use_proc", True)),
            use_history=bool(m.get("use_history", True)),
        )
    if name == "nearest":
        if drug_prior is None:
            drug_prior = torch.zeros(n_drug)
        return NearestVisitBaseline(n_drug, drug_prior)
    raise ValueError(f"unknown model name '{name}'")
