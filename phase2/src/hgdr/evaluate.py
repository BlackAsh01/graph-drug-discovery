"""Evaluate a saved checkpoint on the validation and test splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from .train import _select_device, drug_prior_from_records, evaluate, load_data, make_loaders
from .models import build_model
from .utils.logging import get_logger
from .utils.seed import set_seed


def evaluate_checkpoint(cfg: Dict[str, Any], checkpoint: str | Path, threshold: float | None = None) -> Dict[str, Any]:
    """Load ``checkpoint`` into the model described by ``cfg`` and score val/test."""
    log = get_logger("hgdr.eval")
    set_seed(int(cfg.get("seed", 0)))
    data, graph, train, val, test = load_data(cfg)
    n_drug = graph.num_nodes["drug"]
    loaders = make_loaders(cfg, n_drug, train, val, test)
    device = _select_device(cfg, log)
    model = build_model(cfg, graph, drug_prior_from_records(train, n_drug)).to(device)
    state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state)
    thr = float(threshold if threshold is not None else cfg["train"].get("threshold", 0.5))
    ddi = graph.ddi_adj.numpy()
    out = {"val": evaluate(model, loaders[1], device, ddi, thr), "test": evaluate(model, loaders[2], device, ddi, thr),
           "checkpoint": str(checkpoint), "threshold": thr}
    log.info(f"val  : {out['val']}")
    log.info(f"test : {out['test']}")
    return out
