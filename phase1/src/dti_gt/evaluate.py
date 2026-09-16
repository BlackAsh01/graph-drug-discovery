"""Evaluate a saved run (``results/runs/<name>/best.pt``) on the val / test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from dti_gt.data.dataset import build_dataloaders
from dti_gt.models.dti_model import build_model
from dti_gt.train import TargetScaler, evaluate_loader, select_device
from dti_gt.utils.config import load_config
from dti_gt.utils.seed import set_seed


def evaluate_run(run_dir: str | Path, root: Path, splits=("val", "test"), device: str = "auto") -> Dict[str, Dict[str, float]]:
    """Reload the best checkpoint of a run and recompute metrics on the requested splits."""
    run_dir = Path(run_dir)
    ckpt = run_dir / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found (checkpoints are git-ignored; re-train with scripts/train.py)")
    cfg = load_config(run_dir / "config.yaml")
    set_seed(int(cfg["training"]["seed"]))
    dev = select_device(device)
    loaders, info = build_dataloaders(cfg, root)
    model = build_model(cfg, int(info["protein_input_dim"])).to(dev)
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(state["model_state"])
    ts = state.get("target_scaler", {"mean": 0.0, "std": 1.0})
    scaler = TargetScaler(ts["mean"], ts["std"])
    out = {s: evaluate_loader(model, loaders[s], dev, scaler) for s in splits}
    (run_dir / "eval_metrics.json").write_text(json.dumps(out, indent=2))
    return out
