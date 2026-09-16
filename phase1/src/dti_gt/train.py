"""Configuration-driven training loop.

Usage from Python::

    from dti_gt.train import train_from_config
    metrics = train_from_config(cfg, root=Path("."), run_name="my_run")

Artifacts are written to ``results/runs/<run_name>/``:
``config.yaml`` (resolved config), ``history.csv`` (per-epoch losses / val metrics),
``metrics.json`` (final val + test metrics, runtime, parameter count), ``test_predictions.csv``,
``train.log`` and ``best.pt`` (best-validation checkpoint, git-ignored).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dti_gt.data.dataset import build_dataloaders
from dti_gt.models.dti_model import build_model, count_parameters
from dti_gt.utils.config import save_config
from dti_gt.utils.logging import get_logger
from dti_gt.utils.metrics import regression_metrics
from dti_gt.utils.seed import set_seed


def select_device(requested: str = "auto") -> torch.device:
    """Resolve ``'auto' | 'cuda' | 'cpu'`` to a device, falling back to CPU when CUDA is absent."""
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[train] CUDA requested but unavailable - using CPU")
    return torch.device("cpu")


def _to_device(batch, device: torch.device):
    graphs, prot, y = batch
    return graphs.to(device, non_blocking=True), prot.to(device, non_blocking=True), y.to(device, non_blocking=True)


class TargetScaler:
    """Optional z-scoring of the regression target (fitted on the training split only).

    The network predicts ``(y - mean) / std``; predictions are mapped back to KIBA units
    before any metric is computed, so all reported numbers are in the original scale.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean, self.std = float(mean), float(std) if std > 0 else 1.0

    @classmethod
    def fit(cls, y: torch.Tensor, enabled: bool) -> "TargetScaler":
        return cls(y.mean().item(), y.std().item()) if enabled else cls()

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, scaler: Optional[TargetScaler] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return ``(y_true, y_pred)`` as NumPy arrays (original units)."""
    model.eval()
    scaler = scaler or TargetScaler()
    ys, ps = [], []
    for batch in loader:
        graphs, prot, y = _to_device(batch, device)
        ps.append(scaler.inverse(model(graphs, prot).float()).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, scaler: Optional[TargetScaler] = None) -> Dict[str, float]:
    """Metrics for a loader (see :func:`dti_gt.utils.metrics.regression_metrics`)."""
    y, p = predict(model, loader, device, scaler)
    return regression_metrics(y, p)


def _make_optimizer(model: nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    """AdamW without weight decay on biases / norm parameters. Uses the fused CUDA kernel when
    possible - the model has ~350 small parameter tensors, so the default (for-each) implementation
    spends more time in Python/launch overhead than in the actual update."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    on_cuda = all(p.is_cuda for p in decay + no_decay)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(tcfg.get("weight_decay", 1e-5))}, {"params": no_decay, "weight_decay": 0.0}],
        lr=float(tcfg["lr"]),
        fused=on_cuda,
    )


def train_from_config(cfg: dict, root: Path, run_name: Optional[str] = None, loaders: Optional[Dict[str, DataLoader]] = None, data_info: Optional[dict] = None) -> Dict[str, object]:
    """Train and evaluate one model. Returns the content of ``metrics.json``.

    Args:
        cfg: resolved configuration dictionary (see ``configs/default.yaml``).
        root: repository root (paths inside ``cfg`` are relative to it).
        run_name: folder name under ``results/runs``; defaults to ``cfg['run_name']``.
        loaders / data_info: optionally pre-built loaders (used by the ablation driver to
            avoid re-featurising the data for each seed).
    """
    tcfg = cfg["training"]
    run_name = run_name or cfg.get("run_name") or "run"
    cfg = {**cfg, "run_name": run_name}  # the saved config.yaml names the run it actually belongs to
    run_dir = root / cfg.get("results_dir", "results") / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"dti_gt.{run_name}", run_dir / "train.log")

    set_seed(int(tcfg["seed"]), bool(tcfg.get("deterministic", True)))
    device = select_device(str(tcfg.get("device", "auto")))
    save_config(cfg, run_dir / "config.yaml")

    if loaders is None:
        loaders, data_info = build_dataloaders(cfg, root)
    assert data_info is not None
    logger.info("data: %s", json.dumps(data_info))

    model = build_model(cfg, int(data_info["protein_input_dim"])).to(device)
    n_params = count_parameters(model)
    logger.info("model: %s backbone, %d trainable parameters, device=%s", cfg["model"].get("backbone", "transformer"), n_params, device)
    scaler_y = TargetScaler.fit(loaders["train"].dataset.y, bool(tcfg.get("normalize_target", True)))
    logger.info("target scaler: %s", json.dumps(scaler_y.state_dict()))

    optimizer = _make_optimizer(model, tcfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(tcfg.get("lr_factor", 0.5)), patience=int(tcfg.get("lr_patience", 5))
    )
    loss_fn = nn.MSELoss()
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_epochs = int(tcfg["epochs"])
    patience = int(tcfg.get("early_stopping_patience", 10))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    monitor = tcfg.get("monitor", "mse")

    history_path = run_dir / "history.csv"
    best_metric, best_epoch, bad_epochs = float("inf"), 0, 0
    best_path = run_dir / "best.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    with open(history_path, "w", newline="", encoding="utf-8") as f_hist:
        writer: Optional[csv.DictWriter] = None
        for epoch in range(1, max_epochs + 1):
            model.train()
            t_epoch = time.time()
            running, n_seen = 0.0, 0
            for batch in loaders["train"]:
                graphs, prot, y = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        pred = model(graphs, prot)
                        loss = loss_fn(pred.float(), scaler_y.transform(y))
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                except torch.cuda.OutOfMemoryError:  # pragma: no cover - hardware dependent
                    if device.type != "cuda":
                        raise
                    logger.warning("CUDA OOM at epoch %d - moving this run to CPU", epoch)
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    model = model.to(device)
                    optimizer = _make_optimizer(model, tcfg)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(tcfg.get("lr_factor", 0.5)), patience=int(tcfg.get("lr_patience", 5)))
                    use_amp = False
                    scaler = torch.amp.GradScaler("cuda", enabled=False)
                    continue
                running += loss.item() * y.size(0)
                n_seen += y.size(0)
            train_loss = running / max(n_seen, 1) * scaler_y.std**2  # report in original units

            val_metrics = evaluate_loader(model, loaders["val"], device, scaler_y)
            scheduler.step(val_metrics[monitor])
            row = {"epoch": epoch, "train_mse": train_loss, "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": time.time() - t_epoch}
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            if writer is None:
                writer = csv.DictWriter(f_hist, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f_hist.flush()
            logger.info(
                "epoch %3d/%d | train_mse %.4f | val_mse %.4f | val_ci %.4f | val_pearson %.4f | lr %.2e | %.1fs",
                epoch, max_epochs, train_loss, val_metrics["mse"], val_metrics["ci"], val_metrics["pearson"], row["lr"], row["epoch_time_s"],
            )

            if val_metrics[monitor] < best_metric - 1e-6:
                best_metric, best_epoch, bad_epochs = val_metrics[monitor], epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "config": cfg, "target_scaler": scaler_y.state_dict()}, best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                    break
            if device.type == "cuda":
                torch.cuda.empty_cache()

    train_time = time.time() - t_start
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    val_metrics = evaluate_loader(model, loaders["val"], device, scaler_y)
    y_test, p_test = predict(model, loaders["test"], device, scaler_y)
    test_metrics = regression_metrics(y_test, p_test)
    with open(run_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "y_pred"])
        w.writerows(zip(y_test.tolist(), p_test.tolist()))

    result = {
        "run_name": run_name,
        "variant": cfg.get("variant", run_name),
        "seed": int(tcfg["seed"]),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train_time_s": round(train_time, 1),
        "n_parameters": n_params,
        "device": device.type,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None,
        "target_scaler": scaler_y.state_dict(),
        "data": data_info,
        "val": val_metrics,
        "test": test_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    logger.info("TEST | mse %.4f | rmse %.4f | ci %.4f | rm2 %.4f | pearson %.4f | (%.0fs)", test_metrics["mse"], test_metrics["rmse"], test_metrics["ci"], test_metrics["rm2"], test_metrics["pearson"], train_time)
    if not bool(tcfg.get("keep_checkpoint", True)):
        best_path.unlink(missing_ok=True)
    if device.type == "cuda":
        del model, optimizer
        torch.cuda.empty_cache()
    return result
