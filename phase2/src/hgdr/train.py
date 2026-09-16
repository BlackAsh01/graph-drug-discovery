"""Training / evaluation engine (config-driven).

Typical use::

    from hgdr.train import run_experiment
    metrics = run_experiment(cfg, run_name="full_seed0")

``run_experiment`` loads the processed data + graph, builds the model,
trains with early stopping on validation Jaccard, evaluates on the test
split and writes everything to ``results/runs/<run_name>/``:

* ``config.yaml``      – the fully-resolved config,
* ``history.csv``      – per-epoch train loss and validation metrics,
* ``metrics.json``     – final validation + test metrics, runtime, device,
* ``best.pt``          – best checkpoint (git-ignored),
* ``train.log``        – log file.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .config import config_get
from .data.dataset import VisitDataset, make_collate, move_batch
from .data.graph import HeteroGraph
from .data.mimic import ProcessedData
from .data.splits import split_records, subsample_records
from .models import build_model, recommendation_loss
from .utils.logging import close_logger, get_logger, save_json, setup_run_dir
from .utils.metrics import multi_label_metrics, topk_metrics
from .utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(cfg: Dict[str, Any]) -> Tuple[ProcessedData, HeteroGraph, list, list, list]:
    """Load processed records + graph and split by patient hash."""
    proc_dir = Path(cfg["data"]["processed_dir"])
    data = ProcessedData.load(proc_dir)
    graph = HeteroGraph.load(proc_dir / cfg["data"].get("graph_file", "graph.pt"))
    fr = cfg["data"].get("split_fractions", [0.8, 0.1, 0.1])
    records = subsample_records(data.records, float(cfg["data"].get("subset_fraction", 1.0)),
                                int(cfg["data"].get("subset_seed", 0)))
    train, val, test = split_records(records, fr, int(cfg["data"].get("split_seed", 0)))
    return data, graph, train, val, test


def make_loaders(cfg: Dict[str, Any], n_drug: int, train, val, test) -> Tuple[DataLoader, DataLoader, DataLoader]:
    bs = int(cfg["train"].get("batch_size", 128))
    max_hist = int(cfg["data"].get("max_history", 10))
    collate = make_collate(n_drug)
    mk = lambda recs, shuffle, b: DataLoader(VisitDataset(recs, max_hist), batch_size=b, shuffle=shuffle,  # noqa: E731
                                             collate_fn=collate, num_workers=0)
    return mk(train, True, bs), mk(val, False, bs * 2), mk(test, False, bs * 2)


def drug_prior_from_records(train, n_drug: int) -> torch.Tensor:
    cnt = torch.zeros(n_drug)
    n = 0
    for p in train:
        for v in p:
            cnt[torch.tensor(v["drug"], dtype=torch.long)] += 1
            n += 1
    return cnt / max(1, n)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    nodes = model.encode_graph()
    ys, ps = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch, nodes)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
    return np.concatenate(ys, 0), np.concatenate(ps, 0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, ddi_adj: np.ndarray,
             threshold: float = 0.5, topk: int = 10) -> Dict[str, float]:
    """Compute the full metric set on a loader."""
    y, p = predict(model, loader, device)
    out = multi_label_metrics(y, p, threshold=threshold, ddi_adj=ddi_adj)
    out.update(topk_metrics(y, p, k=topk))
    out["n_visits"] = int(y.shape[0])
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _select_device(cfg: Dict[str, Any], log) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        want = "cpu"
    return torch.device(want)


def _selection_score(val: Dict[str, float], metric: str) -> float:
    """Model-selection score from validation metrics.

    ``metric`` is a metric name (e.g. ``jaccard``, the SafeDrug convention) or
    a ``+``-joined sum of names (e.g. ``jaccard+prauc``, more stable on very
    small validation sets where Jaccard is quantised).
    """
    return float(sum(val[m.strip()] for m in metric.split("+")))


def train_model(
    cfg: Dict[str, Any],
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    ddi_adj: torch.Tensor,
    device: torch.device,
    run_dir: Path,
    log,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """Train with early stopping on validation Jaccard; returns best model + history."""
    tcfg = cfg["train"]
    train_loader, val_loader, _ = loaders
    epochs = int(tcfg.get("epochs", 30))
    patience = int(tcfg.get("patience", 5))
    ddi_w = float(tcfg.get("ddi_weight", 0.0))
    margin_w = float(tcfg.get("margin_weight", 0.0))
    threshold = float(tcfg.get("threshold", 0.5))
    clip = float(tcfg.get("grad_clip", 1.0))
    select_metric = str(tcfg.get("select_metric", "jaccard"))
    min_epochs = int(tcfg.get("min_epochs", 3))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(tcfg.get("lr", 1e-3)), weight_decay=float(tcfg.get("weight_decay", 1e-5)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=max(1, patience // 2))
    ddi_adj_dev = ddi_adj.to(device)
    ddi_np = ddi_adj.cpu().numpy()

    history: List[Dict[str, float]] = []
    best_score, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    n_params = sum(p.numel() for p in trainable)
    log.info(f"trainable parameters: {n_params:,}")
    if cfg["model"].get("name", "hgdr") == "nearest":
        epochs = 0  # parameter-free

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot, tot_bce, tot_ddi, nb = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch)  # graph re-encoded per step (gradients flow to the GNN)
            losses = recommendation_loss(logits, batch["y"], ddi_adj_dev, ddi_w, margin_w)
            losses["loss"].backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip)
            opt.step()
            tot += losses["loss"].item()
            tot_bce += losses["bce"].item()
            tot_ddi += losses["ddi"].item()
            nb += 1
        val = evaluate(model, val_loader, device, ddi_np, threshold)
        sched.step(val["jaccard"])
        rec = {"epoch": epoch, "train_loss": tot / max(1, nb), "train_bce": tot_bce / max(1, nb),
               "train_ddi": tot_ddi / max(1, nb), "epoch_time_s": time.time() - t0,
               "lr": opt.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        log.info(f"epoch {epoch:03d} | loss {rec['train_loss']:.4f} (bce {rec['train_bce']:.4f}, ddi {rec['train_ddi']:.5f}) "
                 f"| val jaccard {val['jaccard']:.4f} f1 {val['f1']:.4f} prauc {val['prauc']:.4f} "
                 f"ddi {val['ddi_rate']:.4f} #drugs {val['avg_drugs']:.1f} | {rec['epoch_time_s']:.1f}s")
        score = _selection_score(val, select_metric)
        rec["val_select_score"] = score
        if score > best_score:
            best_score, bad = score, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience and epoch >= min_epochs:
                log.info(f"early stopping (no val improvement for {patience} epochs)")
                break
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model.load_state_dict(best_state)
    return model, history


def run_experiment(cfg: Dict[str, Any], run_name: Optional[str] = None, results_dir: Optional[str] = None) -> Dict[str, Any]:
    """End-to-end: load data, train, evaluate, persist artefacts. Returns metrics dict."""
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    results_dir = Path(results_dir or cfg.get("results_dir", "results"))
    run_name = run_name or f"{cfg.get('name', 'run')}_seed{seed}"
    run_dir = setup_run_dir(results_dir, run_name)
    log = get_logger(f"hgdr.{run_name}", run_dir / "train.log")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    t_start = time.time()
    data, graph, train, val, test = load_data(cfg)
    n_drug = graph.num_nodes["drug"]
    log.info(f"data: {len(train)}/{len(val)}/{len(test)} train/val/test patients; "
             f"{sum(len(p) for p in train)}/{sum(len(p) for p in val)}/{sum(len(p) for p in test)} visits; "
             f"{graph.num_nodes}")
    loaders = make_loaders(cfg, n_drug, train, val, test)
    device = _select_device(cfg, log)
    log.info(f"device: {device}")

    def _build(dev: torch.device) -> nn.Module:
        prior = drug_prior_from_records(train, n_drug)
        return build_model(cfg, graph, prior).to(dev)

    model = _build(device)
    try:
        model, history = train_model(cfg, model, loaders, graph.ddi_adj, device, run_dir, log)
    except torch.cuda.OutOfMemoryError:  # graceful CPU fallback for the shared 4 GB GPU
        log.warning("CUDA out of memory -> retrying on CPU")
        del model
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = _build(device)
        model, history = train_model(cfg, model, loaders, graph.ddi_adj, device, run_dir, log)

    ddi_np = graph.ddi_adj.numpy()
    thr = float(cfg["train"].get("threshold", 0.5))
    val_metrics = evaluate(model, loaders[1], device, ddi_np, thr)
    test_metrics = evaluate(model, loaders[2], device, ddi_np, thr)
    # ground-truth DDI rate of the test prescriptions (reference point)
    y_test, _ = predict(model, loaders[2], device)
    from .utils.metrics import ddi_rate_score

    gt_ddi = ddi_rate_score(y_test, ddi_np)
    runtime = time.time() - t_start
    out = {
        "run_name": run_name,
        "config_name": cfg.get("name", "run"),
        "seed": seed,
        "model": cfg["model"].get("name", "hgdr"),
        "device": str(device),
        "runtime_s": runtime,
        "epochs_run": len(history),
        "best_epoch": int(max(history, key=lambda r: r["val_select_score"])["epoch"]) if history else 0,
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "val": val_metrics,
        "test": test_metrics,
        "test_ground_truth_ddi_rate": gt_ddi,
        "graph": {k: (int(v.shape[1]) if hasattr(v, "shape") else v) for k, v in
                  {"|".join(k): v for k, v in graph.edge_index.items()}.items()},
        "num_nodes": graph.num_nodes,
    }
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_json(out, run_dir / "metrics.json")
    log.info(f"TEST | jaccard {test_metrics['jaccard']:.4f} | prauc {test_metrics['prauc']:.4f} | "
             f"f1 {test_metrics['f1']:.4f} | ddi {test_metrics['ddi_rate']:.4f} (gt {gt_ddi:.4f}) | "
             f"#drugs {test_metrics['avg_drugs']:.2f} | P@10 {test_metrics['precision@10']:.4f} | "
             f"{runtime/60:.1f} min")
    if not bool(cfg.get("keep_checkpoint", True)):
        (run_dir / "best.pt").unlink(missing_ok=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    close_logger(log)
    return out
