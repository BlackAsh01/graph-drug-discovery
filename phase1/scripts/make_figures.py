"""Generate figures from ``results/`` into ``results/figures/``.

* ``ablation_bar.png``        - test MSE and CI per variant (mean +/- std over seeds),
* ``loss_curves.png``         - train / val MSE per epoch for the full-model runs,
* ``ablation_val_curves.png`` - validation MSE per epoch for every variant (seed 0),
* ``pred_vs_true.png``        - test-set scatter for the best full-model run,
* ``fulldata_loss_curve.png`` - loss curve of the full-data run if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_ablation import VARIANT_DESCRIPTIONS  # noqa: E402


def ablation_bar(results_csv: Path, out: Path) -> None:
    df = pd.read_csv(results_csv)
    order = [v for v in VARIANT_DESCRIPTIONS if v in set(df["variant"])] + sorted(set(df["variant"]) - set(VARIANT_DESCRIPTIONS))
    g = df.groupby("variant")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, label in zip(axes, ["test_mse", "test_ci"], ["Test MSE (lower is better)", "Test CI (higher is better)"]):
        means = [g.get_group(v)[key].mean() for v in order]
        stds = [g.get_group(v)[key].std(ddof=0) for v in order]
        colors = ["#1f77b4" if v == "full" else "#7f7f7f" for v in order]
        ax.bar(range(len(order)), means, yerr=stds, color=colors, capsize=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(label)
        lo, hi = min(means) - 3 * max(max(stds), 1e-3), max(means) + 3 * max(max(stds), 1e-3)
        ax.set_ylim(max(0, lo) if key == "test_mse" else lo, hi)
        ax.axhline(g.get_group("full")[key].mean() if "full" in g.groups else np.nan, color="#1f77b4", ls="--", lw=1)
        ax.grid(axis="y", alpha=0.3)
    n_seeds = df.groupby("variant")["seed"].nunique().max()
    fig.suptitle(f"Ablation study on KIBA (20 % stratified subset, ≤ 12 epochs, mean ± std over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def loss_curves(runs_dir: Path, pattern: str, out: Path, title: str) -> bool:
    files = sorted(runs_dir.glob(f"{pattern}/history.csv"))
    if not files:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for f in files:
        h = pd.read_csv(f)
        (line,) = ax.plot(h["epoch"], h["train_mse"], lw=1.2, label=f"{f.parent.name} train")
        ax.plot(h["epoch"], h["val_mse"], lw=1.2, ls="--", color=line.get_color(), label=f"{f.parent.name} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def variant_val_curves(runs_dir: Path, seed: int, out: Path) -> bool:
    """Validation MSE per epoch for every ablation variant (one seed), full model highlighted."""
    files = {f.parent.name[: -len(f"_seed{seed}")]: f for f in runs_dir.glob(f"*_seed{seed}/history.csv")}
    if not files:
        return False
    order = [v for v in VARIANT_DESCRIPTIONS if v in files] + sorted(set(files) - set(VARIANT_DESCRIPTIONS))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for v in order:
        h = pd.read_csv(files[v])
        ax.plot(h["epoch"], h["val_mse"], lw=2.2 if v == "full" else 1.1, color="#1f77b4" if v == "full" else None, label=v, marker="o" if v == "full" else None, ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE")
    ax.set_title(f"Ablation variants, seed {seed}: validation MSE per epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def pred_vs_true(runs_dir: Path, pattern: str, out: Path) -> bool:
    best, best_mse = None, float("inf")
    for m in runs_dir.glob(f"{pattern}/metrics.json"):
        d = json.loads(m.read_text())
        if d["test"]["mse"] < best_mse:
            best, best_mse = m.parent, d["test"]["mse"]
    if best is None or not (best / "test_predictions.csv").is_file():
        return False
    p = pd.read_csv(best / "test_predictions.csv")
    d = json.loads((best / "metrics.json").read_text())["test"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(p["y_true"], p["y_pred"], s=4, alpha=0.35)
    lo, hi = float(min(p.min())), float(max(p.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true KIBA score")
    ax.set_ylabel("predicted KIBA score")
    ax.set_title(f"{best.name}: MSE {d['mse']:.3f}, CI {d['ci']:.3f}, r {d['pearson']:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = results_dir / "runs"

    csv = results_dir / "ablation_results.csv"
    if csv.is_file():
        ablation_bar(csv, fig_dir / "ablation_bar.png")
        print("wrote", fig_dir / "ablation_bar.png")
    if loss_curves(runs, "full_seed*", fig_dir / "loss_curves.png", "Full model, subset protocol: train / val MSE"):
        print("wrote", fig_dir / "loss_curves.png")
    if variant_val_curves(runs, 0, fig_dir / "ablation_val_curves.png"):
        print("wrote", fig_dir / "ablation_val_curves.png")
    if pred_vs_true(runs, "full_seed*", fig_dir / "pred_vs_true.png"):
        print("wrote", fig_dir / "pred_vs_true.png")
    if loss_curves(runs, "full_model_fulldata*", fig_dir / "fulldata_loss_curve.png", "Full model, full KIBA: train / val MSE"):
        print("wrote", fig_dir / "fulldata_loss_curve.png")
    if pred_vs_true(runs, "full_model_fulldata*", fig_dir / "fulldata_pred_vs_true.png"):
        print("wrote", fig_dir / "fulldata_pred_vs_true.png")


if __name__ == "__main__":
    main()
