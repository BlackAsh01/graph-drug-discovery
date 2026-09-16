#!/usr/bin/env python
"""Step 6 - Figures from the ablation results and run histories.

Produces (PNG, < 1 MB each) in ``results/figures/``:

* ``ablation_bar.png``      - Jaccard / PR-AUC / F1 per variant (mean ± std),
* ``ddi_vs_jaccard.png``    - DDI-rate vs Jaccard trade-off scatter,
* ``training_curves.png``   - train loss + validation Jaccard for the full model (all seeds),
* ``graph_schema.png``      - schematic of the heterogeneous graph.

Example::

    python scripts/make_figures.py --results_dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _bootstrap  # noqa: F401,E402
from run_ablation import DESCRIPTIONS, VARIANTS  # noqa: E402


def ablation_bar(summ: pd.DataFrame, out: Path) -> None:
    metrics = [("jaccard", "Jaccard ↑"), ("prauc", "PR-AUC ↑"), ("f1", "F1 ↑"), ("ddi_rate", "DDI rate ↓")]
    order = [v for v in VARIANTS if v in set(summ["variant"])]
    summ = summ.set_index("variant").loc[order]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6), sharey=False)
    colors = ["#1f77b4" if not v.startswith("baseline") else "#7f7f7f" for v in order]
    colors = ["#d62728" if v == "full" else c for v, c in zip(order, colors)]
    for ax, (key, label) in zip(axes, metrics):
        m, s = summ[f"test_{key}_mean"].values, summ[f"test_{key}_std"].values
        y = np.arange(len(order))
        ax.barh(y, m, xerr=s, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(order if ax is axes[0] else [""] * len(order))
        ax.invert_yaxis()
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.3)
        lo = max(0.0, np.nanmin(m - s) - 0.05 * np.nanmax(m))
        ax.set_xlim(lo, np.nanmax(m + s) * 1.03)
    fig.suptitle("Ablation study (test split, mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def ddi_vs_jaccard(summ: pd.DataFrame, gt_ddi: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for _, r in summ.iterrows():
        v = r["variant"]
        marker = "s" if v.startswith("baseline") else ("*" if v == "full" else "o")
        ax.errorbar(r["test_ddi_rate_mean"], r["test_jaccard_mean"], xerr=r["test_ddi_rate_std"],
                    yerr=r["test_jaccard_std"], fmt=marker, ms=12 if v == "full" else 7, capsize=2,
                    label=f"{v}: {DESCRIPTIONS.get(v, '')}")
        ax.annotate(v, (r["test_ddi_rate_mean"], r["test_jaccard_mean"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.axvline(gt_ddi, ls="--", color="gray", lw=1)
    ax.text(gt_ddi, ax.get_ylim()[0], f" ground-truth DDI rate = {gt_ddi:.3f}", rotation=90, va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("DDI rate of recommended sets (lower is safer)")
    ax.set_ylabel("Jaccard (higher is more accurate)")
    ax.set_title("Safety / accuracy trade-off across variants")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def training_curves(runs_dir: Path, variant: str, out: Path) -> bool:
    hist_files = sorted(runs_dir.glob(f"{variant}_seed*/history.csv"))
    if not hist_files:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for hf in hist_files:
        h = pd.read_csv(hf)
        seed = hf.parent.name.split("seed")[-1]
        axes[0].plot(h["epoch"], h["train_loss"], label=f"seed {seed}")
        axes[1].plot(h["epoch"], h["val_jaccard"], label=f"seed {seed}")
        axes[2].plot(h["epoch"], h["val_ddi_rate"], label=f"seed {seed}")
    axes[0].set_title("training loss (BCE + DDI penalty)")
    axes[1].set_title("validation Jaccard")
    axes[2].set_title("validation DDI rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Training curves - {variant}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _box(ax, xy, w, h, text, fc, fontsize=8.5, lw=1.0):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec="black",
                                lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def _arrow(ax, p, q, text=None, style="-|>", rad=0.0, fontsize=7.5, offset=(0, 0.02)):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle=style, color="black", lw=1.0, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text, ha="center", va="center",
                fontsize=fontsize, bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.0), zorder=4)


def graph_schema(out: Path) -> None:
    """Architecture figure: heterogeneous graph schema (left) and the HGDR model (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.0, 1.15]})
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---------------- left: graph schema ----------------
    ax1.set_title("(a) Heterogeneous entity graph (built from TRAIN admissions only)", fontsize=10)
    _box(ax1, (0.04, 0.66), 0.30, 0.16, "Diagnosis nodes\nICD-9 codes (3,868)", "#ffd59e")
    _box(ax1, (0.04, 0.18), 0.30, 0.16, "Procedure nodes\nICD-9 / MetaVision items (115)", "#b5e7a0")
    _box(ax1, (0.55, 0.42), 0.30, 0.20, "Drug nodes\nnormalised names (606)\nPubChem CID + SMILES", "#a9cff5", lw=1.6)
    _box(ax1, (0.55, 0.02), 0.40, 0.14, "TWOSIDES DDI pairs\n(top-40 side effects)", "#f4c2c2")
    _box(ax1, (0.55, 0.82), 0.40, 0.14, "Molecular graphs (RDKit)\natoms = nodes, bonds = edges", "#e0e0e0")
    _arrow(ax1, (0.34, 0.74), (0.55, 0.58), "diag -treated_by-> drug\n(co-occurrence >= 5, top-20 per node)", style="<|-|>")
    _arrow(ax1, (0.34, 0.26), (0.55, 0.46), "proc -followed_by-> drug", style="<|-|>", offset=(0, -0.03))
    _arrow(ax1, (0.72, 0.16), (0.72, 0.42), "drug -interacts-> drug", offset=(0.14, 0))
    _arrow(ax1, (0.72, 0.82), (0.72, 0.62), "SMILES per drug", offset=(0.12, 0))
    ax1.annotate("", xy=(0.86, 0.60), xytext=(0.86, 0.44), zorder=1,
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, connectionstyle="arc3,rad=-1.8"))
    ax1.text(0.97, 0.52, "drug -coprescribed-> drug", fontsize=7.5, rotation=90, ha="center", va="center")

    # ---------------- right: model ----------------
    ax2.set_title("(b) HGDR model - every dashed block is an ablation switch", fontsize=10)
    _box(ax2, (0.02, 0.80), 0.28, 0.13, "ID embeddings\ndiag / proc / drug", "#f5f5f5")
    _box(ax2, (0.34, 0.80), 0.28, 0.13, "Molecular encoder\nGAT/GIN on atoms -> gate", "#e0e0e0")
    _box(ax2, (0.66, 0.80), 0.32, 0.13, "Relation-aware GNN\nHeteroConv(SAGE) x 2", "#a9cff5")
    _arrow(ax2, (0.30, 0.865), (0.34, 0.865))
    _arrow(ax2, (0.62, 0.865), (0.66, 0.865))
    ax2.text(0.50, 0.75, "refined node embeddings  {diag, proc, drug}", ha="center", fontsize=8, style="italic")
    _arrow(ax2, (0.82, 0.80), (0.82, 0.60))
    _arrow(ax2, (0.16, 0.80), (0.16, 0.60))
    _box(ax2, (0.02, 0.44), 0.30, 0.16, "Current admission t\nattention pooling over\ndiagnoses + procedures", "#ffd59e")
    _box(ax2, (0.36, 0.44), 0.30, 0.16, "History GRU\nadmissions t-k .. t-1\n(codes + prescribed drugs)", "#f7cbe3")
    _box(ax2, (0.70, 0.44), 0.28, 0.16, "Drug keys\nK = W_k h_drug", "#a9cff5")
    _arrow(ax2, (0.32, 0.52), (0.36, 0.52))
    _box(ax2, (0.20, 0.20), 0.36, 0.13, "admission query q\n(fuse current + history)", "#ffffff")
    _arrow(ax2, (0.17, 0.44), (0.30, 0.33))
    _arrow(ax2, (0.51, 0.44), (0.44, 0.33))
    _arrow(ax2, (0.84, 0.44), (0.56, 0.265))
    _box(ax2, (0.62, 0.20), 0.36, 0.13, "scores = q K^T / sqrt(d) + b\n-> sigmoid over 606 drugs", "#ffffff", lw=1.6)
    _arrow(ax2, (0.56, 0.265), (0.62, 0.265))
    _box(ax2, (0.06, 0.02), 0.40, 0.11, "BCE(y, p)   +   lambda * soft-DDI-rate(p, A_ddi)", "#f4c2c2")
    _box(ax2, (0.54, 0.02), 0.44, 0.11, "recommended set = {p > 0.5}\nJaccard, PR-AUC, F1, DDI rate, #drugs", "#f5f5f5")
    _arrow(ax2, (0.72, 0.20), (0.36, 0.13))
    _arrow(ax2, (0.80, 0.20), (0.76, 0.13))
    for (x, y, w, h) in [(0.34, 0.80, 0.28, 0.13), (0.66, 0.80, 0.32, 0.13), (0.36, 0.44, 0.30, 0.16),
                         (0.06, 0.02, 0.40, 0.11)]:
        ax2.add_patch(plt.Rectangle((x - 0.012, y - 0.012), w + 0.024, h + 0.024, fill=False, ls="--", lw=0.9,
                                    ec="#555555", zorder=1))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--curves_variant", default="full")
    args = ap.parse_args()
    rd = Path(args.results_dir)
    fig_dir = rd / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    graph_schema(fig_dir / "graph_schema.png")
    print(f"[figures] wrote {fig_dir / 'graph_schema.png'}")
    summ_path = rd / "ablation_summary.csv"
    if summ_path.exists():
        summ = pd.read_csv(summ_path)
        res = pd.read_csv(rd / "ablation_results.csv")
        ablation_bar(summ, fig_dir / "ablation_bar.png")
        ddi_vs_jaccard(summ, float(res["test_ground_truth_ddi_rate"].mean()), fig_dir / "ddi_vs_jaccard.png")
        print(f"[figures] wrote ablation_bar.png, ddi_vs_jaccard.png")
    else:
        print(f"[figures] {summ_path} not found - run scripts/run_ablation.py first (skipping ablation plots)")
    if training_curves(rd / "runs", args.curves_variant, fig_dir / "training_curves.png"):
        print(f"[figures] wrote training_curves.png ({args.curves_variant})")


if __name__ == "__main__":
    main()
