#!/usr/bin/env python
"""Step 5 - Run the ablation study: every variant x every seed.

Writes

* ``results/runs/<variant>_seed<k>/``  - per-run artefacts (see hgdr.train),
* ``results/ablation_results.csv``     - one row per run (all metrics + runtime),
* ``results/ablation_summary.csv``     - mean / std per variant,
* ``results/ablation_table.md``        - publication-style markdown table.

The driver is **resumable**: a run whose directory already contains
``metrics.json`` is skipped (unless ``--force``), ``ablation_results.csv`` is
rewritten after *every* finished run, and a run that crashes is logged and
skipped so the remaining runs still execute.  Runs are executed strictly one
after another (the GPU is shared) and, by default, seed-major
(``--order seeds``: seed 0 of every variant first, then seed 1, ...) so an
interrupted study still has at least one seed per variant.

Examples::

    python scripts/run_ablation.py --seeds 0 1            # the release protocol (12 variants x 2 seeds)
    python scripts/run_ablation.py --seeds 0 1 --variants full no_ddi no_mol
    python scripts/run_ablation.py --set train.epochs=2   # quick smoke run
    python scripts/run_ablation.py --processed_dir data/processed/demo --results_dir results_demo
    python scripts/run_ablation.py --summary_only         # only rebuild the CSV / markdown from results/runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import _bootstrap  # noqa: F401

# The markdown table contains arrows; a redirected stdout on Windows defaults to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
from hgdr.config import apply_overrides, load_config
from hgdr.train import run_experiment

# Variant name -> config file.  Order defines the order of the table.
VARIANTS: Dict[str, str] = {
    "full": "ablation_base.yaml",
    "no_ddi": "ablation_no_ddi.yaml",
    "no_ddi_loss": "ablation_no_ddi_loss.yaml",
    "no_mol": "ablation_no_mol.yaml",
    "homo_graph": "ablation_homo_graph.yaml",
    "no_proc": "ablation_no_proc.yaml",
    "no_history": "ablation_no_history.yaml",
    "no_gnn": "ablation_no_gnn.yaml",
    "gnn_1layer": "ablation_gnn_1layer.yaml",
    "baseline_mlp": "baseline_mlp.yaml",
    "baseline_lr": "baseline_lr.yaml",
    "baseline_nearest": "baseline_nearest.yaml",
}

DESCRIPTIONS: Dict[str, str] = {
    "full": "HGDR (full model)",
    "no_ddi": "- DDI edges & DDI loss",
    "no_ddi_loss": "- DDI loss (edges kept)",
    "no_mol": "- molecular (SMILES) drug encoder",
    "homo_graph": "homogeneous graph (types collapsed)",
    "no_proc": "- procedure nodes",
    "no_history": "- visit-history GRU",
    "no_gnn": "- graph message passing (0 layers)",
    "gnn_1layer": "1 GNN layer (instead of 2)",
    "baseline_mlp": "MLP on multi-hot codes",
    "baseline_lr": "logistic regression on multi-hot codes",
    "baseline_nearest": "copy previous admission",
}

TABLE_METRICS = [("jaccard", "Jaccard"), ("prauc", "PR-AUC"), ("f1", "F1"),
                 ("ddi_rate", "DDI rate"), ("avg_drugs", "#drugs")]


def flatten_metrics(out: dict) -> dict:
    row = {
        "run_name": out.get("run_name", f"{out['config_name']}_seed{out['seed']}"),
        "variant": out["config_name"], "seed": out["seed"], "model": out["model"], "device": out["device"],
        "runtime_s": out["runtime_s"], "epochs_run": out["epochs_run"], "best_epoch": out["best_epoch"],
        "n_params": out["n_params"], "test_ground_truth_ddi_rate": out["test_ground_truth_ddi_rate"],
    }
    row.update({f"val_{k}": v for k, v in out["val"].items()})
    row.update({f"test_{k}": v for k, v in out["test"].items()})
    return row


def summarise(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith("test_") or c in ("runtime_s", "epochs_run")]
    g = df.groupby("variant")[metric_cols]
    summ = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std"))
    summ["n_seeds"] = g.size()
    summ = summ.reindex([v for v in order if v in summ.index])
    return summ.reset_index()


def to_markdown(summ: pd.DataFrame, gt_ddi: float, title: str) -> str:
    lines = [f"# {title}", "",
             f"Mean ± std over seeds (`n_seeds` column). Ground-truth DDI rate of the test prescriptions: {gt_ddi:.4f}.",
             "Arrows: ↑ higher is better, ↓ lower is better. `#drugs` is the average recommended-set size.", "",
             "| Variant | Description | " + " | ".join(
                 f"{lab} {'↓' if key == 'ddi_rate' else ('' if key == 'avg_drugs' else '↑')}".strip()
                 for key, lab in TABLE_METRICS) + " | Runtime (min) | Seeds |",
             "|---|---|" + "---:|" * (len(TABLE_METRICS) + 2)]
    for _, r in summ.iterrows():
        cells = []
        for key, _ in TABLE_METRICS:
            m, s = r[f"test_{key}_mean"], r[f"test_{key}_std"]
            cells.append(f"{m:.4f} ± {s:.4f}" if key != "avg_drugs" else f"{m:.2f} ± {s:.2f}")
        lines.append(f"| `{r['variant']}` | {DESCRIPTIONS.get(r['variant'], '')} | " + " | ".join(cells)
                     + f" | {r['runtime_s_mean'] / 60:.1f} | {int(r['n_seeds'])} |")
    return "\n".join(lines) + "\n"


def _load_cached(metrics_path: Path) -> Optional[dict]:
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_outputs(rows: List[dict], results_dir: Path, title: str, final: bool = False) -> Optional[pd.DataFrame]:
    """(Re)write ``ablation_results.csv`` (+ summary / markdown when ``final``) from the rows so far."""
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("run_name", keep="last")
    df.to_csv(results_dir / "ablation_results.csv", index=False)
    if final:
        summ = summarise(df, list(VARIANTS))
        summ.to_csv(results_dir / "ablation_summary.csv", index=False)
        md = to_markdown(summ, float(df["test_ground_truth_ddi_rate"].mean()), title)
        (results_dir / "ablation_table.md").write_text(md, encoding="utf-8")
        print(md)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), help=f"subset of {list(VARIANTS)}")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--processed_dir", default=None, help="override data.processed_dir for every variant")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides applied to every variant")
    ap.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    ap.add_argument("--order", choices=["seeds", "variants"], default="seeds",
                    help="'seeds': seed 0 of every variant first (default); 'variants': all seeds of a variant in a row")
    ap.add_argument("--summary_only", action="store_true",
                    help="do not train; rebuild the CSV / table from the metrics.json files that exist")
    ap.add_argument("--title", default="HGDR ablation study (MIMIC-III)")
    args = ap.parse_args()

    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant '{variant}'; choose from {list(VARIANTS)}")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.order == "seeds":
        jobs = [(v, s) for s in args.seeds for v in args.variants]
    else:
        jobs = [(v, s) for v in args.variants for s in args.seeds]

    rows: List[dict] = []
    failed: List[str] = []
    t_all = time.time()
    for i, (variant, seed) in enumerate(jobs, 1):
        cfg = load_config(Path(args.configs_dir) / VARIANTS[variant])
        cfg["seed"] = seed
        cfg["keep_checkpoint"] = False  # checkpoints are large; the study only needs metrics
        if args.processed_dir:
            cfg["data"]["processed_dir"] = args.processed_dir
        apply_overrides(cfg, args.set)
        run_name = f"{cfg['name']}_seed{seed}"
        metrics_path = results_dir / "runs" / run_name / "metrics.json"
        out = None if args.force else _load_cached(metrics_path)
        if out is not None:
            print(f"[ablation] ({i}/{len(jobs)}) {run_name}: cached", flush=True)
        elif args.summary_only:
            print(f"[ablation] ({i}/{len(jobs)}) {run_name}: missing (summary_only)", flush=True)
            continue
        else:
            print(f"[ablation] ({i}/{len(jobs)}) {run_name}: training ...", flush=True)
            try:
                out = run_experiment(cfg, run_name=run_name, results_dir=str(results_dir))
            except Exception as exc:  # noqa: BLE001 - keep the study going, report at the end
                print(f"[ablation] {run_name}: FAILED - {exc!r}", flush=True)
                traceback.print_exc()
                failed.append(run_name)
                continue
        rows.append(flatten_metrics(out))
        print(f"[ablation] {run_name}: test jaccard {out['test']['jaccard']:.4f} "
              f"ddi {out['test']['ddi_rate']:.4f} ({out['runtime_s'] / 60:.1f} min) | "
              f"elapsed {(time.time() - t_all) / 60:.1f} min", flush=True)
        write_outputs(rows, results_dir, args.title, final=False)  # incremental, survives interruption

    write_outputs(rows, results_dir, args.title, final=True)
    print(f"[ablation] {len(rows)} runs, total wall time {(time.time() - t_all) / 60:.1f} min")
    if failed:
        print(f"[ablation] FAILED runs: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
