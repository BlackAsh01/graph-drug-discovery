"""Re-evaluate a trained run from its checkpoint, or summarise metrics.json files.

Examples::

    python scripts/evaluate.py --run results/runs/full_model_fulldata_seed0          # needs best.pt
    python scripts/evaluate.py --summarize results/runs                               # table from metrics.json files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from dti_gt.evaluate import evaluate_run

METRICS = ["mse", "rmse", "ci", "rm2", "pearson", "spearman", "auroc", "auprc"]


def summarize(runs_dir: Path) -> None:
    rows = []
    for mfile in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(mfile.read_text())
        rows.append((m["run_name"], m["seed"], m["best_epoch"], m["train_time_s"], m["test"]))
    if not rows:
        print(f"no metrics.json found under {runs_dir}")
        return
    header = ["run", "seed", "best_epoch", "time_s"] + METRICS
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for run, seed, ep, t, test in rows:
        print(f"| {run} | {seed} | {ep} | {t:.0f} | " + " | ".join(f"{test.get(k, float('nan')):.4f}" for k in METRICS) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run directory containing best.pt and config.yaml")
    ap.add_argument("--summarize", default=None, metavar="RUNS_DIR", help="print a markdown table of all runs")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.summarize:
        summarize(Path(args.summarize))
    if args.run:
        out = evaluate_run(args.run, ROOT, device=args.device)
        print(json.dumps(out, indent=2))
    if not args.run and not args.summarize:
        ap.print_help()


if __name__ == "__main__":
    main()
