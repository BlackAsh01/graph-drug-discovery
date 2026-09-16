#!/usr/bin/env python
"""Step 4 - Train and evaluate one configuration.

Examples::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/ablation_no_ddi.yaml --seed 1
    python scripts/train.py --config configs/default.yaml --set train.epochs=3 model.hidden_dim=32

Artefacts go to ``results/runs/<name>_seed<seed>/``.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from hgdr.config import load_config
from hgdr.train import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    out = run_experiment(cfg, run_name=args.run_name, results_dir=args.results_dir)
    print(json.dumps({"run": out["run_name"], "test": out["test"], "runtime_s": out["runtime_s"]}, indent=2))


if __name__ == "__main__":
    main()
