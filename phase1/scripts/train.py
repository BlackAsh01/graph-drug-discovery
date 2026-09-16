"""Train one model from a YAML config.

Examples::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/ablation/full.yaml --seed 1 --run-name full_seed1
    python scripts/train.py --config configs/default.yaml --set training.epochs=5 training.device=cpu
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from dti_gt.train import train_from_config
from dti_gt.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--seed", type=int, default=None, help="override training.seed")
    ap.add_argument("--run-name", default=None, help="folder under results/runs (default: from config)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. training.epochs=5")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.seed is not None:
        overrides.append(f"training.seed={args.seed}")
    cfg = load_config(args.config, overrides)
    run_name = args.run_name or (f"{cfg.get('variant', 'run')}_seed{cfg['training']['seed']}" if args.seed is not None else cfg.get("run_name"))
    result = train_from_config(cfg, ROOT, run_name=run_name)
    print(json.dumps({"run": result["run_name"], "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
