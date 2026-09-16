#!/usr/bin/env python
"""Step 5 - Evaluate a saved checkpoint.

Example::

    python scripts/evaluate.py --config results/runs/full_seed0/config.yaml \
        --checkpoint results/runs/full_seed0/best.pt
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from hgdr.config import load_config
from hgdr.evaluate import evaluate_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = evaluate_checkpoint(cfg, args.checkpoint, args.threshold)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
