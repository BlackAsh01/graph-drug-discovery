#!/usr/bin/env python
"""Step 2 - Reduce raw TWOSIDES to a compact CID-pair table.

The output (``data/mappings/twosides_pairs_top40.csv``) is vocabulary
independent and small, so it is shipped with the repository; you only need
to run this script if you want a different ``--top_k`` or a fresh TWOSIDES
download (see ``scripts/download_twosides.py``).

Example::

    python scripts/build_ddi_graph.py --twosides data/raw/twosides.csv --top_k 40
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from hgdr.data.ddi import build_cid_pair_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twosides", default=os.environ.get("TWOSIDES_CSV", "data/raw/twosides.csv"))
    ap.add_argument("--top_k", type=int, default=40, help="most frequent side-effect types to keep (0 = all)")
    ap.add_argument("--out", default=None, help="output CSV (default data/mappings/twosides_pairs_top<k>.csv)")
    args = ap.parse_args()
    out = Path(args.out or f"data/mappings/twosides_pairs_top{args.top_k}.csv")
    pairs, se = build_cid_pair_table(args.twosides, top_k_side_effects=args.top_k)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out, index=False)
    se.to_csv(out.with_name(out.stem + "_side_effects.csv"), index=False)
    print(f"[ddi] wrote {len(pairs):,} CID pairs to {out}")


if __name__ == "__main__":
    main()
