#!/usr/bin/env python
"""Step 1 - Build admission-level records + vocabularies from raw MIMIC-III.

Example (full data, requires PhysioNet credentialed access)::

    python scripts/preprocess_mimic.py --mimic_dir /path/to/mimic-iii-clinical-database-1.4 \
        --out data/processed/mimic3

Example (public 100-patient demo)::

    python scripts/preprocess_mimic.py --mimic_dir data/raw/mimic-iii-clinical-database-demo-1.4 \
        --out data/processed/demo --min_drug_admissions 3 --min_diag_count 2 --min_proc_count 2

The ``--mimic_dir`` may also be given through the ``MIMIC_DIR`` environment
variable (set by ``scripts/link_local_data.ps1``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from hgdr.data.mimic import preprocess_mimic, write_demo_license_note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic_dir", default=os.environ.get("MIMIC_DIR"), help="raw MIMIC-III directory")
    ap.add_argument("--out", default="data/processed/mimic3", help="output directory")
    ap.add_argument("--min_drug_admissions", type=int, default=50)
    ap.add_argument("--min_diag_count", type=int, default=5)
    ap.add_argument("--min_proc_count", type=int, default=5)
    ap.add_argument("--max_drugs", type=int, default=None)
    ap.add_argument("--no_procedures", action="store_true")
    ap.add_argument("--max_admissions", type=int, default=None,
                    help="subset protocol: keep a fixed-seed sample of N admissions")
    ap.add_argument("--subset_seed", type=int, default=0)
    args = ap.parse_args()
    if not args.mimic_dir:
        ap.error("--mimic_dir (or env MIMIC_DIR) is required")

    data = preprocess_mimic(
        args.mimic_dir,
        min_drug_admissions=args.min_drug_admissions,
        min_diag_count=args.min_diag_count,
        min_proc_count=args.min_proc_count,
        max_drugs=args.max_drugs,
        use_procedures=not args.no_procedures,
        max_admissions=args.max_admissions,
        subset_seed=args.subset_seed,
    )
    out = Path(args.out)
    data.save(out)
    write_demo_license_note(out, Path(args.mimic_dir))
    print(f"[preprocess] saved records.pkl / voc.json / stats.json to {out}")


if __name__ == "__main__":
    main()
