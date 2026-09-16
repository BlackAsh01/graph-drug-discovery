#!/usr/bin/env python
"""Build / extend the drug-name -> PubChem (CID, SMILES) dictionary.

The shipped ``data/mappings/drug_name_to_pubchem.json`` already covers the
frequent MIMIC-III drug names.  Use this script to

* merge additional legacy JSON dictionaries (``--merge a.json b.json``), or
* resolve names of a processed vocabulary that are still unmapped through
  the PubChem PUG-REST API (``--processed_dir ... --online``; needs network).

Example::

    python scripts/build_drug_mapping.py --processed_dir data/processed/mimic3 --online
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from hgdr.data.drug_mapping import is_non_drug, load_mapping, lookup_pubchem_online, normalize_drug_name, save_mapping
from hgdr.data.mimic import ProcessedData


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", default="data/mappings/drug_name_to_pubchem.json")
    ap.add_argument("--merge", nargs="*", default=[], help="legacy {name: {cid, smiles}} JSON files to merge")
    ap.add_argument("--processed_dir", default=None, help="vocabulary whose unmapped names should be resolved")
    ap.add_argument("--online", action="store_true", help="query PubChem for unmapped names")
    ap.add_argument("--max_queries", type=int, default=None)
    args = ap.parse_args()

    path = Path(args.mapping)
    mapping = load_mapping(path) if path.exists() else {}
    print(f"[mapping] {len(mapping)} entries loaded")
    for extra in args.merge:
        add = load_mapping(extra)
        new = 0
        for k, v in add.items():
            key = normalize_drug_name(k)
            if key and key not in mapping and not is_non_drug(key):
                mapping[key] = v
                new += 1
        print(f"[mapping] merged {extra}: +{new}")

    if args.processed_dir:
        data = ProcessedData.load(args.processed_dir)
        words = data.drug_voc.idx2word
        unmapped = [w for w in words if w not in mapping]
        print(f"[mapping] vocabulary {len(words)} drugs; {len(words) - len(unmapped)} mapped, {len(unmapped)} unmapped")
        if args.online and unmapped:
            lookup_pubchem_online(unmapped, mapping, max_queries=args.max_queries)
            still = [w for w in words if w not in mapping]
            print(f"[mapping] after online lookup: {len(words) - len(still)} mapped, {len(still)} unmapped")
            if still:
                print("[mapping] unmapped examples:", still[:30])
    save_mapping(mapping, path)
    print(f"[mapping] saved {len(mapping)} entries to {path}")


if __name__ == "__main__":
    main()
