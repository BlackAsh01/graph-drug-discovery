#!/usr/bin/env python
"""Step 3 - Build the heterogeneous entity graph (+ DDI matrix, molecular graphs).

Uses only the **training** split (hash-based patient split with
``--split_seed``) for co-occurrence edges, so validation/test labels never
leak into the graph.

Example::

    python scripts/build_hetero_graph.py --processed_dir data/processed/mimic3 \
        --mapping data/mappings/drug_name_to_pubchem.json \
        --ddi_pairs data/mappings/twosides_pairs_top40.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from hgdr.data.ddi import build_ddi_adjacency
from hgdr.data.drug_mapping import load_mapping
from hgdr.data.graph import build_hetero_graph
from hgdr.data.mimic import ProcessedData
from hgdr.data.splits import split_records
from hgdr.utils.metrics import ddi_rate_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", default="data/processed/mimic3")
    ap.add_argument("--mapping", default="data/mappings/drug_name_to_pubchem.json")
    ap.add_argument("--ddi_pairs", default="data/mappings/twosides_pairs_top40.csv")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--split_fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--min_cooccur", type=int, default=5)
    ap.add_argument("--top_k_per_node", type=int, default=20)
    ap.add_argument("--drug_cooccur_min", type=int, default=5)
    ap.add_argument("--drug_cooccur_top_k", type=int, default=20)
    ap.add_argument("--out", default=None, help="output file (default <processed_dir>/graph.pt)")
    args = ap.parse_args()

    pdir = Path(args.processed_dir)
    data = ProcessedData.load(pdir)
    mapping = load_mapping(args.mapping)
    pairs = pd.read_csv(args.ddi_pairs)
    drug_to_cid = {w: mapping[w]["cid"] for w in data.drug_voc.idx2word if w in mapping}
    ddi_adj = build_ddi_adjacency(data.drug_voc.idx2word, drug_to_cid, pairs)

    train, val, test = split_records(data.records, args.split_fractions, args.split_seed)
    graph = build_hetero_graph(
        train, len(data.diag_voc), len(data.proc_voc), data.drug_voc.idx2word, mapping, ddi_adj,
        min_cooccur=args.min_cooccur, top_k_per_node=args.top_k_per_node,
        drug_cooccur_min=args.drug_cooccur_min, drug_cooccur_top_k=args.drug_cooccur_top_k,
    )
    graph.meta.update({"split_seed": args.split_seed, "split_fractions": args.split_fractions,
                       "n_train_patients": len(train), "n_val_patients": len(val), "n_test_patients": len(test)})
    out = Path(args.out or pdir / "graph.pt")
    graph.save(out)

    # dataset-level statistics that are safe to publish (no patient rows)
    n_drug = len(data.drug_voc)
    y = np.zeros((sum(len(p) for p in data.records), n_drug), dtype=np.float32)
    i = 0
    for p in data.records:
        for v in p:
            y[i, v["drug"]] = 1
            i += 1
    stats = {
        "n_drugs": n_drug,
        "n_drugs_with_cid": graph.meta["n_drugs_with_cid"],
        "n_drugs_with_smiles": graph.meta["n_drugs_with_smiles"],
        "ddi_edges": int(ddi_adj.sum() // 2),
        "ddi_density": float(ddi_adj.sum() / max(1, n_drug * (n_drug - 1))),
        "ground_truth_ddi_rate_all_visits": ddi_rate_score(y, ddi_adj),
        "split_visits": {"train": sum(len(p) for p in train), "val": sum(len(p) for p in val),
                         "test": sum(len(p) for p in test)},
        "split_patients": {"train": len(train), "val": len(val), "test": len(test)},
        "edges": {"|".join(k): int(v.shape[1]) for k, v in graph.edge_index.items()},
    }
    with open(pdir / "graph_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[graph] saved {out}\n[graph] stats: {json.dumps(stats, indent=1)}")


if __name__ == "__main__":
    main()
