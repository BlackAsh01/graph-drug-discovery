"""Preprocess KIBA: normalise the table, featurise drugs, write splits and a compact copy.

Steps
-----
1. locate ``data/raw/kiba.tab`` (or the compact ``data/kiba/`` copy),
2. write the compact three-file copy ``data/kiba/{pairs.csv.gz, drugs.csv, targets.csv}``,
3. featurise all unique SMILES into graphs -> ``data/processed/drug_graphs.pt`` (cache),
4. write the fixed splits ``data/splits/kiba_random_seed42`` (full) and
   ``data/splits/kiba_subset20_seed42`` (20 % stratified subset used by the ablation study).

Protein-language-model embeddings are produced separately by ``scripts/embed_proteins.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

import torch

from dti_gt.data.dataset import load_or_build_drug_graphs
from dti_gt.data.kiba import make_split, resolve_kiba, save_split, split_name, summarize, write_compact_kiba


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--pairs-file", default=None, help="explicit KIBA table (default: data/raw/kiba.tab or data/kiba/)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--subset-fractions", type=float, nargs="+", default=[1.0, 0.2], help="splits to generate")
    ap.add_argument("--no-compact", action="store_true", help="do not (re)write data/kiba/ compact copy")
    ap.add_argument("--force", action="store_true", help="re-featurise even if the cache exists")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    df = resolve_kiba(data_dir, args.pairs_file)
    stats = summarize(df)
    print("[preprocess] KIBA:", json.dumps(stats))

    if not args.no_compact:
        write_compact_kiba(df, data_dir / "kiba")
        print(f"[preprocess] compact copy written to {data_dir / 'kiba'}")

    cache = data_dir / "processed" / "drug_graphs.pt"
    if args.force and cache.is_file():
        cache.unlink()
    graphs = load_or_build_drug_graphs(df, cache)
    n_atoms = torch.tensor([g.num_nodes for g in graphs.values()], dtype=torch.float)
    print(f"[preprocess] {len(graphs)} drug graphs cached at {cache} (atoms: mean {n_atoms.mean():.1f}, max {int(n_atoms.max())})")

    for frac in args.subset_fractions:
        split = make_split(df, seed=args.split_seed, subset_fraction=frac)
        name = split_name(args.split_seed, frac)
        save_split(split, data_dir / "splits" / name, {"seed": args.split_seed, "subset_fraction": frac, "protocol": "random pair-level split 80/10/10" + ("" if frac >= 1 else f", stratified {int(frac * 100)}% subset over 10 quantile bins of y")})
        print(f"[preprocess] split {name}: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))


if __name__ == "__main__":
    main()
