"""Loading, normalising and splitting the KIBA drug-target affinity dataset.

The canonical raw file is the Therapeutics Data Commons (TDC) ``kiba.tab`` (tab separated,
columns ``ID1, X1, ID2, X2, Y`` = drug id, SMILES, target id, sequence, KIBA score;
117,657 pairs, 2,068 drugs, 229 targets). :func:`load_kiba_table` also accepts the
TDC ``Drug_ID, Drug, Target_ID, Target, Y`` column names and the compact three-file
format written by :func:`write_compact_kiba`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

TDC_DATAVERSE_FILE_ID = 5255037  # https://dataverse.harvard.edu/api/access/datafile/5255037
KIBA_URL = f"https://dataverse.harvard.edu/api/access/datafile/{TDC_DATAVERSE_FILE_ID}"

CANONICAL_COLUMNS = ["drug_id", "smiles", "target_id", "sequence", "y"]
_COLUMN_ALIASES = {
    "ID1": "drug_id",
    "X1": "smiles",
    "ID2": "target_id",
    "X2": "sequence",
    "Y": "y",
    "Drug_ID": "drug_id",
    "Drug": "smiles",
    "Target_ID": "target_id",
    "Target": "sequence",
}


def load_kiba_table(path: str | Path) -> pd.DataFrame:
    """Read a KIBA pair table (``.tab``/``.tsv``/``.csv``) and normalise column names."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tab", ".tsv"} else ","
    df = pd.read_csv(path, sep=sep)
    df = df.rename(columns=_COLUMN_ALIASES)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(df.columns)}")
    df = df[CANONICAL_COLUMNS].copy()
    df["drug_id"] = df["drug_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)
    df["y"] = df["y"].astype(np.float32)
    return df.reset_index(drop=True)


def write_compact_kiba(df: pd.DataFrame, out_dir: str | Path) -> None:
    """Write the normalised table as three small files (pairs / drugs / targets)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[["drug_id", "target_id", "y"]].to_csv(out_dir / "pairs.csv.gz", index=False, compression="gzip")
    df[["drug_id", "smiles"]].drop_duplicates("drug_id").to_csv(out_dir / "drugs.csv", index=False)
    df[["target_id", "sequence"]].drop_duplicates("target_id").to_csv(out_dir / "targets.csv", index=False)


def load_compact_kiba(compact_dir: str | Path) -> pd.DataFrame:
    """Inverse of :func:`write_compact_kiba` (row order of ``pairs.csv.gz`` is preserved)."""
    compact_dir = Path(compact_dir)
    pairs = pd.read_csv(compact_dir / "pairs.csv.gz", dtype={"drug_id": str, "target_id": str})
    drugs = pd.read_csv(compact_dir / "drugs.csv", dtype=str)
    targets = pd.read_csv(compact_dir / "targets.csv", dtype=str)
    df = pairs.merge(drugs, on="drug_id", how="left").merge(targets, on="target_id", how="left")
    df["y"] = df["y"].astype(np.float32)
    return df[CANONICAL_COLUMNS].reset_index(drop=True)


def resolve_kiba(data_dir: str | Path, pairs_file: Optional[str | Path] = None) -> pd.DataFrame:
    """Locate the KIBA table, trying (in order) an explicit file, ``raw/kiba.tab`` and the compact copy."""
    data_dir = Path(data_dir)
    candidates = []
    if pairs_file is not None:
        candidates.append(Path(pairs_file))
    candidates += [data_dir / "raw" / "kiba.tab", data_dir / "raw" / "kiba.csv"]
    for c in candidates:
        if c.is_file():
            return load_kiba_table(c)
    compact = data_dir / "kiba"
    if (compact / "pairs.csv.gz").is_file():
        return load_compact_kiba(compact)
    raise FileNotFoundError(
        "KIBA not found. Run `python scripts/download_kiba.py` (or scripts/link_local_data.ps1) "
        f"to create {data_dir / 'raw' / 'kiba.tab'}."
    )


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def stratified_subset(y: np.ndarray, fraction: float, seed: int, n_bins: int = 10) -> np.ndarray:
    """Return sorted indices of a ``fraction`` subset stratified over quantile bins of ``y``."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(y, edges)
    keep = []
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        n_keep = int(round(len(idx) * fraction))
        keep.append(rng.choice(idx, size=n_keep, replace=False))
    return np.sort(np.concatenate(keep))


def random_split(indices: np.ndarray, seed: int, val_frac: float = 0.1, test_frac: float = 0.1) -> Dict[str, np.ndarray]:
    """Shuffle ``indices`` and split into train / val / test (pair-level, transductive split)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    n = len(perm)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    return {
        "test": np.sort(perm[:n_test]),
        "val": np.sort(perm[n_test : n_test + n_val]),
        "train": np.sort(perm[n_test + n_val :]),
    }


def make_split(df: pd.DataFrame, seed: int = 42, subset_fraction: float = 1.0, val_frac: float = 0.1, test_frac: float = 0.1) -> Dict[str, np.ndarray]:
    """Create (and optionally subsample) a reproducible split of the KIBA pairs."""
    subset = stratified_subset(df["y"].to_numpy(), subset_fraction, seed)
    return random_split(subset, seed, val_frac, test_frac)


def save_split(split: Dict[str, np.ndarray], out_dir: str | Path, meta: Optional[dict] = None) -> None:
    """Write ``train.txt / val.txt / test.txt`` (one row index per line) plus ``meta.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in split.items():
        np.savetxt(out_dir / f"{name}.txt", np.asarray(idx, dtype=np.int64), fmt="%d")
    meta = dict(meta or {})
    meta.update({k: int(len(v)) for k, v in split.items()})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_split(split_dir: str | Path) -> Dict[str, np.ndarray]:
    """Read a split written by :func:`save_split`."""
    split_dir = Path(split_dir)
    out = {}
    for name in ("train", "val", "test"):
        f = split_dir / f"{name}.txt"
        if not f.is_file():
            raise FileNotFoundError(f"split file {f} not found; run scripts/preprocess.py")
        out[name] = np.loadtxt(f, dtype=np.int64).reshape(-1)
    return out


def split_name(seed: int, subset_fraction: float) -> str:
    """Canonical folder name for a split, e.g. ``kiba_random_seed42`` or ``kiba_subset20_seed42``."""
    if subset_fraction >= 1.0:
        return f"kiba_random_seed{seed}"
    return f"kiba_subset{int(round(subset_fraction * 100))}_seed{seed}"


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    """Basic dataset statistics used in logs and the README."""
    return {
        "pairs": int(len(df)),
        "drugs": int(df["drug_id"].nunique()),
        "targets": int(df["target_id"].nunique()),
        "y_mean": float(df["y"].mean()),
        "y_std": float(df["y"].std()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
    }
