"""TWOSIDES drug-drug-interaction processing.

The raw TWOSIDES file (as distributed by Therapeutics Data Commons) has
one row per *(drug pair, side effect)* with PubChem-style ids
(``CID000002173``), a side-effect id/name and the two SMILES strings.
Following SafeDrug we keep the ``top_k`` most frequent side-effect types
and treat any pair reported for one of them as an interaction.

Two artefacts are produced:

* a vocabulary-independent **CID pair table** (small, shippable), and
* a **DDI adjacency matrix** over a given drug vocabulary (built from the
  pair table + the drug->CID mapping).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .drug_mapping import twosides_id_to_cid

_COLUMN_ALIASES = {
    "ID1": ["ID1", "Drug1_ID", "DRUG1_ID", "drug1_id", "STITCH 1", "stitch_id1"],
    "ID2": ["ID2", "Drug2_ID", "DRUG2_ID", "drug2_id", "STITCH 2", "stitch_id2"],
    "SE": ["Side Effect Name", "side_effect_name", "Side Effect", "condition_concept_name", "Y_name"],
    "Y": ["Y", "y", "label", "side_effect_id"],
    "X1": ["X1", "Drug1", "smiles1"],
    "X2": ["X2", "Drug2", "smiles2"],
}


def _find_col(columns: Iterable[str], key: str) -> Optional[str]:
    cols = list(columns)
    for alias in _COLUMN_ALIASES[key]:
        if alias in cols:
            return alias
    return None


def build_cid_pair_table(
    twosides_csv: str | Path,
    top_k_side_effects: int = 40,
    chunksize: int = 1_000_000,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce raw TWOSIDES to ``(cid1, cid2)`` pairs for the top-k side effects.

    Args:
        twosides_csv: Path to the raw ``twosides.csv`` (~650 MB).
        top_k_side_effects: Number of most frequent side-effect types kept
            (``40`` reproduces the SafeDrug protocol; ``0`` keeps all).
        chunksize: Rows per chunk when streaming the file.

    Returns:
        ``(pairs, side_effects)`` where ``pairs`` has integer columns
        ``cid1 < cid2`` (unique) and ``side_effects`` lists the kept
        side-effect names with their pair counts.
    """
    twosides_csv = Path(twosides_csv)
    header = pd.read_csv(twosides_csv, nrows=0).columns
    c_id1, c_id2 = _find_col(header, "ID1"), _find_col(header, "ID2")
    c_se = _find_col(header, "SE") or _find_col(header, "Y")
    if c_id1 is None or c_id2 is None or c_se is None:
        raise KeyError(f"unrecognised TWOSIDES columns: {list(header)}")

    # Pass 1: side-effect frequencies
    se_counts: Dict[str, int] = {}
    for chunk in pd.read_csv(twosides_csv, usecols=[c_se], chunksize=chunksize):
        vc = chunk[c_se].astype(str).value_counts()
        for k, v in vc.items():
            se_counts[k] = se_counts.get(k, 0) + int(v)
    se_sorted = sorted(se_counts.items(), key=lambda kv: -kv[1])
    if top_k_side_effects and top_k_side_effects > 0:
        se_sorted = se_sorted[:top_k_side_effects]
    keep = {k for k, _ in se_sorted}
    if verbose:
        print(f"[ddi] {len(se_counts)} side-effect types, keeping {len(keep)}")

    # Pass 2: pairs
    pair_set = set()
    for chunk in pd.read_csv(twosides_csv, usecols=[c_id1, c_id2, c_se], chunksize=chunksize):
        chunk = chunk[chunk[c_se].astype(str).isin(keep)]
        if chunk.empty:
            continue
        a = chunk[c_id1].map(twosides_id_to_cid).to_numpy()
        b = chunk[c_id2].map(twosides_id_to_cid).to_numpy()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        pair_set.update(zip(lo.tolist(), hi.tolist()))
    pairs = pd.DataFrame(sorted(pair_set), columns=["cid1", "cid2"])
    pairs = pairs[pairs["cid1"] != pairs["cid2"]].reset_index(drop=True)
    if verbose:
        print(f"[ddi] {len(pairs):,} unique interacting CID pairs")
    se_df = pd.DataFrame(se_sorted, columns=["side_effect", "n_rows"])
    return pairs, se_df


def build_ddi_adjacency(
    drug_idx2word: Iterable[str],
    drug_to_cid: Dict[str, int],
    pairs: pd.DataFrame,
) -> np.ndarray:
    """Symmetric binary DDI adjacency over the drug vocabulary.

    Drugs without a CID (or CIDs absent from TWOSIDES) simply have no
    DDI edges.  Several vocabulary entries may share a CID (e.g. salt
    forms); all of them inherit the CID's interactions.
    """
    words = list(drug_idx2word)
    n = len(words)
    adj = np.zeros((n, n), dtype=np.float32)
    cid_to_idx: Dict[int, list] = {}
    for i, w in enumerate(words):
        cid = drug_to_cid.get(w)
        if cid is not None:
            cid_to_idx.setdefault(int(cid), []).append(i)
    for c1, c2 in pairs[["cid1", "cid2"]].itertuples(index=False):
        if c1 in cid_to_idx and c2 in cid_to_idx:
            for i in cid_to_idx[c1]:
                for j in cid_to_idx[c2]:
                    if i != j:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
    return adj
