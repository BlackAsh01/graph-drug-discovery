"""Deterministic, hash-based patient-level train/val/test split.

Patient IDs are MIMIC-derived and must not be shipped, so instead of a
list of IDs we ship the *procedure*: every patient is assigned to a split
by ``md5(f"{split_seed}:{subject_id}")``.  Given the same raw data and
``split_seed`` the split is bit-for-bit reproducible on any machine.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple


def split_bucket(subject_id: int, split_seed: int = 0) -> float:
    """Map a patient to a uniform pseudo-random number in ``[0, 1)``."""
    h = hashlib.md5(f"{split_seed}:{int(subject_id)}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def assign_split(subject_id: int, fractions: Sequence[float] = (0.8, 0.1, 0.1),
                 split_seed: int = 0) -> str:
    """Return ``'train'``, ``'val'`` or ``'test'`` for a patient."""
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    u = split_bucket(subject_id, split_seed)
    if u < fractions[0]:
        return "train"
    if u < fractions[0] + fractions[1]:
        return "val"
    return "test"


def subsample_records(
    records: List[List[Dict[str, object]]],
    fraction: float = 1.0,
    subset_seed: int = 0,
) -> List[List[Dict[str, object]]]:
    """Keep a deterministic, hash-selected fraction of *patients*.

    This is the documented **subset protocol** used for the ablation study on
    limited hardware: with ``fraction=0.3`` every patient whose bucket
    ``md5(f"subset{subset_seed}:{subject_id}")`` falls below 0.3 is kept, so
    the same 30 % of patients is used by every variant and every training
    seed.  ``fraction >= 1`` returns the records unchanged.
    """
    if fraction >= 1.0:
        return list(records)
    kept = []
    for patient in records:
        sid = int(patient[0]["subject_id"])
        h = hashlib.md5(f"subset{subset_seed}:{sid}".encode("utf-8")).hexdigest()
        if int(h[:12], 16) / float(16 ** 12) < fraction:
            kept.append(patient)
    return kept


def split_records(
    records: List[List[Dict[str, object]]],
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    split_seed: int = 0,
) -> Tuple[List[List[Dict[str, object]]], List[List[Dict[str, object]]], List[List[Dict[str, object]]]]:
    """Split patient records into (train, val, test) by patient hash."""
    out = {"train": [], "val": [], "test": []}
    for patient in records:
        sid = int(patient[0]["subject_id"])
        out[assign_split(sid, fractions, split_seed)].append(patient)
    return out["train"], out["val"], out["test"]
