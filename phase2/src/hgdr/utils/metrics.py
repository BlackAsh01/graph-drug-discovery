"""Evaluation metrics for multi-label medication recommendation.

Two metric families are provided:

* **Set-based metrics (SafeDrug/GAMENet protocol)** – computed per visit
  from a thresholded prediction and averaged: Jaccard, PR-AUC, F1,
  average number of recommended drugs and the DDI rate of the
  recommended set.
* **Top-k metrics** – Precision@k, Jaccard@k and HitRate@k, which the
  original Phase-2 notebooks reported (``k=10``).

All functions operate on NumPy arrays of shape ``(n_visits, n_drugs)``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def ddi_rate_score(pred_sets: np.ndarray, ddi_adj: np.ndarray) -> float:
    """DDI rate of a batch of recommended drug sets.

    Follows GAMENet/SafeDrug: the fraction of recommended drug *pairs*
    (over all visits) that are known interactions.

    Args:
        pred_sets: Binary matrix ``(n_visits, n_drugs)``.
        ddi_adj: Symmetric binary DDI adjacency ``(n_drugs, n_drugs)``.

    Returns:
        DDI rate in ``[0, 1]`` (0.0 if no pairs were recommended).
    """
    pred_sets = np.asarray(pred_sets, dtype=np.float32)
    ddi_adj = np.asarray(ddi_adj, dtype=np.float32)
    all_pairs = 0.0
    ddi_pairs = 0.0
    for row in pred_sets:
        idx = np.flatnonzero(row)
        n = len(idx)
        if n < 2:
            continue
        all_pairs += n * (n - 1) / 2.0
        sub = ddi_adj[np.ix_(idx, idx)]
        ddi_pairs += np.triu(sub, k=1).sum()
    return _safe_div(ddi_pairs, all_pairs)


def multi_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    ddi_adj: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Visit-averaged Jaccard / PR-AUC / F1 / avg #drugs (+ DDI rate).

    Args:
        y_true: Binary ground-truth matrix ``(n_visits, n_drugs)``.
        y_prob: Predicted probabilities, same shape.
        threshold: Probability threshold for the recommended set.  If a
            visit ends up with an empty set, its single top-scoring drug
            is recommended instead (as in the SafeDrug code base).
        ddi_adj: Optional DDI adjacency to compute ``ddi_rate``.

    Returns:
        Dict with keys ``jaccard, prauc, f1, precision, recall, avg_drugs``
        and ``ddi_rate`` (if ``ddi_adj`` given).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    assert y_true.shape == y_prob.shape, "shape mismatch"
    n_visits, _ = y_true.shape

    y_pred = (y_prob >= threshold).astype(np.int64)
    empty = y_pred.sum(1) == 0
    if empty.any():
        top1 = y_prob[empty].argmax(1)
        y_pred[np.flatnonzero(empty), top1] = 1

    jac, f1s, precs, recs, praucs = [], [], [], [], []
    for i in range(n_visits):
        t = set(np.flatnonzero(y_true[i]))
        p = set(np.flatnonzero(y_pred[i]))
        inter = len(t & p)
        union = len(t | p)
        jac.append(_safe_div(inter, union))
        prec = _safe_div(inter, len(p))
        rec = _safe_div(inter, len(t))
        precs.append(prec)
        recs.append(rec)
        f1s.append(_safe_div(2 * prec * rec, prec + rec))
        if y_true[i].sum() > 0 and y_true[i].sum() < y_true.shape[1]:
            praucs.append(average_precision_score(y_true[i], y_prob[i]))
    out = {
        "jaccard": float(np.mean(jac)),
        "prauc": float(np.mean(praucs)) if praucs else 0.0,
        "f1": float(np.mean(f1s)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "avg_drugs": float(y_pred.sum(1).mean()),
    }
    if ddi_adj is not None:
        out["ddi_rate"] = ddi_rate_score(y_pred, ddi_adj)
    return out


def topk_metrics(y_true: np.ndarray, y_prob: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Precision@k, Jaccard@k and HitRate@k (visit-averaged).

    These mirror the ``precision_at_k`` / ``jaccard_at_k`` /
    ``hit_rate_at_k`` helpers in the original research notebook.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    k = min(k, y_prob.shape[1])
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    precs, jacs, hits = [], [], []
    for i in range(y_true.shape[0]):
        t = set(np.flatnonzero(y_true[i]))
        p = set(topk[i].tolist())
        inter = len(t & p)
        precs.append(inter / float(k))
        jacs.append(_safe_div(inter, len(t | p)))
        hits.append(1.0 if inter > 0 else 0.0)
    return {
        f"precision@{k}": float(np.mean(precs)),
        f"jaccard@{k}": float(np.mean(jacs)),
        f"hit_rate@{k}": float(np.mean(hits)),
    }
