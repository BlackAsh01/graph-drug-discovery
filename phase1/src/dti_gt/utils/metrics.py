"""Regression metrics used in the drug-target affinity literature (DeepDTA, GraphDTA, ...).

* MSE / RMSE / MAE
* Concordance Index (CI, Gonen & Heller 2005)
* :math:`r_m^2` index (Roy et al. 2013) as used by DeepDTA
* Pearson and Spearman correlation
* AUROC / AUPRC after binarising KIBA scores at the conventional threshold 12.1
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

KIBA_BINARY_THRESHOLD = 12.1


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, chunk: int = 2048) -> float:
    """Concordance index: P(pred_i > pred_j | true_i > true_j), ties in prediction count 0.5.

    Implemented with chunked vectorisation so that it runs in a few seconds for ~10k points.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    if n < 2:
        return float("nan")
    order = np.argsort(y_true)
    t, p = y_true[order], y_pred[order]
    concordant = 0.0
    comparable = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # pairs (i in chunk, j < i) -> compare against all earlier points; mask ties in y_true
        ti, pi = t[start:stop, None], p[start:stop, None]
        tj, pj = t[None, :stop], p[None, :stop]
        valid = ti > tj  # strictly larger true value (upper-triangular by construction)
        concordant += np.sum(valid & (pi > pj)) + 0.5 * np.sum(valid & (pi == pj))
        comparable += int(valid.sum())
    return float(concordant / comparable) if comparable else float("nan")


def _r_squared(y: np.ndarray, f: np.ndarray) -> float:
    """Squared correlation coefficient r^2 (with intercept)."""
    y_mean, f_mean = y.mean(), f.mean()
    num = np.sum((y - y_mean) * (f - f_mean)) ** 2
    den = np.sum((y - y_mean) ** 2) * np.sum((f - f_mean) ** 2)
    return float(num / den) if den > 0 else 0.0


def _r0_squared(y: np.ndarray, f: np.ndarray) -> float:
    """r_0^2: squared correlation of the regression through the origin (DeepDTA definition)."""
    k = np.sum(y * f) / np.sum(f * f) if np.sum(f * f) > 0 else 0.0
    y_mean = y.mean()
    num = np.sum((y - k * f) ** 2)
    den = np.sum((y - y_mean) ** 2)
    return float(1.0 - num / den) if den > 0 else 0.0


def rm2_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Modified squared correlation :math:`r_m^2 = r^2 (1 - \\sqrt{|r^2 - r_0^2|})`."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    f = np.asarray(y_pred, dtype=np.float64).ravel()
    r2, r02 = _r_squared(y, f), _r0_squared(y, f)
    return float(r2 * (1.0 - np.sqrt(np.abs(r2 - r02))))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, binary_threshold: float = KIBA_BINARY_THRESHOLD) -> Dict[str, float]:
    """Compute the full metric set. Returns plain floats (JSON serialisable)."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    f = np.asarray(y_pred, dtype=np.float64).ravel()
    if y.shape != f.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {f.shape}")
    err = f - y
    out: Dict[str, float] = {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "ci": concordance_index(y, f),
        "rm2": rm2_index(y, f),
    }
    if len(y) > 2 and np.std(f) > 0 and np.std(y) > 0:
        out["pearson"] = float(stats.pearsonr(y, f)[0])
        out["spearman"] = float(stats.spearmanr(y, f)[0])
    else:
        out["pearson"] = out["spearman"] = float("nan")
    labels = (y >= binary_threshold).astype(int)
    if 0 < labels.sum() < len(labels):
        out["auroc"] = float(roc_auc_score(labels, f))
        out["auprc"] = float(average_precision_score(labels, f))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out
