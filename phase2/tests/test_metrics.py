"""Metric functions: hand-computed expectations."""

import numpy as np

from hgdr.utils.metrics import ddi_rate_score, multi_label_metrics, topk_metrics


def test_perfect_prediction_scores_one():
    y = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32)
    p = y.copy()
    m = multi_label_metrics(y, p, threshold=0.5)
    assert m["jaccard"] == 1.0 and m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert abs(m["avg_drugs"] - 1.5) < 1e-9


def test_jaccard_and_f1_hand_computed():
    y = np.array([[1, 1, 0, 0]], dtype=np.float32)
    p = np.array([[0.9, 0.1, 0.8, 0.2]], dtype=np.float32)  # predicts {0, 2}, truth {0, 1}
    m = multi_label_metrics(y, p, threshold=0.5)
    assert abs(m["jaccard"] - 1 / 3) < 1e-9
    assert abs(m["f1"] - 0.5) < 1e-9
    assert m["avg_drugs"] == 2.0


def test_ddi_rate():
    adj = np.zeros((4, 4), dtype=np.float32)
    adj[0, 1] = adj[1, 0] = 1.0
    sets = np.array([[1, 1, 0, 0],   # pair (0,1) interacts -> 1 of 1 pairs
                     [1, 0, 1, 1]],   # pairs (0,2),(0,3),(2,3) -> 0 of 3
                    dtype=np.float32)
    rate = ddi_rate_score(sets, adj)
    assert abs(rate - 1 / 4) < 1e-9  # aggregated over all pairs: 1 / (1 + 3)
    assert ddi_rate_score(np.zeros((2, 4)), adj) == 0.0


def test_topk_metrics():
    y = np.array([[1, 1, 0, 0, 0]], dtype=np.float32)
    p = np.array([[0.9, 0.2, 0.8, 0.1, 0.0]], dtype=np.float32)
    m = topk_metrics(y, p, k=2)  # top-2 = {0, 2}; hit 1 of 2
    assert abs(m["precision@2"] - 0.5) < 1e-9
    assert m["hit_rate@2"] == 1.0
    assert abs(m["jaccard@2"] - 1 / 3) < 1e-9


def test_prauc_between_zero_and_one():
    rng = np.random.default_rng(0)
    y = (rng.random((50, 10)) < 0.3).astype(np.float32)
    p = rng.random((50, 10)).astype(np.float32)
    m = multi_label_metrics(y, p)
    assert 0.0 <= m["prauc"] <= 1.0
