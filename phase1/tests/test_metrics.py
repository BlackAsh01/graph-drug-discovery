import numpy as np
import pytest

from dti_gt.utils.metrics import concordance_index, regression_metrics, rm2_index


def _naive_ci(y, f):
    s, n = 0.0, 0
    for i in range(len(y)):
        for j in range(len(y)):
            if y[i] > y[j]:
                n += 1
                s += 1.0 if f[i] > f[j] else 0.5 if f[i] == f[j] else 0.0
    return s / n


def test_concordance_index_matches_naive():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    f = y + rng.normal(scale=0.5, size=300)
    f[:20] = f[0]  # add prediction ties
    assert concordance_index(y, f, chunk=64) == pytest.approx(_naive_ci(y, f), abs=1e-9)


def test_perfect_prediction():
    y = np.linspace(0, 1, 50)
    m = regression_metrics(y, y, binary_threshold=0.5)
    assert m["mse"] == 0 and m["ci"] == 1.0 and m["pearson"] == pytest.approx(1.0)
    assert rm2_index(y, y) == pytest.approx(1.0)
    assert m["auroc"] == 1.0


def test_metrics_keys_and_ranges():
    rng = np.random.default_rng(1)
    y = rng.normal(11.7, 0.8, size=500)
    f = y + rng.normal(scale=0.7, size=500)
    m = regression_metrics(y, f)
    for k in ["mse", "rmse", "mae", "ci", "rm2", "pearson", "spearman", "auroc", "auprc"]:
        assert k in m and np.isfinite(m[k])
    assert 0.5 < m["ci"] <= 1.0 and 0 <= m["auroc"] <= 1
