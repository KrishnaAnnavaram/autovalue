"""Metrics and the replacement of the paired t-test (reference problem 6)."""
import numpy as np
import pytest

from autovalue.evaluate import bias_ci, interval_metrics, mae_ci, paired_mae_difference, regression_metrics


def test_regression_metrics_known_values():
    m = regression_metrics([100, 200, 300], [110, 190, 330])
    assert m.mae == pytest.approx(50 / 3)
    assert m.bias == pytest.approx(10.0)
    assert m.mape == pytest.approx((0.1 + 0.05 + 0.1) / 3)
    assert m.rmse == pytest.approx(np.sqrt((100 + 100 + 900) / 3))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])


def test_unbiased_but_bad_predictor_is_not_called_accurate():
    """A t-test of mean error accepts this predictor. MAE and the bias interval tell the truth."""
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 2000)
    pred = y + rng.choice([-50.0, 50.0], size=y.size)  # mean error near 0, every error is 50
    low, high = bias_ci(y, pred, n_boot=300)
    assert low < 0 < high  # no detected bias ...
    assert regression_metrics(y, pred).mae == pytest.approx(50.0)  # ... but large errors


def test_biased_predictor_is_detected():
    y = np.arange(1, 501, dtype=float)
    low, _ = bias_ci(y, y + 5 + np.sin(y), n_boot=300)
    assert low > 0


def test_mae_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    pred = y + rng.normal(0, 1, 300)
    low, high = mae_ci(y, pred, n_boot=300)
    assert low < np.mean(np.abs(pred - y)) < high


def test_paired_difference_picks_the_better_model():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 500)
    good = y + rng.normal(0, 0.1, 500)
    bad = y + rng.normal(0, 1.0, 500)
    assert paired_mae_difference(y, good, bad, n_boot=300)["verdict"] == "a better"
    assert paired_mae_difference(y, bad, good, n_boot=300)["verdict"] == "b better"
    assert paired_mae_difference(y, good, good, n_boot=300)["verdict"] == "no clear difference"


def test_interval_metrics():
    out = interval_metrics([1, 2, 3, 4], [0, 0, 0, 5], [2, 2, 2, 6])
    assert out["coverage"] == 0.5
    assert out["mean_width"] == 1.75
