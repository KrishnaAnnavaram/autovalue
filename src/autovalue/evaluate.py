"""Metrics, bootstrap confidence intervals and the comparison with the baselines.

A paired t-test between predictions and actual values only tests the mean error (bias).
It does not show accuracy. This module reports the bias separately, with a bootstrap
interval, and compares models with a paired bootstrap of the absolute-error difference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RegressionMetrics:
    n: int
    mae: float
    rmse: float
    mape: float
    r2: float
    bias: float  # mean of (prediction - actual)

    def as_dict(self) -> dict:
        return asdict(self)


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays with the same shape")
    err = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(err) / np.abs(y)
    return RegressionMetrics(
        n=int(y.size),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        mape=float(np.mean(ape[np.isfinite(ape)])),
        r2=r2,
        bias=float(np.mean(err)),
    )


def interval_metrics(y_true, low, high) -> dict:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)
    width = hi - lo
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(width)),
        "mean_relative_width": float(np.mean(width / np.abs(y))),
    }


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 1000, level: float = 0.95, seed: int = 42):
    """Percentile bootstrap interval of ``statistic`` over ``values``."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    stats = np.apply_along_axis(statistic, 1, v[idx])
    tail = (1.0 - level) / 2.0
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1.0 - tail))


def mae_ci(y_true, y_pred, seed: int = 42, n_boot: int = 1000):
    return bootstrap_ci(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)), seed=seed, n_boot=n_boot)


def bias_ci(y_true, y_pred, seed: int = 42, n_boot: int = 1000):
    """Interval of the mean error. An interval that contains 0 says only 'no detected bias'."""
    return bootstrap_ci(np.asarray(y_pred, float) - np.asarray(y_true, float), seed=seed, n_boot=n_boot)


def paired_mae_difference(y_true, pred_a, pred_b, seed: int = 42, n_boot: int = 1000) -> dict:
    """MAE(a) - MAE(b) with a paired bootstrap interval. A negative upper bound means a is better."""
    y = np.asarray(y_true, float)
    diff = np.abs(np.asarray(pred_a, float) - y) - np.abs(np.asarray(pred_b, float) - y)
    low, high = bootstrap_ci(diff, seed=seed, n_boot=n_boot)
    if high < 0:
        verdict = "a better"
    elif low > 0:
        verdict = "b better"
    else:
        verdict = "no clear difference"
    return {"mean_difference": float(diff.mean()), "ci_low": low, "ci_high": high, "verdict": verdict}
