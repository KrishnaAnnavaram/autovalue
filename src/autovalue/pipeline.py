"""Train, calibrate and evaluate one task. The CLI and the API call these functions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .approval import evaluate_rule, make_mispriced
from .config import Settings
from .evaluate import bias_ci, interval_metrics, mae_ci, paired_mae_difference, regression_metrics
from .explain import validation_importance
from .models import GroupMedianBaseline, LinearBaseline, QuantileModel
from .split import Split, time_split
from .tasks import Task


@dataclass
class TrainResult:
    task: Task
    model: QuantileModel
    split: Split
    report: dict


def train_and_evaluate(df: pd.DataFrame, task: Task, settings: Settings, n_boot: int = 500) -> TrainResult:
    """Fit on train, calibrate on validation, report on test. ``df`` must be validated."""
    split = time_split(df, task.time_col, seed=settings.seed)
    model = QuantileModel(
        task, settings.quantiles, backend=settings.backend, seed=settings.seed, max_iter=settings.max_iter
    ).fit(split.train)

    test = split.test
    y = test[task.target].to_numpy(float)
    uncalibrated = model.predict_band(test)
    model.calibrate(split.val)
    band = model.predict_band(test)

    group = GroupMedianBaseline(task).fit(split.train)
    linear = LinearBaseline(task, seed=settings.seed).fit(split.train)
    predictions = {
        "group_median": group.predict(test),
        "linear": linear.predict(test),
        "quantile_gbm": band["mid"].to_numpy(),
    }
    if task.oracle_col in test.columns:
        predictions["oracle_noise_floor"] = test[task.oracle_col].to_numpy(float)

    metrics = {}
    for name, pred in predictions.items():
        row = regression_metrics(y, pred).as_dict()
        row["mae_ci95"] = mae_ci(y, pred, seed=settings.seed, n_boot=n_boot)
        metrics[name] = row

    best_baseline = min(("group_median", "linear"), key=lambda k: metrics[k]["mae"])
    report = {
        "task": task.name,
        "rows": split.sizes(),
        "split": {"method": split.method, "cutoffs": split.cutoffs, "notes": split.notes},
        "metrics": metrics,
        "model_vs_best_baseline": {
            "baseline": best_baseline,
            **paired_mae_difference(y, predictions["quantile_gbm"], predictions[best_baseline], seed=settings.seed, n_boot=n_boot),
        },
        "bias_ci95": bias_ci(y, predictions["quantile_gbm"], seed=settings.seed, n_boot=n_boot),
        "band": {
            "quantiles": list(model.quantiles),
            "nominal_coverage": round(model.nominal_coverage, 4),
            "calibration_offset": model.offset_,
            "uncalibrated": interval_metrics(y, uncalibrated["low"], uncalibrated["high"]),
            "calibrated": interval_metrics(y, band["low"], band["high"]),
        },
        "validation_importance": validation_importance(model, split.val, seed=settings.seed)
        .head(10)
        .to_dict(orient="records"),
        "settings": {"seed": settings.seed, "backend": settings.backend, "max_iter": settings.max_iter},
    }
    if task.name == "price":
        priced = make_mispriced(test, price_col=task.target, seed=settings.seed)
        report["approval"] = evaluate_rule(priced["listed_price"].to_numpy(), priced["is_mispriced"].to_numpy(), band)
    return TrainResult(task, model, split, report)


def format_report(report: dict) -> str:
    """Plain-text table of a report for the terminal."""
    unit = "minutes" if report["task"] == "eta" else "currency units"
    lines = [f"task: {report['task']}   rows: {report['rows']}   split: {report['split']['method']}"]
    for note in report["split"]["notes"]:
        lines.append(f"note: {note}")
    lines.append(f"{'model':<20}{'MAE':>14}{'MAE 95% CI':>28}{'MAPE':>8}{'R2':>8}{'bias':>12}")
    for name, m in report["metrics"].items():
        ci = f"[{m['mae_ci95'][0]:,.1f}, {m['mae_ci95'][1]:,.1f}]"
        lines.append(f"{name:<20}{m['mae']:>14,.1f}{ci:>28}{m['mape']:>8.3f}{m['r2']:>8.3f}{m['bias']:>12,.1f}")
    cmp_ = report["model_vs_best_baseline"]
    lines.append(
        f"quantile_gbm - {cmp_['baseline']}: MAE difference {cmp_['mean_difference']:,.1f} {unit}, "
        f"95% CI [{cmp_['ci_low']:,.1f}, {cmp_['ci_high']:,.1f}] -> {cmp_['verdict'].replace('a ', 'model ').replace('b ', 'baseline ')}"
    )
    b = report["band"]
    lines.append(
        f"band {b['quantiles'][0]:.2f}-{b['quantiles'][2]:.2f}: coverage {b['uncalibrated']['coverage']:.3f} before, "
        f"{b['calibrated']['coverage']:.3f} after calibration (nominal {b['nominal_coverage']:.2f}), "
        f"mean relative width {b['calibrated']['mean_relative_width']:.3f}"
    )
    if "approval" in report:
        a = report["approval"]
        lines.append(
            f"approval rule: auto-approve {a['auto_approve_rate']:.3f}, precision {a['precision']:.3f}, "
            f"recall {a['recall']:.3f}, false review rate {a['false_review_rate']:.3f}"
        )
    top = ", ".join(f"{r['column']} ({r['importance']:.3f})" for r in report["validation_importance"][:5])
    lines.append(f"validation importance (top 5): {top}")
    return "\n".join(lines)


def to_jsonable(value):
    """Convert numpy scalars and tuples so that ``json.dumps`` accepts a report."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
