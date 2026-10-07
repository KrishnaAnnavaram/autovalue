"""Two baselines that each model must beat.

* ``GroupMedianBaseline``: the median target of the most specific group with enough
  training rows (for price: brand-model-year, then brand-model, then brand, then all).
* ``LinearBaseline``: ridge regression on the same features as the quantile model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ..schema import normalize_category
from ..tasks import Task
from .quantile import build_preprocessor


def _keys(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    if not cols:
        return pd.Series("*", index=df.index)
    parts = []
    for col in cols:
        values = df[col]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            values = values.round().astype("Int64")
        else:
            values = normalize_category(values)
        parts.append(values.astype(str))
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key


class GroupMedianBaseline:
    def __init__(self, task: Task, min_rows: int = 5) -> None:
        self.task = task
        self.min_rows = min_rows

    def fit(self, train: pd.DataFrame) -> "GroupMedianBaseline":
        y = pd.Series(self.task.y(train), index=train.index)
        self.tables_ = []
        for cols in self.task.baseline_keys:
            grouped = y.groupby(_keys(train, cols))
            stats = grouped.median()[grouped.size() >= (self.min_rows if cols else 1)]
            self.tables_.append((cols, stats.to_dict()))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        for cols, table in self.tables_:
            keys = _keys(df, cols).to_numpy()
            fill = np.isnan(out)
            out[fill] = [table.get(k, np.nan) for k in keys[fill]]
        return self.task.from_model_scale(out)


class LinearBaseline:
    def __init__(self, task: Task, alpha: float = 1.0, seed: int = 42) -> None:
        self.task = task
        self.alpha = alpha
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "LinearBaseline":
        self.pipeline_ = Pipeline(
            [
                ("features", self.task.feature_builder()),
                ("prep", build_preprocessor(self.task.feature_builder, seed=self.seed, scale=True, high_card="onehot")),
                ("model", Ridge(alpha=self.alpha)),
            ]
        ).fit(train, self.task.y(train))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.task.from_model_scale(self.pipeline_.predict(df))
