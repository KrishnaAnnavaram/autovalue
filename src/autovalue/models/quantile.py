"""Quantile gradient-boosting models that give a low, middle and high estimate.

One pipeline is fit for each quantile. Each pipeline holds the feature builder, the
preprocessor and the regressor, so a call to ``fit`` with the training rows fits all
the preprocessing on those rows only. ``calibrate`` widens the band on the validation
rows (conformalized quantile regression), so the band has its nominal coverage.
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from ..tasks import Task


def build_preprocessor(builder, seed: int = 42, scale: bool = False, high_card: str = "target") -> ColumnTransformer:
    """Make the column preprocessor for the columns of a feature builder.

    ``high_card`` is ``"target"`` (cross-fitted target encoding) or ``"onehot"``.
    """
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    onehot = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
        ]
    )
    parts = [("num", Pipeline(numeric), list(builder.numeric)), ("cat", onehot, list(builder.categorical))]
    if builder.high_cardinality:
        if high_card == "target":
            encoder = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("target", TargetEncoder(target_type="continuous", random_state=seed)),
                ]
            )
        else:
            encoder = onehot
        parts.append(("high_card", encoder, list(builder.high_cardinality)))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


class QuantileModel:
    """Low / middle / high quantile models for one task."""

    def __init__(
        self,
        task: Task,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        backend: str = "sklearn",
        seed: int = 42,
        max_iter: int = 300,
    ) -> None:
        low, mid, high = quantiles
        if not 0 < low < mid < high < 1:
            raise ValueError(f"quantiles must be increasing and inside (0, 1), got {quantiles}")
        self.task = task
        self.quantiles = (float(low), float(mid), float(high))
        self.backend = backend
        self.seed = seed
        self.max_iter = max_iter
        self.offset_ = 0.0
        self.calibrated_ = False

    @property
    def nominal_coverage(self) -> float:
        return self.quantiles[2] - self.quantiles[0]

    def _regressor(self, q: float):
        if self.backend == "sklearn":
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=self.seed,
            )
        if self.backend == "lightgbm":
            lgb = importlib.import_module("lightgbm")  # optional extra "boost"
            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.max_iter,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                random_state=self.seed,
                verbose=-1,
            )
        raise ValueError(f"unknown backend {self.backend!r}")

    def fit(self, train: pd.DataFrame) -> "QuantileModel":
        y = self.task.y(train)
        self.pipelines_: dict[float, Pipeline] = {}
        for q in self.quantiles:
            pipe = Pipeline(
                [
                    ("features", self.task.feature_builder()),
                    ("prep", build_preprocessor(self.task.feature_builder, seed=self.seed)),
                    ("model", self._regressor(q)),
                ]
            )
            self.pipelines_[q] = pipe.fit(train, y)
        self.n_train_ = len(train)
        self.offset_ = 0.0
        self.calibrated_ = False
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "pipelines_"):
            raise RuntimeError("the model is not fitted: call fit() first")
        raw = np.column_stack([self.pipelines_[q].predict(df) for q in self.quantiles])
        return np.sort(raw, axis=1)  # no quantile crossing

    def calibrate(self, val: pd.DataFrame) -> float:
        """Widen (or narrow) the band on validation rows. Return the offset on the model scale."""
        raw = self._raw(val)
        y = self.task.y(val)
        scores = np.maximum(raw[:, 0] - y, y - raw[:, 2])
        n = len(scores)
        alpha = 1.0 - self.nominal_coverage
        level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
        self.offset_ = float(np.quantile(scores, level, method="higher"))
        self.calibrated_ = True
        self.n_calibration_ = n
        return self.offset_

    def predict_band(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self._raw(df)
        low = raw[:, 0] - self.offset_
        high = raw[:, 2] + self.offset_
        mid = np.clip(raw[:, 1], low, high)
        f = self.task.from_model_scale
        return pd.DataFrame({"low": f(low), "mid": f(mid), "high": f(high)}, index=df.index)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_band(df)["mid"].to_numpy()

    def middle_pipeline(self) -> Pipeline:
        return self.pipelines_[self.quantiles[1]]
