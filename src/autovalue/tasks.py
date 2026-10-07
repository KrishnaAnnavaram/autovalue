"""The two prediction tasks. Each task names its table, target, time column and features.

All models of one task (baselines and quantile models) use the same task object, so they
see the same rows and the same feature list and their scores are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .features import CarFeatures, DeliveryFeatures


@dataclass(frozen=True)
class Task:
    name: str
    table: str
    target: str
    time_col: str
    id_col: str
    oracle_col: str
    log_target: bool
    feature_builder: Callable[[], object]
    baseline_keys: tuple[tuple[str, ...], ...]
    unit: str

    def y(self, df: pd.DataFrame) -> np.ndarray:
        values = df[self.target].to_numpy(float)
        return np.log(values) if self.log_target else values

    def from_model_scale(self, values: np.ndarray) -> np.ndarray:
        return np.exp(values) if self.log_target else np.asarray(values, dtype=float)


PRICE = Task(
    name="price",
    table="listings",
    target="price",
    time_col="listed_at",
    id_col="listing_id",
    oracle_col="oracle_price",
    log_target=True,
    feature_builder=CarFeatures,
    baseline_keys=(("brand", "model", "year"), ("brand", "model"), ("brand",), ()),
    unit="currency",
)

ETA = Task(
    name="eta",
    table="deliveries",
    target="delivery_minutes",
    time_col="ordered_at",
    id_col="order_id",
    oracle_col="oracle_minutes",
    log_target=False,
    feature_builder=DeliveryFeatures,
    baseline_keys=(("traffic", "area"), ("traffic",), ()),
    unit="minutes",
)

TASKS = {"price": PRICE, "eta": ETA}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown task {name!r}, use one of {sorted(TASKS)}") from exc
