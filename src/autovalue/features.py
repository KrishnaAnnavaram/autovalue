"""Feature builders. They are stateless scikit-learn transformers, so they live inside
the model pipeline and see exactly the same rows at fit time and at predict time.

Dates and times become numbers (age, hour, weekday). They are never label-encoded strings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schema import normalize_category

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between two arrays of points in degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _col(X: pd.DataFrame, name: str) -> pd.Series:
    if name in X.columns:
        return X[name]
    return pd.Series(np.nan, index=X.index, dtype=float)


class _Stateless(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):  # noqa: D401 - scikit-learn API
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns, dtype=object)


class CarFeatures(_Stateless):
    """Listing columns to model features: age, mileage per year and a brand-model key."""

    categorical = ["brand", "fuel", "transmission", "body_type", "city"]
    high_cardinality = ["brand_model"]
    numeric = ["age_years", "odometer_km", "km_per_year", "owners", "accidents", "engine_l", "listing_time"]
    output_columns = numeric + categorical + high_cardinality

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        listed = pd.to_datetime(X["listed_at"])
        age = (listed.dt.year + listed.dt.dayofyear / 365.25 - X["year"].astype(float)).clip(lower=0.0)
        out = pd.DataFrame(index=X.index)
        out["age_years"] = age
        out["odometer_km"] = X["odometer_km"].astype(float)
        out["km_per_year"] = X["odometer_km"].astype(float) / age.clip(lower=0.5)
        out["owners"] = pd.to_numeric(_col(X, "owners"), errors="coerce")
        out["accidents"] = _col(X, "accidents").map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}).astype(float)
        out["engine_l"] = pd.to_numeric(_col(X, "engine_l"), errors="coerce")
        # years since 2000: a linear model can follow a market trend with this column
        out["listing_time"] = listed.dt.year - 2000 + listed.dt.dayofyear / 365.25
        for col in self.categorical:
            out[col] = normalize_category(_col(X, col))
        out["brand_model"] = normalize_category(X["brand"]).astype(str) + "|" + normalize_category(X["model"]).astype(str)
        return out[self.output_columns]


class DeliveryFeatures(_Stateless):
    """Order columns to model features: distance, pickup wait, hour, weekday and peak flag."""

    categorical = ["weather", "traffic", "area", "vehicle"]
    high_cardinality: list[str] = []
    numeric = ["distance_km", "pickup_wait_min", "order_hour", "order_weekday", "is_weekend", "is_peak"]
    output_columns = numeric + categorical

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ordered = pd.to_datetime(X["ordered_at"])
        picked = pd.to_datetime(X["picked_at"])
        out = pd.DataFrame(index=X.index)
        out["distance_km"] = haversine_km(X["store_lat"], X["store_lon"], X["drop_lat"], X["drop_lon"])
        out["pickup_wait_min"] = (picked - ordered).dt.total_seconds() / 60.0
        out["order_hour"] = ordered.dt.hour + ordered.dt.minute / 60.0
        out["order_weekday"] = ordered.dt.weekday.astype(float)
        out["is_weekend"] = (ordered.dt.weekday >= 5).astype(float)
        out["is_peak"] = ordered.dt.hour.between(17, 21).astype(float)
        for col in self.categorical:
            out[col] = normalize_category(_col(X, col))
        return out[self.output_columns]
