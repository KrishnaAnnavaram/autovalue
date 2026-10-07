"""Permutation importance on the validation rows.

Importance is measured on the validation part, never on the test part. Thus a
feature ranking (or a feature selection that uses it) cannot see the test rows.
"""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from .models.quantile import QuantileModel


def input_columns(model: QuantileModel, df: pd.DataFrame) -> list[str]:
    from .schema import schema_for

    names = [c.name for c in schema_for(model.task.table) if c.name not in {model.task.target, model.task.id_col}]
    return [n for n in names if n in df.columns]


def validation_importance(model: QuantileModel, val: pd.DataFrame, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """Mean increase of the absolute error of the middle model when one input column is shuffled."""
    cols = input_columns(model, val)
    pipe = model.middle_pipeline()
    y = model.task.y(val)
    result = permutation_importance(
        pipe,
        val[cols],
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=seed,
    )
    table = pd.DataFrame(
        {"column": cols, "importance": result.importances_mean, "std": result.importances_std}
    ).sort_values("importance", ascending=False, ignore_index=True)
    table["importance"] = table["importance"].astype(float)
    return table


def top_columns(table: pd.DataFrame, k: int) -> list[str]:
    return list(table.loc[table["importance"] > 0, "column"].head(k))
