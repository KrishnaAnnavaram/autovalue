"""Time-based train / validation / test splits.

Rows are cut by timestamp, not by position, so all rows with the same timestamp stay in
one part. The validation part calibrates the price band and ranks features. The test
part is used only once, for the final report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    method: str
    cutoffs: tuple[str, str] | None = None
    notes: list[str] = field(default_factory=list)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def time_split(
    df: pd.DataFrame, time_col: str, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> Split:
    """Split ``df`` so that train < val < test in time.

    If the time column has fewer than 3 distinct values, a time split is not possible.
    Then the function uses a seeded random split and records a note.
    """
    if not 0 < val_frac < 0.5 or not 0 < test_frac < 0.5:
        raise ValueError("val_frac and test_frac must be in (0, 0.5)")
    times = pd.to_datetime(df[time_col])
    if times.nunique() < 3:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
        n_test = int(round(len(df) * test_frac))
        n_val = int(round(len(df) * val_frac))
        test_idx, val_idx, train_idx = order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :]
        return Split(
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
            method="random",
            notes=[f"{time_col} has fewer than 3 distinct values: seeded random split used"],
        )
    c1 = times.quantile(1.0 - val_frac - test_frac)
    c2 = times.quantile(1.0 - test_frac)
    train = df[times < c1]
    val = df[(times >= c1) & (times < c2)]
    test = df[times >= c2]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("time split gave an empty part: the time column has too few distinct values")
    return Split(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        method="time",
        cutoffs=(str(c1), str(c2)),
    )
