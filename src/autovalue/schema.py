"""Column schemas and row validation for the two tables.

The listings table and the deliveries table are separate. Each table has its own
schema, its own target and its own key. Nothing joins them by row position.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """The table cannot be used: a required column is absent or no valid row is left."""


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "id", "num", "cat", "bool", "date", "datetime"
    required: bool = True  # a row with a missing value is dropped
    low: float | None = None
    high: float | None = None


LISTING_COLUMNS: tuple[Column, ...] = (
    Column("listing_id", "id"),
    Column("listed_at", "date"),
    Column("brand", "cat"),
    Column("model", "cat"),
    Column("year", "num", low=1980, high=2100),
    Column("odometer_km", "num", low=0, high=2_000_000),
    Column("fuel", "cat"),
    Column("transmission", "cat"),
    Column("owners", "num", required=False, low=1, high=15),
    Column("accidents", "bool", required=False),
    Column("engine_l", "num", required=False, low=0.5, high=8.5),
    Column("body_type", "cat", required=False),
    Column("city", "cat", required=False),
    Column("price", "num", low=1, high=1e9),
)

DELIVERY_COLUMNS: tuple[Column, ...] = (
    Column("order_id", "id"),
    Column("store_lat", "num", low=-90, high=90),
    Column("store_lon", "num", low=-180, high=180),
    Column("drop_lat", "num", low=-90, high=90),
    Column("drop_lon", "num", low=-180, high=180),
    Column("ordered_at", "datetime"),
    Column("picked_at", "datetime"),
    Column("weather", "cat", required=False),
    Column("traffic", "cat"),
    Column("area", "cat", required=False),
    Column("vehicle", "cat", required=False),
    Column("delivery_minutes", "num", low=1, high=24 * 60),
)

# Spelling variants that the source data contains. Keys are normalised values.
CATEGORY_ALIASES: dict[str, str] = {
    "metropolitian": "metropolitan",
    "semi-urban": "semi_urban",
    "semi_urban": "semi_urban",
    "semiurban": "semi_urban",
    "traffic_jam": "jam",
    "manual_transmission": "manual",
    "automatic_transmission": "automatic",
    "nan": "",
    "none": "",
    "null": "",
}

_TRUE = {"true", "yes", "y", "1", "t"}
_FALSE = {"false", "no", "n", "0", "f"}


def normalize_category(values: pd.Series) -> pd.Series:
    """Strip, lower-case and join words with ``_``. Map known aliases. Empty text becomes NaN.

    The source data has values such as ``'High '`` and ``'Jam '`` with a trailing space.
    After this function they are ``'high'`` and ``'jam'``.
    """
    out = values.astype("string").str.strip().str.lower()
    out = out.map(lambda v: v if v is pd.NA else re.sub(r"\s+", "_", v), na_action="ignore")
    out = out.replace(CATEGORY_ALIASES)
    out = out.replace("", pd.NA)
    return out.astype(object).where(out.notna(), np.nan)


def _to_bool(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    return text.map(lambda v: True if v in _TRUE else (False if v in _FALSE else np.nan), na_action="ignore")


@dataclass
class ValidationReport:
    table: str
    rows_in: int
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def drop(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    def summary(self) -> str:
        lines = [f"{self.table}: {self.rows_in} rows in, {self.rows_out} rows out"]
        for reason, count in sorted(self.dropped.items()):
            lines.append(f"  dropped {count:>6}  {reason}")
        for text in self.warnings:
            lines.append(f"  warning: {text}")
        return "\n".join(lines)


def _coerce(df: pd.DataFrame, columns: tuple[Column, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col.name not in out.columns:
            out[col.name] = np.nan
        series = out[col.name]
        if col.kind == "num":
            out[col.name] = pd.to_numeric(series, errors="coerce").astype(float)
        elif col.kind == "cat":
            out[col.name] = normalize_category(series)
        elif col.kind == "bool":
            out[col.name] = _to_bool(series)
        elif col.kind in {"date", "datetime"}:
            out[col.name] = pd.to_datetime(series, errors="coerce")
        elif col.kind == "id":
            out[col.name] = series.astype("string").str.strip().astype(object)
    return out


def validate(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate a table and return the clean rows and a report of each dropped row.

    ``table`` is ``"listings"`` or ``"deliveries"``. Columns that are not in the schema
    pass through without change (for example ``oracle_price`` in synthetic data).
    """
    columns = schema_for(table)
    report = ValidationReport(table=table, rows_in=len(df))
    required = [c.name for c in columns if c.required]
    absent = [name for name in required if name not in df.columns]
    if absent:
        raise SchemaError(f"{table}: required columns are absent: {absent}")

    out = _coerce(df, columns)
    for col in columns:
        values = out[col.name]
        if col.required:
            missing = values.isna()
            report.drop(f"missing {col.name}", missing.sum())
            out = out[~missing]
            values = out[col.name]
        if col.kind == "num" and (col.low is not None or col.high is not None):
            low = -np.inf if col.low is None else col.low
            high = np.inf if col.high is None else col.high
            bad = values.notna() & ((values < low) | (values > high))
            if col.required:
                report.drop(f"{col.name} outside [{col.low}, {col.high}]", bad.sum())
                out = out[~bad]
            elif bad.any():
                report.warnings.append(f"{int(bad.sum())} values of {col.name} outside range set to NaN")
                out.loc[bad, col.name] = np.nan

    id_col = columns[0].name
    dup = out.duplicated(subset=[id_col], keep="first")
    report.drop(f"duplicate {id_col}", dup.sum())
    out = out[~dup]

    if table == "listings":
        future = out["year"] > out["listed_at"].dt.year + 1
        report.drop("model year after the listing date", future.sum())
        out = out[~future]
        out["year"] = out["year"].round().astype(int)
    else:
        early = out["picked_at"] < out["ordered_at"]
        report.drop("pickup before order", early.sum())
        out = out[~early]

    report.rows_out = len(out)
    if out.empty:
        raise SchemaError(f"{table}: no valid rows left after validation")
    return out.reset_index(drop=True), report


def coerce_input(records: list[dict] | pd.DataFrame, table: str, target: str) -> pd.DataFrame:
    """Coerce new rows for prediction. The target and the ID are not necessary.

    Unlike :func:`validate`, this function drops no row. A required feature that is
    absent or not valid raises :class:`SchemaError` with the column name.
    """
    columns = schema_for(table)
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
    if df.empty:
        raise SchemaError("no input rows")
    id_col = columns[0].name
    if id_col not in df.columns:
        df[id_col] = [f"input-{i}" for i in range(len(df))]
    out = _coerce(df, columns)
    for col in columns:
        if col.required and col.name not in {target, id_col} and out[col.name].isna().any():
            raise SchemaError(f"{table}: column {col.name!r} is absent or not valid in the input")
    return out


def schema_for(table: str) -> tuple[Column, ...]:
    if table == "listings":
        return LISTING_COLUMNS
    if table == "deliveries":
        return DELIVERY_COLUMNS
    raise SchemaError(f"unknown table {table!r}, use 'listings' or 'deliveries'")
