"""Read CSV files and map public datasets to the project schema.

Two adapters are included:

* ``from_cardekho``: the public "Vehicle dataset from CarDekho" used-car table
  (columns ``name, year, selling_price, km_driven, fuel, seller_type, transmission, owner``).
* ``from_amazon_delivery``: the public "Amazon Delivery Dataset" order table
  (columns ``Order_ID, Store_Latitude, ..., Weather, Traffic, Vehicle, Area, Delivery_Time``).

Read ``data/README.md`` for the download links and the terms of use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ValidationReport, validate

OWNER_WORDS = {"first owner": 1, "second owner": 2, "third owner": 3, "fourth & above owner": 4, "test drive car": 1}


def load_table(path: str | Path, table: str) -> tuple[pd.DataFrame, ValidationReport]:
    """Read a CSV in the project schema and validate it."""
    df = pd.read_csv(path)
    return validate(df, table)


def from_cardekho(df: pd.DataFrame, snapshot_date: str = "2020-06-01") -> pd.DataFrame:
    """Map the CarDekho table to the listings schema.

    The source has no listing date. All rows get ``snapshot_date``, so the time split
    falls back to a seeded random split, and the report says so.
    """
    names = df["name"].astype(str).str.strip().str.split()
    owners = df["owner"].astype(str).str.strip().str.lower().map(OWNER_WORDS)
    return pd.DataFrame(
        {
            "listing_id": [f"cd-{i:06d}" for i in range(len(df))],
            "listed_at": snapshot_date,
            "brand": names.str[0],
            "model": names.str[1],
            "year": df["year"],
            "odometer_km": df["km_driven"],
            "fuel": df["fuel"],
            "transmission": df["transmission"],
            "owners": owners,
            "price": df["selling_price"],
        }
    )


def _combine(date: pd.Series, clock: pd.Series) -> pd.Series:
    return pd.to_datetime(date.astype(str).str.strip() + " " + clock.astype(str).str.strip(), errors="coerce", dayfirst=False)


def from_amazon_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Amazon delivery table to the deliveries schema.

    * ``Order_Date`` + ``Order_Time`` gives ``ordered_at``.
    * ``Order_Date`` + ``Pickup_Time`` gives ``picked_at``. If that time is earlier than
      the order time, the pickup was after midnight, so one day is added.
    * Negative store latitudes in the source are sign errors for Indian stores, so the
      adapter uses the absolute value.
    """
    ordered = _combine(df["Order_Date"], df["Order_Time"])
    picked = _combine(df["Order_Date"], df["Pickup_Time"])
    picked = picked.where(~(picked < ordered), picked + pd.Timedelta(days=1))
    return pd.DataFrame(
        {
            "order_id": df["Order_ID"],
            "store_lat": np.abs(pd.to_numeric(df["Store_Latitude"], errors="coerce")),
            "store_lon": np.abs(pd.to_numeric(df["Store_Longitude"], errors="coerce")),
            "drop_lat": df["Drop_Latitude"],
            "drop_lon": df["Drop_Longitude"],
            "ordered_at": ordered,
            "picked_at": picked,
            "weather": df["Weather"],
            "traffic": df["Traffic"],
            "area": df["Area"],
            "vehicle": df["Vehicle"],
            "delivery_minutes": df["Delivery_Time"],
        }
    )


ADAPTERS = {"cardekho": ("listings", from_cardekho), "amazon-delivery": ("deliveries", from_amazon_delivery)}
