"""Seeded synthetic generators for the two tables.

The generators exist so that the demo and the tests run with no download.
They use only ``numpy.random.Generator`` with an explicit seed. They never use the
built-in ``hash()``, so the labels are identical in each process and on each machine.

Each table also gets an ``oracle_*`` column: the label without noise. The evaluation
uses it to report the noise floor, that is, the best error that any model can get.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_NOISE_SIGMA = 0.12  # log-normal noise on price
DELIVERY_NOISE_SD = 6.0  # minutes, normal noise on delivery time

BRANDS: dict[str, tuple[float, dict[str, float]]] = {
    # brand: (base price, {model: multiplier})
    "maruti": (520_000, {"swift": 1.0, "baleno": 1.12, "dzire": 1.05, "ertiga": 1.35}),
    "hyundai": (640_000, {"i20": 1.0, "creta": 1.65, "verna": 1.3, "venue": 1.25}),
    "honda": (700_000, {"city": 1.15, "amaze": 0.95, "jazz": 0.9}),
    "toyota": (980_000, {"innova": 1.6, "corolla": 1.25, "glanza": 0.75, "fortuner": 2.6}),
    "tata": (560_000, {"nexon": 1.25, "tiago": 0.8, "harrier": 1.9}),
    "mahindra": (820_000, {"xuv500": 1.45, "scorpio": 1.35, "thar": 1.5}),
    "bmw": (3_200_000, {"3_series": 1.0, "x1": 1.05, "5_series": 1.6}),
    "mercedes": (3_600_000, {"c_class": 1.0, "e_class": 1.5, "gla": 0.95}),
}
LUXURY = ("bmw", "mercedes")
FUEL = {"petrol": 1.0, "diesel": 1.08, "cng": 0.92, "electric": 1.25}
TRANSMISSION = {"manual": 1.0, "automatic": 1.09}
BODY = {"hatchback": 0.95, "sedan": 1.0, "suv": 1.1, "muv": 1.03}
CITY = {"delhi": 1.0, "mumbai": 1.04, "bengaluru": 1.05, "chennai": 1.0, "kolkata": 0.96, "pune": 1.02}

CITY_CENTRES = {
    "bengaluru": (12.97, 77.59),
    "mumbai": (19.07, 72.87),
    "delhi": (28.61, 77.21),
    "hyderabad": (17.38, 78.48),
    "pune": (18.52, 73.86),
}
TRAFFIC = {"low": 1.0, "medium": 1.15, "high": 1.35, "jam": 1.6}
WEATHER = {"sunny": 1.0, "cloudy": 1.03, "windy": 1.05, "fog": 1.12, "stormy": 1.15, "sandstorms": 1.18}
AREA = {"urban": 4.0, "metropolitan": 7.0, "semi_urban": 2.0, "other": 3.0}
VEHICLE = {"motorcycle": 1.0, "scooter": 1.06, "van": 1.12}


def _pick(rng: np.random.Generator, options, n: int, p=None) -> np.ndarray:
    return rng.choice(np.array(list(options), dtype=object), size=n, p=p)


def oracle_price(df: pd.DataFrame) -> np.ndarray:
    """Noise-free price of each listing. ``df`` holds normalised category values."""
    base = np.array([BRANDS[b][0] * BRANDS[b][1][m] for b, m in zip(df["brand"], df["model"])])
    listed = pd.to_datetime(df["listed_at"])
    age = np.clip(listed.dt.year + listed.dt.dayofyear / 365.25 - df["year"], 0, None).to_numpy(float)
    owners = df["owners"].fillna(1).to_numpy(float)
    accidents = df["accidents"].fillna(False).astype(bool).to_numpy()
    years_since_2021 = (listed - pd.Timestamp("2021-01-01")).dt.days.to_numpy(float) / 365.25
    luxury = df["brand"].isin(LUXURY).to_numpy()
    # non-linear parts: luxury cars lose value faster, and the odometer effect saturates
    rate = np.where(luxury, 0.16, 0.09)
    km = df["odometer_km"].to_numpy(float)
    price = (
        base
        * np.exp(-rate * age)
        * np.exp(-0.18 * np.log1p(km / 20_000.0))
        * df["fuel"].map(FUEL).to_numpy(float)
        * df["transmission"].map(TRANSMISSION).to_numpy(float)
        * df["body_type"].map(BODY).fillna(1.0).to_numpy(float)
        * df["city"].map(CITY).fillna(1.0).to_numpy(float)
        * (1.0 - 0.05 * (owners - 1))
        * np.where(accidents, np.where(luxury, 0.78, 0.9), 1.0)
        * (1.0 + 0.04 * years_since_2021)  # slow market drift, so a time split matters
    )
    return price


def make_listings(n: int = 4000, seed: int = 42, dirty: bool = True) -> pd.DataFrame:
    """Make ``n`` used-car listings with a price label.

    If ``dirty`` is true, the table also gets the defects that real exports have:
    padded or mixed-case categories, a few missing optional values, duplicate IDs
    and rows with an impossible odometer. Validation removes or repairs them.
    """
    rng = np.random.default_rng(seed)
    brand = _pick(rng, BRANDS, n, p=[0.2, 0.18, 0.12, 0.13, 0.12, 0.1, 0.08, 0.07])
    model = np.array([rng.choice(list(BRANDS[b][1])) for b in brand], dtype=object)
    start, end = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-31")
    days = rng.integers(0, (end - start).days + 1, size=n)
    listed_at = start + pd.to_timedelta(days, unit="D")
    age_years = rng.integers(0, 16, size=n)
    year = listed_at.year.to_numpy() - age_years
    odometer = np.clip(age_years * rng.uniform(7_000, 17_000, size=n) + rng.normal(4_000, 2_500, size=n), 50, None)
    df = pd.DataFrame(
        {
            "listing_id": [f"L{seed:03d}-{i:06d}" for i in range(n)],
            "listed_at": listed_at.strftime("%Y-%m-%d"),
            "brand": brand,
            "model": model,
            "year": year,
            "odometer_km": odometer.round(0),
            "fuel": _pick(rng, FUEL, n, p=[0.5, 0.3, 0.12, 0.08]),
            "transmission": _pick(rng, TRANSMISSION, n, p=[0.62, 0.38]),
            "owners": rng.choice([1, 2, 3, 4], size=n, p=[0.55, 0.3, 0.11, 0.04]).astype(float),
            "accidents": rng.random(n) < 0.15,
            "engine_l": np.round(rng.uniform(1.0, 3.0, size=n), 1),
            "body_type": _pick(rng, BODY, n),
            "city": _pick(rng, CITY, n),
        }
    )
    clean_price = oracle_price(df)
    df["price"] = np.round(clean_price * np.exp(rng.normal(0.0, PRICE_NOISE_SIGMA, size=n)), -2)
    df["oracle_price"] = clean_price
    if dirty:
        df = _add_listing_defects(df, rng)
    return df


def _add_listing_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()
    df["owners"] = df["owners"].astype(object)
    pad = rng.random(n) < 0.1
    df.loc[pad, "fuel"] = df.loc[pad, "fuel"].str.title() + " "
    upper = rng.random(n) < 0.05
    df.loc[upper, "brand"] = df.loc[upper, "brand"].str.upper()
    miss = rng.random(n) < 0.03
    df.loc[miss, "engine_l"] = np.nan
    df.loc[rng.random(n) < 0.02, "body_type"] = None
    bad = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[bad, "odometer_km"] = -1.0
    dups = df.sample(n=max(1, n // 100), random_state=int(rng.integers(0, 2**31 - 1)))
    return pd.concat([df, dups], ignore_index=True)


def oracle_minutes(df: pd.DataFrame) -> np.ndarray:
    """Noise-free delivery time in minutes. ``df`` holds normalised category values."""
    from .features import haversine_km

    dist = haversine_km(df["store_lat"], df["store_lon"], df["drop_lat"], df["drop_lon"])
    ordered = pd.to_datetime(df["ordered_at"])
    wait = (pd.to_datetime(df["picked_at"]) - ordered).dt.total_seconds().to_numpy() / 60.0
    hour = ordered.dt.hour.to_numpy()
    peak = ((hour >= 17) & (hour <= 21)).astype(float)
    travel = (
        3.2
        * dist
        * df["traffic"].map(TRAFFIC).to_numpy(float)
        * df["weather"].map(WEATHER).fillna(1.05).to_numpy(float)
        * df["vehicle"].map(VEHICLE).fillna(1.0).to_numpy(float)
    )
    return 12.0 + wait + travel + df["area"].map(AREA).fillna(3.0).to_numpy(float) + 9.0 * peak


def make_deliveries(n: int = 4000, seed: int = 7, dirty: bool = True) -> pd.DataFrame:
    """Make ``n`` delivery orders with a delivery-time label in minutes.

    If ``dirty`` is true, ``traffic`` and ``area`` values get a trailing space and
    ``metropolitian`` gets the source misspelling, as in the public delivery data.
    """
    rng = np.random.default_rng(seed)
    city = _pick(rng, CITY_CENTRES, n)
    centre = np.array([CITY_CENTRES[c] for c in city])
    store = centre + rng.normal(0, 0.05, size=(n, 2))
    drop = store + rng.uniform(-0.12, 0.12, size=(n, 2))
    start = pd.Timestamp("2023-01-01")
    minutes = rng.integers(0, 365 * 24 * 60, size=n)
    ordered = start + pd.to_timedelta(minutes, unit="min")
    ordered = ordered.floor("5min")
    picked = ordered + pd.to_timedelta(rng.integers(5, 16, size=n), unit="min")
    df = pd.DataFrame(
        {
            "order_id": [f"O{seed:03d}-{i:06d}" for i in range(n)],
            "store_lat": store[:, 0].round(6),
            "store_lon": store[:, 1].round(6),
            "drop_lat": drop[:, 0].round(6),
            "drop_lon": drop[:, 1].round(6),
            "ordered_at": ordered.strftime("%Y-%m-%d %H:%M:%S"),
            "picked_at": picked.strftime("%Y-%m-%d %H:%M:%S"),
            "weather": _pick(rng, WEATHER, n),
            "traffic": _pick(rng, TRAFFIC, n, p=[0.35, 0.25, 0.2, 0.2]),
            "area": _pick(rng, AREA, n, p=[0.3, 0.5, 0.1, 0.1]),
            "vehicle": _pick(rng, VEHICLE, n, p=[0.6, 0.3, 0.1]),
        }
    )
    clean = oracle_minutes(df)
    df["delivery_minutes"] = np.clip(np.round(clean + rng.normal(0, DELIVERY_NOISE_SD, size=n)), 5, None)
    df["oracle_minutes"] = clean
    if dirty:
        df["traffic"] = df["traffic"].str.title() + " "
        df["area"] = df["area"].replace({"metropolitan": "Metropolitian"}).str.title() + " "
        df.loc[rng.random(n) < 0.02, "weather"] = None
    return df
