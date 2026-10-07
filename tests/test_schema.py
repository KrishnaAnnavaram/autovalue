"""Validation, category normalisation (reference problem 3) and coercion of new rows."""
import numpy as np
import pandas as pd
import pytest

from autovalue.schema import SchemaError, coerce_input, normalize_category, validate
from autovalue.synthetic import make_deliveries, make_listings


def test_trailing_spaces_and_case_are_removed():
    values = pd.Series(["High ", "Jam ", " low", "MEDIUM", "Metropolitian ", "Semi-Urban", "", None])
    out = normalize_category(values).tolist()
    assert out[:6] == ["high", "jam", "low", "medium", "metropolitan", "semi_urban"]
    assert pd.isna(out[6]) and pd.isna(out[7])


def test_dirty_delivery_traffic_maps_to_four_levels():
    clean, _ = validate(make_deliveries(500, seed=4), "deliveries")
    assert set(clean["traffic"]) == {"low", "medium", "high", "jam"}
    assert "metropolitan" in set(clean["area"])


def test_missing_required_column_raises():
    df = make_listings(50, seed=1).drop(columns=["price"])
    with pytest.raises(SchemaError, match="price"):
        validate(df, "listings")


def test_invalid_rows_and_duplicates_are_dropped_with_reasons():
    raw = make_listings(1000, seed=2)
    clean, report = validate(raw, "listings")
    assert report.dropped["duplicate listing_id"] > 0
    assert report.dropped["odometer_km outside [0, 2000000]"] > 0
    assert clean["listing_id"].is_unique
    assert (clean["odometer_km"] >= 0).all()
    assert report.rows_out == len(clean) < len(raw)


def test_model_year_after_listing_is_dropped():
    df = make_listings(20, seed=3, dirty=False)
    df.loc[0, "year"] = 2030
    clean, report = validate(df, "listings")
    assert report.dropped["model year after the listing date"] == 1
    assert len(clean) == 19


def test_pickup_before_order_is_dropped():
    df = make_deliveries(20, seed=3, dirty=False)
    df.loc[0, "picked_at"] = "2000-01-01 00:00:00"
    _, report = validate(df, "deliveries")
    assert report.dropped["pickup before order"] == 1


def test_optional_value_out_of_range_becomes_nan_with_warning():
    df = make_listings(20, seed=3, dirty=False)
    df.loc[0, "engine_l"] = 40.0
    clean, report = validate(df, "listings")
    assert np.isnan(clean.loc[clean["listing_id"] == df.loc[0, "listing_id"], "engine_l"]).all()
    assert any("engine_l" in w for w in report.warnings)


def test_unknown_table_raises():
    with pytest.raises(SchemaError):
        validate(pd.DataFrame({"a": [1]}), "cars")


def test_coerce_input_needs_required_features_but_not_target():
    row = {
        "listed_at": "2024-01-01", "brand": "Honda", "model": "City", "year": 2018,
        "odometer_km": 40000, "fuel": "Petrol", "transmission": "Manual",
    }
    frame = coerce_input([row], "listings", "price")
    assert frame.loc[0, "brand"] == "honda"
    del row["fuel"]
    with pytest.raises(SchemaError, match="fuel"):
        coerce_input([row], "listings", "price")
