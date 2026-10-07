"""Feature builders: distance, dates as numbers (reference problem 7), stateless behaviour."""
import numpy as np
import pandas as pd
import pytest

from autovalue.features import CarFeatures, DeliveryFeatures, haversine_km


def test_haversine_one_degree_of_latitude():
    assert float(haversine_km(0, 0, 1, 0)) == pytest.approx(111.19, abs=0.05)
    assert float(haversine_km(12.97, 77.59, 12.97, 77.59)) == 0


def test_delivery_features_are_numeric_not_label_codes():
    df = pd.DataFrame(
        {
            "store_lat": [12.9716], "store_lon": [77.5946], "drop_lat": [13.0358], "drop_lon": [77.5970],
            "ordered_at": ["2023-12-09 18:30:00"], "picked_at": ["2023-12-09 18:42:00"],
            "weather": ["Stormy"], "traffic": ["Jam "], "area": ["Urban "], "vehicle": ["van"],
        }
    )
    out = DeliveryFeatures().fit_transform(df)
    assert out.loc[0, "pickup_wait_min"] == 12.0
    assert out.loc[0, "order_hour"] == 18.5
    assert out.loc[0, "is_weekend"] == 1.0  # 2023-12-09 is a Saturday
    assert out.loc[0, "is_peak"] == 1.0
    assert out.loc[0, "traffic"] == "jam"
    assert out.loc[0, "distance_km"] == pytest.approx(7.14, abs=0.05)


def test_car_features_age_and_mileage():
    df = pd.DataFrame(
        {
            "listed_at": ["2024-07-01"], "brand": ["Honda"], "model": ["City"], "year": [2019],
            "odometer_km": [55000.0], "fuel": ["Petrol"], "transmission": ["Manual"],
        }
    )
    out = CarFeatures().fit_transform(df)
    assert out.loc[0, "age_years"] == pytest.approx(5.5, abs=0.01)
    assert out.loc[0, "km_per_year"] == pytest.approx(55000 / 5.5, rel=0.01)
    assert out.loc[0, "brand_model"] == "honda|city"
    assert np.isnan(out.loc[0, "owners"])  # an absent optional column becomes NaN, not an error
    assert list(out.columns) == CarFeatures.output_columns


def test_builders_have_no_fitted_state():
    df = pd.DataFrame(
        {"listed_at": ["2024-01-01"], "brand": ["a"], "model": ["b"], "year": [2020],
         "odometer_km": [1.0], "fuel": ["petrol"], "transmission": ["manual"]}
    )
    builder = CarFeatures()
    before = dict(vars(builder))
    builder.fit(df)
    assert dict(vars(builder)) == before
