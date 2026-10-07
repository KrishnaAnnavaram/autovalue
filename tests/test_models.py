"""Quantile models, baselines, calibration and leakage (reference problems 4, 5 and 9)."""
import numpy as np
import pandas as pd
import pytest

from autovalue.explain import validation_importance
from autovalue.models import GroupMedianBaseline, LinearBaseline, QuantileModel
from autovalue.split import time_split
from autovalue.tasks import ETA, PRICE


def test_band_is_ordered(price_result):
    band = price_result.model.predict_band(price_result.split.test)
    assert (band["low"] <= band["mid"]).all() and (band["mid"] <= band["high"]).all()
    assert (band["low"] > 0).all()


def test_model_beats_group_median_baseline_on_price(price_result):
    metrics = price_result.report["metrics"]
    assert metrics["quantile_gbm"]["mae"] < metrics["group_median"]["mae"]
    assert metrics["oracle_noise_floor"]["mae"] < metrics["quantile_gbm"]["mae"]


def test_eta_model_beats_group_median_and_is_near_noise_floor(eta_result):
    metrics = eta_result.report["metrics"]
    assert metrics["quantile_gbm"]["mae"] < 0.5 * metrics["group_median"]["mae"]
    assert metrics["quantile_gbm"]["mae"] < 1.3 * metrics["oracle_noise_floor"]["mae"]


def test_calibration_moves_coverage_toward_nominal(price_result):
    band = price_result.report["band"]
    nominal = band["nominal_coverage"]
    before = abs(band["uncalibrated"]["coverage"] - nominal)
    after = abs(band["calibrated"]["coverage"] - nominal)
    assert after <= before
    assert 0.65 < band["calibrated"]["coverage"] < 0.92


def test_traffic_changes_the_eta(eta_result):
    """Reference problem 3: with clean categories, a jam gives a longer delivery than low traffic."""
    rows = eta_result.split.test.head(50).copy()
    jam = rows.assign(traffic="jam")
    low = rows.assign(traffic="low")
    model = eta_result.model
    assert (model.predict(jam) - model.predict(low)).mean() > 3.0


def test_preprocessing_is_fit_on_train_rows_only(listings):
    """Reference problem 5: a category that exists only in the test part is unknown to the encoder."""
    split = time_split(listings, "listed_at")
    train = split.train
    test = split.test.copy()
    test.loc[test.index[:5], "city"] = "atlantis"
    model = QuantileModel(PRICE, max_iter=30).fit(train)
    prep = model.middle_pipeline().named_steps["prep"]
    city_categories = prep.named_transformers_["cat"].named_steps["onehot"].categories_[
        list(model.task.feature_builder.categorical).index("city")
    ]
    assert "atlantis" not in city_categories
    numeric_medians = prep.named_transformers_["num"].named_steps["impute"].statistics_
    engine_idx = list(model.task.feature_builder.numeric).index("engine_l")
    assert numeric_medians[engine_idx] == pytest.approx(np.nanmedian(train["engine_l"]))
    assert np.isfinite(model.predict(test)).all()


def test_importance_uses_validation_rows_only(price_result):
    """Reference problem 5: the ranking is measured on the validation part, not on the test part."""
    table = validation_importance(price_result.model, price_result.split.val, n_repeats=2)
    assert {"brand", "model", "year"} & set(table.head(4)["column"])
    assert price_result.report["validation_importance"][0]["column"] in set(table["column"])


def test_all_models_of_a_task_use_the_same_rows_and_columns(price_result):
    """Reference problem 7: baselines and the model use one task object and one split."""
    rows = price_result.report["rows"]
    for m in price_result.report["metrics"].values():
        assert m["n"] == rows["test"]


def test_group_median_backs_off_to_coarser_groups(listings):
    base = GroupMedianBaseline(PRICE).fit(listings)
    unknown = listings.head(3).assign(model="unknown_model", year=1999)
    pred = base.predict(unknown)
    brand_median = np.exp(np.median(np.log(listings.loc[listings["brand"] == unknown["brand"].iloc[0], "price"])))
    assert pred[0] == pytest.approx(brand_median, rel=1e-6)


def test_linear_baseline_predicts_positive_prices(listings):
    pred = LinearBaseline(PRICE).fit(listings).predict(listings.head(20))
    assert (pred > 0).all()


def test_quantiles_must_increase():
    with pytest.raises(ValueError):
        QuantileModel(ETA, quantiles=(0.5, 0.1, 0.9))


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        QuantileModel(ETA).predict(pd.DataFrame())


def test_lightgbm_backend(deliveries):
    pytest.importorskip("lightgbm")
    split = time_split(deliveries, "ordered_at")
    model = QuantileModel(ETA, backend="lightgbm", max_iter=50).fit(split.train)
    band = model.predict_band(split.test)
    assert (band["low"] <= band["high"]).all()
