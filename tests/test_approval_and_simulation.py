"""The approval rule (reference problem 9) and the negotiation (reference problem 8)."""
import numpy as np
import pandas as pd
import pytest

from autovalue.approval import Decision, decide, decide_many, evaluate_rule, make_mispriced
from autovalue.schema import coerce_input
from autovalue.simulate import simulate_negotiation


def test_decide_boundaries():
    assert decide(100, 90, 100, 110).decision is Decision.AUTO_APPROVE
    assert decide(90, 90, 100, 110).decision is Decision.AUTO_APPROVE
    assert decide(89, 90, 100, 110).decision is Decision.REVIEW_LOW
    assert decide(111, 90, 100, 110).decision is Decision.REVIEW_HIGH
    assert decide(111, 90, 100, 110).as_dict()["decision"] == "review_high"


def test_decide_rejects_bad_input():
    with pytest.raises(ValueError):
        decide(0, 90, 100, 110)
    with pytest.raises(ValueError):
        decide(100, 110, 100, 90)


def test_decide_many_matches_decide():
    band = pd.DataFrame({"low": [90, 90, 90], "mid": [100, 100, 100], "high": [110, 110, 110]})
    assert list(decide_many(np.array([80, 100, 120]), band)) == ["review_low", "auto_approve", "review_high"]


def test_rule_finds_mispriced_listings(price_result):
    approval = price_result.report["approval"]
    assert approval["recall"] > 0.75
    assert approval["precision"] > 0.5
    assert 0.0 < approval["false_review_rate"] < 0.35


def test_mispriced_set_is_seeded_and_labels_a_fraction():
    df = pd.DataFrame({"price": np.full(1000, 100.0)})
    a = make_mispriced(df, seed=3)
    b = make_mispriced(df, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert 0.25 < a["is_mispriced"].mean() < 0.35
    assert (a.loc[~a["is_mispriced"], "listed_price"] == 100).all()


def test_evaluate_rule_counts():
    band = pd.DataFrame({"low": [90.0] * 4, "mid": [100.0] * 4, "high": [110.0] * 4})
    out = evaluate_rule(np.array([50, 100, 105, 200]), np.array([True, False, True, False]), band)
    assert out["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_simulation_is_seeded_and_uses_the_band():
    a = simulate_negotiation(90, 100, 120, ask=130, seed=4)
    b = simulate_negotiation(90, 100, 120, ask=130, seed=4)
    assert a == b
    assert 90 <= a.mean_deal_price <= 130
    cheap = simulate_negotiation(45, 50, 60, ask=130, seed=4)
    assert cheap.deal_rate < a.deal_rate  # the band of the car changes the result


def test_simulation_low_ask_does_not_move():
    out = simulate_negotiation(90, 100, 120, ask=95, episodes=500, seed=1)
    assert out.mean_deal_price == pytest.approx(95)


def test_simulation_prices_the_given_car(price_result):
    """Reference problem 8: the band comes from the car that the user gives, not from a training row."""
    model = price_result.model
    old = {"listed_at": "2024-06-01", "brand": "maruti", "model": "swift", "year": 2010, "odometer_km": 150000,
           "fuel": "petrol", "transmission": "manual"}
    new = {**old, "brand": "bmw", "model": "5_series", "year": 2023, "odometer_km": 9000}
    bands = model.predict_band(coerce_input([old, new], "listings", "price"))
    assert bands.loc[1, "mid"] > 3 * bands.loc[0, "mid"]
    sims = [simulate_negotiation(r.low, r.mid, r.high, ask=r.high, seed=7) for r in bands.itertuples()]
    assert sims[1].mean_deal_price > sims[0].mean_deal_price


def test_simulation_rejects_bad_band():
    with pytest.raises(ValueError):
        simulate_negotiation(120, 100, 90, ask=100)
