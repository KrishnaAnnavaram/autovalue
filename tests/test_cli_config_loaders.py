"""CLI, settings, adapters and the optional HTTP API."""
import json

import pandas as pd
import pytest

from autovalue import bundle
from autovalue.cli import EXAMPLE_CAR, EXAMPLE_ORDER, main
from autovalue.config import ConfigError, Settings
from autovalue.loaders import from_amazon_delivery, from_cardekho
from autovalue.schema import validate


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    for task in ("price", "eta"):
        assert main(["train", "--task", task, "--synthetic", "1200", "--out", str(out)]) == 0
    return out


def test_settings_defaults_and_env(monkeypatch):
    s = Settings.from_env()
    assert s.quantiles == (0.1, 0.5, 0.9) and s.seed == 42
    monkeypatch.setenv("AUTOVALUE_BAND_LOW", "0.05")
    monkeypatch.setenv("AUTOVALUE_SEED", "7")
    s = Settings.from_env()
    assert s.band_low == 0.05 and s.seed == 7


@pytest.mark.parametrize(
    "name,value", [("AUTOVALUE_BAND_LOW", "0.7"), ("AUTOVALUE_SEED", "x"), ("AUTOVALUE_BACKEND", "xgb")]
)
def test_bad_settings_raise(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_train_writes_model_metrics_and_card(trained):
    for task in ("price", "eta"):
        assert (trained / f"{task}.joblib").exists()
        metrics = json.loads((trained / f"{task}_metrics.json").read_text(encoding="utf-8"))
        assert metrics["task"] == task and "quantile_gbm" in metrics["metrics"]
        assert "SYNTHETIC" in (trained / f"{task}_model_card.md").read_text(encoding="utf-8")
    assert bundle.load(trained / "price.joblib").calibrated_


def test_estimate_approve_simulate_eta_commands(trained, capsys):
    car = json.dumps(EXAMPLE_CAR)
    assert main(["estimate", "--model", str(trained / "price.joblib"), "--input", car]) == 0
    band = json.loads(capsys.readouterr().out)[0]
    assert band["low"] <= band["mid"] <= band["high"]
    assert main(["approve", "--model", str(trained / "price.joblib"), "--input", car, "--price", str(band["mid"] * 3)]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "review_high"
    assert main(["simulate", "--model", str(trained / "price.joblib"), "--input", car, "--episodes", "200"]) == 0
    assert 0 <= json.loads(capsys.readouterr().out)["deal_rate"] <= 1
    assert main(["estimate", "--model", str(trained / "eta.joblib"), "--input", json.dumps(EXAMPLE_ORDER)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["mid"] > 0


def test_generate_and_validate_commands(tmp_path, capsys):
    assert main(["generate", "--out", str(tmp_path), "--listings", "200", "--deliveries", "200"]) == 0
    assert main(["validate", "--table", "deliveries", "--csv", str(tmp_path / "deliveries.csv")]) == 0
    assert "deliveries: 200 rows in" in capsys.readouterr().out


def test_bad_input_gives_error_code(trained, capsys):
    bad = json.dumps({"brand": "honda"})
    assert main(["estimate", "--model", str(trained / "price.joblib"), "--input", bad]) == 1
    assert "error:" in capsys.readouterr().err


def test_cardekho_adapter():
    raw = pd.DataFrame(
        {
            "name": ["Maruti Swift Dzire VDI", "Honda City 1.5"],
            "year": [2014, 2017],
            "selling_price": [450000, 800000],
            "km_driven": [145500, 40000],
            "fuel": ["Diesel", "Petrol"],
            "seller_type": ["Individual", "Dealer"],
            "transmission": ["Manual", "Automatic"],
            "owner": ["First Owner", "Second Owner"],
        }
    )
    clean, _ = validate(from_cardekho(raw), "listings")
    assert list(clean["brand"]) == ["maruti", "honda"]
    assert list(clean["model"]) == ["swift", "city"]
    assert list(clean["owners"]) == [1.0, 2.0]


def test_amazon_delivery_adapter_handles_midnight_and_sign_errors():
    raw = pd.DataFrame(
        {
            "Order_ID": ["a1", "a2"],
            "Store_Latitude": [-12.91, 22.74],
            "Store_Longitude": [77.68, 75.89],
            "Drop_Latitude": [13.04, 22.76],
            "Drop_Longitude": [77.81, 75.91],
            "Order_Date": ["2022-03-19", "2022-03-25"],
            "Order_Time": ["23:50:00", "19:45:00"],
            "Pickup_Time": ["00:05:00", "19:50:00"],
            "Weather": ["Sunny", "Stormy"],
            "Traffic": ["High ", "Jam "],
            "Vehicle": ["motorcycle ", "scooter "],
            "Area": ["Urban ", "Metropolitian "],
            "Delivery_Time": [120, 165],
        }
    )
    clean, report = validate(from_amazon_delivery(raw), "deliveries")
    assert report.rows_out == 2
    assert clean.loc[0, "store_lat"] == pytest.approx(12.91)
    assert (clean.loc[0, "picked_at"] - clean.loc[0, "ordered_at"]).total_seconds() == 15 * 60
    assert list(clean["traffic"]) == ["high", "jam"]


def test_http_api(trained):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from autovalue.api import create_app

    client = TestClient(create_app(trained / "price.joblib", trained / "eta.joblib"))
    assert client.get("/health").json() == {"price_model": True, "eta_model": True}
    band = client.post("/price/estimate", json=EXAMPLE_CAR).json()
    assert band["low"] <= band["mid"] <= band["high"]
    out = client.post("/price/approve", json={"car": EXAMPLE_CAR, "listed_price": band["mid"]}).json()
    assert out["decision"] == "auto_approve"
    assert client.post("/eta", json=EXAMPLE_ORDER).status_code == 200
