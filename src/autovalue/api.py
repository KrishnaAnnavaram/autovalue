"""Optional HTTP API (extra ``api``): ``/price/estimate``, ``/price/approve`` and ``/eta``.

FastAPI is imported only here, so the core package works without it.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import bundle
from .approval import decide
from .models import QuantileModel
from .schema import SchemaError, coerce_input


class Car(BaseModel):
    listed_at: str
    brand: str
    model: str
    year: int
    odometer_km: float = Field(ge=0)
    fuel: str
    transmission: str
    owners: int | None = None
    accidents: bool | None = None
    engine_l: float | None = None
    body_type: str | None = None
    city: str | None = None


class ApproveRequest(BaseModel):
    car: Car
    listed_price: float = Field(gt=0)


class Order(BaseModel):
    store_lat: float
    store_lon: float
    drop_lat: float
    drop_lon: float
    ordered_at: str
    picked_at: str
    weather: str | None = None
    traffic: str
    area: str | None = None
    vehicle: str | None = None


def _band(model: QuantileModel | None, record: dict) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model file not loaded")
    try:
        frame = coerce_input([record], model.task.table, model.task.target)
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row = model.predict_band(frame).iloc[0]
    return {"low": round(float(row["low"]), 2), "mid": round(float(row["mid"]), 2), "high": round(float(row["high"]), 2)}


def create_app(price_model: str | Path | QuantileModel | None, eta_model: str | Path | QuantileModel | None) -> FastAPI:
    def _load(value):
        if value is None or isinstance(value, QuantileModel):
            return value
        return bundle.load(value) if Path(value).exists() else None

    models = {"price": _load(price_model), "eta": _load(eta_model)}
    app = FastAPI(title="autovalue", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"price_model": models["price"] is not None, "eta_model": models["eta"] is not None}

    @app.post("/price/estimate")
    def price_estimate(car: Car) -> dict:
        return _band(models["price"], car.model_dump())

    @app.post("/price/approve")
    def price_approve(req: ApproveRequest) -> dict:
        band = _band(models["price"], req.car.model_dump())
        return decide(req.listed_price, band["low"], band["mid"], band["high"]).as_dict()

    @app.post("/eta")
    def eta(order: Order) -> dict:
        return _band(models["eta"], order.model_dump())

    return app
