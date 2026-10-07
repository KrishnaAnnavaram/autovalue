"""Save and load a trained model with its report and a short model card."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from . import __version__
from .models import QuantileModel
from .pipeline import TrainResult, to_jsonable


def save(result: TrainResult, out_dir: str | Path, data_source: str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = result.task.name
    paths = {
        "model": out / f"{name}.joblib",
        "metrics": out / f"{name}_metrics.json",
        "card": out / f"{name}_model_card.md",
    }
    meta = {
        "autovalue_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": data_source,
    }
    joblib.dump({"model": result.model, "meta": meta}, paths["model"])
    paths["metrics"].write_text(json.dumps(to_jsonable({**meta, **result.report}), indent=2), encoding="utf-8")
    paths["card"].write_text(model_card(result, meta), encoding="utf-8")
    return paths


def load(path: str | Path) -> QuantileModel:
    payload = joblib.load(path)
    model = payload["model"] if isinstance(payload, dict) else payload
    if not isinstance(model, QuantileModel):
        raise TypeError(f"{path} does not contain an autovalue QuantileModel")
    return model


def model_card(result: TrainResult, meta: dict) -> str:
    r = result.report
    m = r["metrics"]["quantile_gbm"]
    b = r["band"]
    synthetic = "synthetic" in meta["data_source"]
    lines = [
        f"# Model card: autovalue {r['task']}",
        "",
        f"- Created (UTC): {meta['created_utc']}",
        f"- Data source: {meta['data_source']}" + (" (SYNTHETIC: these numbers do not describe a real market)" if synthetic else ""),
        f"- Rows: {r['rows']}, split method: {r['split']['method']}",
        f"- Model: quantile gradient boosting ({r['settings']['backend']}), quantiles {b['quantiles']}",
        f"- Test MAE: {m['mae']:,.2f}, MAPE: {m['mape']:.4f}, R2: {m['r2']:.4f}",
        f"- Band coverage on test: {b['calibrated']['coverage']:.3f} (nominal {b['nominal_coverage']})",
        "",
        "## Intended use",
        "",
        "Give a price band or a delivery-time band for a human reviewer. It is not an automatic final decision.",
        "",
        "## Limits",
        "",
        "- The model learns only the patterns of its training period. Retrain it when the market changes.",
        "- A brand or model with few training rows gets a wide band or a band from similar rows.",
    ]
    return "\n".join(lines) + "\n"
