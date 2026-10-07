"""Command line: ``autovalue <command>``. Run ``autovalue --help`` for the list."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import bundle
from .approval import decide
from .config import ConfigError, Settings, load_dotenv
from .loaders import ADAPTERS
from .pipeline import format_report, to_jsonable, train_and_evaluate
from .schema import SchemaError, coerce_input, validate
from .simulate import simulate_negotiation
from .synthetic import make_deliveries, make_listings
from .tasks import get_task

EXAMPLE_CAR = {
    "listed_at": "2024-11-15",
    "brand": "Hyundai",
    "model": "Creta",
    "year": 2019,
    "odometer_km": 48000,
    "fuel": "Diesel",
    "transmission": "Automatic",
    "owners": 1,
    "accidents": False,
    "engine_l": 1.5,
    "body_type": "SUV",
    "city": "Pune",
}
EXAMPLE_ORDER = {
    "store_lat": 12.9716,
    "store_lon": 77.5946,
    "drop_lat": 13.0358,
    "drop_lon": 77.5970,
    "ordered_at": "2023-12-05 18:30:00",
    "picked_at": "2023-12-05 18:42:00",
    "weather": "Stormy",
    "traffic": "Jam ",
    "area": "Metropolitian ",
    "vehicle": "motorcycle",
}


def _read_input(text: str) -> list[dict]:
    path = Path(text)
    raw = path.read_text(encoding="utf-8") if path.suffix == ".json" and path.exists() else text
    data = json.loads(raw)
    return data if isinstance(data, list) else [data]


def _load_frame(args, task) -> tuple[pd.DataFrame, str]:
    if args.csv:
        frame = pd.read_csv(args.csv)
        source = str(args.csv)
        if args.adapter:
            table, adapter = ADAPTERS[args.adapter]
            if table != task.table:
                raise SystemExit(f"error: adapter {args.adapter} gives {table}, but task {task.name} needs {task.table}")
            frame = adapter(frame)
            source += f" via {args.adapter}"
        return frame, source
    maker = make_listings if task.name == "price" else make_deliveries
    return maker(n=args.synthetic, seed=args.seed), f"synthetic ({args.synthetic} rows, seed {args.seed})"


def cmd_generate(args, settings: Settings) -> int:
    out = Path(args.out or settings.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    listings = make_listings(args.listings, seed=args.seed)
    deliveries = make_deliveries(args.deliveries, seed=args.seed + 1)
    listings.to_csv(out / "listings.csv", index=False)
    deliveries.to_csv(out / "deliveries.csv", index=False)
    print(f"wrote {out / 'listings.csv'} ({len(listings)} rows) and {out / 'deliveries.csv'} ({len(deliveries)} rows)")
    return 0


def cmd_validate(args, settings: Settings) -> int:
    _, report = validate(pd.read_csv(args.csv), args.table)
    print(report.summary())
    return 0


def cmd_train(args, settings: Settings) -> int:
    task = get_task(args.task)
    frame, source = _load_frame(args, task)
    clean, vreport = validate(frame, task.table)
    print(vreport.summary())
    result = train_and_evaluate(clean, task, settings)
    paths = bundle.save(result, args.out or settings.model_dir, data_source=source)
    if args.json:
        print(json.dumps(to_jsonable(result.report), indent=2))
    else:
        print(format_report(result.report))
    print(f"saved {paths['model']}, {paths['metrics']}, {paths['card']}")
    return 0


def _band_rows(model, records) -> pd.DataFrame:
    task = model.task
    frame = coerce_input(records, task.table, task.target)
    return model.predict_band(frame)


def cmd_estimate(args, settings: Settings) -> int:
    model = bundle.load(args.model)
    band = _band_rows(model, _read_input(args.input))
    print(json.dumps(to_jsonable(band.round(2).to_dict(orient="records")), indent=2))
    return 0


def cmd_approve(args, settings: Settings) -> int:
    model = bundle.load(args.model)
    if model.task.name != "price":
        raise SystemExit("error: approve needs a price model")
    band = _band_rows(model, _read_input(args.input)).iloc[0]
    result = decide(args.price, band["low"], band["mid"], band["high"])
    print(json.dumps(to_jsonable(result.as_dict()), indent=2))
    return 0


def cmd_simulate(args, settings: Settings) -> int:
    model = bundle.load(args.model)
    band = _band_rows(model, _read_input(args.input)).iloc[0]
    ask = args.ask if args.ask else float(band["high"])
    summary = simulate_negotiation(band["low"], band["mid"], band["high"], ask, episodes=args.episodes, seed=args.seed)
    print(json.dumps(to_jsonable(summary.as_dict()), indent=2))
    return 0


def cmd_demo(args, settings: Settings) -> int:
    print("autovalue offline demo: synthetic data, no download, no key, no network\n")
    models = {}
    for name, maker, seed in (("price", make_listings, args.seed), ("eta", make_deliveries, args.seed + 1)):
        task = get_task(name)
        clean, vreport = validate(maker(n=args.rows, seed=seed), task.table)
        print(vreport.summary())
        result = train_and_evaluate(clean, task, settings, n_boot=300)
        print(format_report(result.report) + "\n")
        models[name] = result.model
    price_band = _band_rows(models["price"], [EXAMPLE_CAR]).iloc[0]
    print(f"example car {EXAMPLE_CAR['brand']} {EXAMPLE_CAR['model']} {EXAMPLE_CAR['year']}: "
          f"band {price_band['low']:,.0f} / {price_band['mid']:,.0f} / {price_band['high']:,.0f}")
    for listed in (0.5 * price_band["mid"], price_band["mid"], 1.8 * price_band["mid"]):
        r = decide(listed, price_band["low"], price_band["mid"], price_band["high"])
        print(f"  listed {listed:>12,.0f} -> {r.decision.value}")
    sim = simulate_negotiation(price_band["low"], price_band["mid"], price_band["high"], ask=price_band["high"], seed=settings.seed)
    print(f"  negotiation: deal rate {sim.deal_rate:.3f}, mean deal price / mid {sim.deal_price_vs_mid:.3f}")
    eta_band = _band_rows(models["eta"], [EXAMPLE_ORDER]).iloc[0]
    print(f"example order (jam, stormy, 7.1 km): {eta_band['low']:.0f} / {eta_band['mid']:.0f} / {eta_band['high']:.0f} minutes")
    return 0


def cmd_serve(args, settings: Settings) -> int:
    try:
        import uvicorn

        from .api import create_app
    except ImportError:
        print("error: install the api extra: pip install -e \".[api]\"", file=sys.stderr)
        return 2
    uvicorn.run(create_app(args.price_model, args.eta_model), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autovalue", description="Used-car price bands and delivery-time bands.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write synthetic listings.csv and deliveries.csv")
    g.add_argument("--out", help="output folder (default AUTOVALUE_DATA_DIR, then data)")
    g.add_argument("--listings", type=int, default=4000)
    g.add_argument("--deliveries", type=int, default=4000)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="validate a CSV against a table schema")
    v.add_argument("--table", choices=["listings", "deliveries"], required=True)
    v.add_argument("--csv", required=True)
    v.set_defaults(func=cmd_validate)

    t = sub.add_parser("train", help="train, calibrate and evaluate one task, then save the model")
    t.add_argument("--task", choices=["price", "eta"], required=True)
    t.add_argument("--csv", help="CSV in the project schema (or a source table with --adapter)")
    t.add_argument("--adapter", choices=sorted(ADAPTERS), help="map a public dataset to the project schema")
    t.add_argument("--synthetic", type=int, default=4000, help="rows of synthetic data if no --csv")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", help="output folder (default AUTOVALUE_MODEL_DIR)")
    t.add_argument("--json", action="store_true", help="print the full report as JSON")
    t.set_defaults(func=cmd_train)

    for name, func, text in (
        ("estimate", cmd_estimate, "print the band for each input row (car or order)"),
        ("approve", cmd_approve, "apply the approval rule to one car and a listed price"),
        ("simulate", cmd_simulate, "run the seeded negotiation for one car"),
    ):
        c = sub.add_parser(name, help=text)
        c.add_argument("--model", required=True, help="path to a .joblib file from 'train'")
        c.add_argument("--input", required=True, help="JSON object or list, inline or a .json file")
        if name == "approve":
            c.add_argument("--price", type=float, required=True)
        if name == "simulate":
            c.add_argument("--ask", type=float, default=0.0, help="asking price (default: the high end of the band)")
            c.add_argument("--episodes", type=int, default=1000)
            c.add_argument("--seed", type=int, default=42)
        c.set_defaults(func=func)

    d = sub.add_parser("demo", help="offline end-to-end demo on synthetic data")
    d.add_argument("--rows", type=int, default=3000)
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="start the HTTP API (extra 'api')")
    s.add_argument("--price-model", default="models/price.joblib")
    s.add_argument("--eta-model", default="models/eta.joblib")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_dotenv()
        settings = Settings.from_env()
        return args.func(args, settings)
    except (ConfigError, SchemaError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
