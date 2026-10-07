"""Reference problems 1 and 2: the labels come from a documented, seeded generator."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from autovalue.synthetic import PRICE_NOISE_SIGMA, make_deliveries, make_listings

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_same_seed_gives_identical_tables():
    pd.testing.assert_frame_equal(make_listings(300, seed=5), make_listings(300, seed=5))
    pd.testing.assert_frame_equal(make_deliveries(300, seed=5), make_deliveries(300, seed=5))


def test_different_seed_gives_different_labels():
    assert not make_listings(300, seed=1)["price"].equals(make_listings(300, seed=2)["price"])


def _label_digest(hashseed: str) -> str:
    code = (
        "import hashlib; from autovalue.synthetic import make_listings, make_deliveries;"
        "a = make_listings(200, seed=3)['price'].to_numpy().tobytes();"
        "b = make_deliveries(200, seed=3)['delivery_minutes'].to_numpy().tobytes();"
        "print(hashlib.sha256(a + b).hexdigest())"
    )
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=SRC)
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_labels_do_not_depend_on_python_hash_seed():
    assert _label_digest("1") == _label_digest("2")


def test_delivery_table_has_no_car_columns():
    deliveries = make_deliveries(100, seed=1)
    for car_column in ("brand", "model", "year", "fuel", "transmission", "price"):
        assert car_column not in deliveries.columns


def test_price_noise_matches_documented_sigma():
    df = make_listings(3000, seed=9, dirty=False)
    log_ratio = np.log(df["price"] / df["oracle_price"])
    assert abs(log_ratio.std() - PRICE_NOISE_SIGMA) < 0.01
