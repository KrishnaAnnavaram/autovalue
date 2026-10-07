import os

# One OpenMP thread: the tests stay fast and identical on a shared CI runner.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402

from autovalue.config import Settings  # noqa: E402
from autovalue.pipeline import train_and_evaluate  # noqa: E402
from autovalue.schema import validate  # noqa: E402
from autovalue.synthetic import make_deliveries, make_listings  # noqa: E402
from autovalue.tasks import ETA, PRICE  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return Settings(max_iter=120)


@pytest.fixture(scope="session")
def listings():
    clean, _ = validate(make_listings(n=2000, seed=11), "listings")
    return clean


@pytest.fixture(scope="session")
def deliveries():
    clean, _ = validate(make_deliveries(n=2000, seed=12), "deliveries")
    return clean


@pytest.fixture(scope="session")
def price_result(listings, settings):
    return train_and_evaluate(listings, PRICE, settings, n_boot=200)


@pytest.fixture(scope="session")
def eta_result(deliveries, settings):
    return train_and_evaluate(deliveries, ETA, settings, n_boot=200)
