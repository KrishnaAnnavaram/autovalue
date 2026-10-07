"""Time-based splits."""
import pandas as pd
import pytest

from autovalue.split import time_split


def test_time_split_keeps_order_and_has_no_overlap(listings):
    split = time_split(listings, "listed_at")
    assert split.method == "time"
    assert split.train["listed_at"].max() < split.val["listed_at"].min()
    assert split.val["listed_at"].max() < split.test["listed_at"].min()
    ids = [set(p["listing_id"]) for p in (split.train, split.val, split.test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(split.sizes().values()) == len(listings)
    assert 0.6 < len(split.train) / len(listings) < 0.8


def test_single_timestamp_falls_back_to_seeded_random_split():
    df = pd.DataFrame({"t": ["2020-01-01"] * 100, "x": range(100)})
    a = time_split(df, "t", seed=1)
    b = time_split(df, "t", seed=1)
    assert a.method == "random" and a.notes
    pd.testing.assert_frame_equal(a.test, b.test)
    assert len(a.test) == 15 and len(a.val) == 15


def test_bad_fractions_raise():
    with pytest.raises(ValueError):
        time_split(pd.DataFrame({"t": ["2020-01-01"]}), "t", val_frac=0.6)
