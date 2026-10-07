"""Seeded buyer / seller negotiation that uses the price band of one actual car.

The seller starts at the asking price and moves toward the middle estimate each round,
but never goes below the low end of the band. Each buyer draws a private value from a
triangular distribution over the band and accepts the first offer at or below it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class NegotiationSummary:
    episodes: int
    deal_rate: float
    mean_deal_price: float | None
    mean_rounds_to_deal: float | None
    deal_price_vs_mid: float | None  # mean deal price / mid estimate
    low: float
    mid: float
    high: float
    ask: float

    def as_dict(self) -> dict:
        return asdict(self)


def simulate_negotiation(
    low: float,
    mid: float,
    high: float,
    ask: float,
    episodes: int = 1000,
    max_rounds: int = 10,
    concession: float = 0.2,
    seed: int = 42,
) -> NegotiationSummary:
    if not 0 < low <= mid <= high:
        raise ValueError("the band must satisfy 0 < low <= mid <= high")
    if ask <= 0 or episodes < 1 or max_rounds < 1 or not 0 < concession < 1:
        raise ValueError("ask > 0, episodes >= 1, max_rounds >= 1 and 0 < concession < 1 are necessary")
    rng = np.random.default_rng(seed)
    # one seller offer path is the same for all buyers: a deterministic concession schedule
    rounds = np.arange(max_rounds)
    if ask > mid:
        offers = np.maximum(low, mid + (ask - mid) * (1.0 - concession) ** rounds)
    else:  # an ask at or below the middle estimate does not move
        offers = np.full(max_rounds, float(ask))
    values = rng.triangular(low, mid, high, size=episodes) if high > low else np.full(episodes, mid)
    accepted = offers[None, :] <= values[:, None]
    deal = accepted.any(axis=1)
    first = np.where(deal, accepted.argmax(axis=1), -1)
    prices = np.where(deal, offers[np.clip(first, 0, None)], np.nan)
    has_deal = bool(deal.any())
    return NegotiationSummary(
        episodes=episodes,
        deal_rate=float(deal.mean()),
        mean_deal_price=float(np.nanmean(prices)) if has_deal else None,
        mean_rounds_to_deal=float(first[deal].mean() + 1) if has_deal else None,
        deal_price_vs_mid=float(np.nanmean(prices) / mid) if has_deal else None,
        low=float(low),
        mid=float(mid),
        high=float(high),
        ask=float(ask),
    )
