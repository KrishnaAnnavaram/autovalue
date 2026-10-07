"""The price-approval rule and its evaluation as a classifier.

Rule: if the listed price is inside the calibrated band [low, high], approve it
automatically. If it is below the band, send it to manual review as ``review_low``.
If it is above the band, send it to manual review as ``review_high``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Decision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    REVIEW_LOW = "review_low"
    REVIEW_HIGH = "review_high"


@dataclass
class ApprovalResult:
    decision: Decision
    listed_price: float
    low: float
    mid: float
    high: float
    ratio_to_mid: float
    reason: str

    def as_dict(self) -> dict:
        out = asdict(self)
        out["decision"] = self.decision.value
        return out


def decide(listed_price: float, low: float, mid: float, high: float) -> ApprovalResult:
    if listed_price <= 0:
        raise ValueError("listed_price must be positive")
    if not low <= mid <= high:
        raise ValueError("the band must satisfy low <= mid <= high")
    ratio = listed_price / mid
    if listed_price < low:
        decision, reason = Decision.REVIEW_LOW, f"price is {low - listed_price:,.0f} below the band"
    elif listed_price > high:
        decision, reason = Decision.REVIEW_HIGH, f"price is {listed_price - high:,.0f} above the band"
    else:
        decision, reason = Decision.AUTO_APPROVE, "price is inside the band"
    return ApprovalResult(decision, float(listed_price), float(low), float(mid), float(high), float(ratio), reason)


def decide_many(listed: np.ndarray, band: pd.DataFrame) -> np.ndarray:
    listed = np.asarray(listed, dtype=float)
    out = np.full(listed.shape, Decision.AUTO_APPROVE.value, dtype=object)
    out[listed < band["low"].to_numpy()] = Decision.REVIEW_LOW.value
    out[listed > band["high"].to_numpy()] = Decision.REVIEW_HIGH.value
    return out


def make_mispriced(
    df: pd.DataFrame, price_col: str = "price", frac: float = 0.3, low=(0.45, 0.7), high=(1.45, 2.2), seed: int = 42
) -> pd.DataFrame:
    """Copy held-out listings and change the price of a fraction ``frac`` of them.

    Changed rows get ``is_mispriced = True`` and a listed price that is far below or far
    above the actual sale price. Other rows keep the actual price as the listed price.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    bad = rng.random(n) < frac
    under = rng.random(n) < 0.5
    factor = np.where(under, rng.uniform(*low, size=n), rng.uniform(*high, size=n))
    out["listed_price"] = np.where(bad, out[price_col].to_numpy(float) * factor, out[price_col].to_numpy(float))
    out["is_mispriced"] = bad
    return out


def evaluate_rule(listed: np.ndarray, is_mispriced: np.ndarray, band: pd.DataFrame) -> dict:
    """Score the rule: a flag (any review decision) is a positive prediction of 'mispriced'."""
    decisions = decide_many(listed, band)
    flagged = decisions != Decision.AUTO_APPROVE.value
    truth = np.asarray(is_mispriced, dtype=bool)
    tp = int(np.sum(flagged & truth))
    fp = int(np.sum(flagged & ~truth))
    fn = int(np.sum(~flagged & truth))
    tn = int(np.sum(~flagged & ~truth))
    return {
        "n": int(truth.size),
        "auto_approve_rate": float(np.mean(~flagged)),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "false_review_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
