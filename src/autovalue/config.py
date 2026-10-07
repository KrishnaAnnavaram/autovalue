"""Settings from environment variables.

Every setting has a default, so the offline demo runs with an empty environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """An environment variable has a value that the code cannot use."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Read ``KEY=VALUE`` lines from a local ``.env`` file. Existing variables win."""
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _path(name: str, default: str) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw or default)


@dataclass(frozen=True)
class Settings:
    """Run settings. Use :meth:`from_env` to read them from the environment."""

    seed: int = 42
    data_dir: Path = field(default_factory=lambda: Path("data"))
    model_dir: Path = field(default_factory=lambda: Path("models"))
    band_low: float = 0.10
    band_high: float = 0.90
    backend: str = "sklearn"
    max_iter: int = 300

    def __post_init__(self) -> None:
        if not 0.0 < self.band_low < 0.5 < self.band_high < 1.0:
            raise ConfigError(
                f"price band quantiles must satisfy 0 < low < 0.5 < high < 1, got {self.band_low}, {self.band_high}"
            )
        if self.backend not in {"sklearn", "lightgbm"}:
            raise ConfigError(f"AUTOVALUE_BACKEND must be 'sklearn' or 'lightgbm', got {self.backend!r}")
        if self.max_iter < 10:
            raise ConfigError("AUTOVALUE_MAX_ITER must be at least 10")

    @property
    def quantiles(self) -> tuple[float, float, float]:
        return (self.band_low, 0.5, self.band_high)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            seed=_int("AUTOVALUE_SEED", 42),
            data_dir=_path("AUTOVALUE_DATA_DIR", "data"),
            model_dir=_path("AUTOVALUE_MODEL_DIR", "models"),
            band_low=_float("AUTOVALUE_BAND_LOW", 0.10),
            band_high=_float("AUTOVALUE_BAND_HIGH", 0.90),
            backend=os.environ.get("AUTOVALUE_BACKEND", "").strip() or "sklearn",
            max_iter=_int("AUTOVALUE_MAX_ITER", 300),
        )
