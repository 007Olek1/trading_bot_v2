from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

from .risk_settings import (
    FIXED_LEVERAGE,
    FIXED_MARGIN_USD,
    MAX_OPEN_POSITIONS,
    PARTIAL_TAKES,
)

load_dotenv()


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _parse_fraction_tuple(value: str) -> Tuple[float, float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except (ValueError, AttributeError):
        parts = []
    while len(parts) < 3:
        parts.append(0.0)
    parts = parts[:3]
    total = sum(parts)
    if total <= 0:
        return (0.4, 0.3, 0.3)
    normalized = tuple(p / total for p in parts)
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class BybitConfig:
    api_key: str
    api_secret: str
    testnet: bool


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    chat_id: str

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)


@dataclass(frozen=True)
class PathsConfig:
    trade_db_path: Path
    disco_model_path: Path


@dataclass(frozen=True)
class TradingConfig:
    margin_usd: float
    leverage: int
    max_positions: int
    scan_interval_minutes: Tuple[int, int]
    scan_fast_timeframe: str
    scan_slow_timeframe: str
    risk_fraction: float
    min_notional_usd: float
    min_qty: float
    partial_takes: Tuple[float, float, float]
    trail_activate_at_r: float
    trail_step_atr_mult: float

    @property
    def exposure_usd(self) -> float:
        return self.margin_usd * self.leverage


@dataclass(frozen=True)
class Config:
    bybit: BybitConfig
    telegram: TelegramConfig
    paths: PathsConfig
    trading: TradingConfig


def load_config() -> Config:
    """Собрать конфигурацию из переменных окружения."""
    bybit = BybitConfig(
        api_key=os.getenv("BYBIT_API_KEY", ""),
        api_secret=os.getenv("BYBIT_API_SECRET", ""),
        testnet=_env_bool("BYBIT_TESTNET", True),
    )

    telegram = TelegramConfig(
        token=os.getenv("TELEGRAM_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )

    trade_db = Path(os.getenv("TRADE_DB_PATH", "/opt/bot/data/trade_history.db"))
    disco_path = Path(os.getenv("DISCO57_MODEL_PATH", "./data/disco57_model.pkl"))
    paths = PathsConfig(
        trade_db_path=trade_db,
        disco_model_path=disco_path,
    )

    trading = TradingConfig(
        margin_usd=FIXED_MARGIN_USD,
        leverage=FIXED_LEVERAGE,
        max_positions=MAX_OPEN_POSITIONS,
        scan_interval_minutes=(
            _env_int("SCAN_INTERVAL_MIN_MIN", 5),
            _env_int("SCAN_INTERVAL_MIN_MAX", 15),
        ),
        scan_fast_timeframe=os.getenv("SCAN_FAST_TIMEFRAME", "5"),
        scan_slow_timeframe=os.getenv("SCAN_SLOW_TIMEFRAME", "15"),
        risk_fraction=0.0,
        min_notional_usd=_env_float("MIN_NOTIONAL_USD", 5.0),
        min_qty=_env_float("MIN_QTY", 0.001),
        partial_takes=PARTIAL_TAKES,
        trail_activate_at_r=_env_float("TRAIL_ACTIVATE_AT_R", 0.0),
        trail_step_atr_mult=_env_float("TRAIL_STEP_ATR_MULT", 0.0),
    )

    return Config(
        bybit=bybit,
        telegram=telegram,
        paths=paths,
        trading=trading,
    )


__all__ = [
    "Config",
    "BybitConfig",
    "TelegramConfig",
    "PathsConfig",
    "TradingConfig",
    "load_config",
]
