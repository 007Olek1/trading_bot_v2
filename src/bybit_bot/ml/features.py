"""Feature engineering utilities for multi-timeframe trading data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange


@dataclass(slots=True, frozen=True)
class FeatureConfig:
    """Configuration for indicator generation."""

    rsi_periods: Iterable[int] = (14, 21)
    ema_periods: Iterable[int] = (9, 21, 55)
    stoch_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")


def compute_indicators(
    df: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Return dataframe with engineered indicator features.

    Parameters
    ----------
    df:
        Dataframe with OHLCV columns: open, high, low, close, volume.
    config:
        Feature configuration; uses defaults if not provided.
    prefix:
        Optional prefix to prepend feature column names (for MTF aggregation).
    """

    if df.empty:
        raise ValueError("Input dataframe for feature computation is empty")

    _require_columns(df, ["open", "high", "low", "close", "volume"])
    cfg = config or FeatureConfig()
    features: List[pd.Series] = []

    # RSI indicators
    for period in cfg.rsi_periods:
        indicator = RSIIndicator(close=df["close"], window=period)
        features.append(indicator.rsi().rename(f"{prefix}rsi_{period}"))

    # Stochastic RSI
    stoch_rsi = StochRSIIndicator(close=df["close"], window=cfg.stoch_period)
    features.extend(
        [
            stoch_rsi.stochrsi().rename(f"{prefix}stoch_rsi"),
            stoch_rsi.stochrsi_k().rename(f"{prefix}stoch_rsi_k"),
            stoch_rsi.stochrsi_d().rename(f"{prefix}stoch_rsi_d"),
        ]
    )

    # EMA indicators
    for period in cfg.ema_periods:
        ema = EMAIndicator(close=df["close"], window=period)
        features.append(ema.ema_indicator().rename(f"{prefix}ema_{period}"))

    # MACD
    macd = MACD(
        close=df["close"],
        window_slow=cfg.macd_slow,
        window_fast=cfg.macd_fast,
        window_sign=cfg.macd_signal,
    )
    features.extend(
        [
            macd.macd().rename(f"{prefix}macd"),
            macd.macd_signal().rename(f"{prefix}macd_signal"),
            macd.macd_diff().rename(f"{prefix}macd_hist"),
        ]
    )

    # ATR / ADX
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=cfg.atr_period)
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=cfg.adx_period)
    features.extend(
        [
            atr.average_true_range().rename(f"{prefix}atr"),
            adx.adx().rename(f"{prefix}adx"),
            adx.adx_pos().rename(f"{prefix}adx_pos"),
            adx.adx_neg().rename(f"{prefix}adx_neg"),
        ]
    )

    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=cfg.bb_period, window_dev=cfg.bb_std)
    features.extend(
        [
            bb.bollinger_hband().rename(f"{prefix}bb_high"),
            bb.bollinger_lband().rename(f"{prefix}bb_low"),
            bb.bollinger_mavg().rename(f"{prefix}bb_mid"),
            bb.bollinger_hband_indicator().rename(f"{prefix}bb_touch_high"),
            bb.bollinger_lband_indicator().rename(f"{prefix}bb_touch_low"),
            bb.bollinger_wband().rename(f"{prefix}bb_width"),
        ]
    )

    # Volume-based
    features.append((df["volume"].pct_change().fillna(0).rename(f"{prefix}volume_pct_change")))

    feature_df = pd.concat(features, axis=1)
    feature_df = feature_df.replace([pd.NA, pd.NaT], 0).fillna(0)
    return feature_df

