"""Utility script to train the ensemble pipeline on historical OHLCV data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bybit_bot.data.provider import DataConfig, MarketDataProvider
from bybit_bot.ml.pipeline import EnsemblePipeline

MODEL_DIR = Path("models/ensemble")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def generate_labels(price_data: dict[str, pd.DataFrame]) -> pd.Series:
    """Binary labels based on next candle direction on the primary timeframe."""
    base = price_data["m15"]["close"]
    future_returns = base.pct_change().shift(-1)
    labels = (future_returns > 0).astype(int)
    return labels


def train() -> None:
    logger = logging.getLogger("train_ensemble")
    _configure_logging()

    logger.info("Fetching market data for training")
    provider = MarketDataProvider(DataConfig())
    price_data = provider.fetch()

    logger.info("Computing labels")
    labels = generate_labels(price_data)

    pipeline = EnsemblePipeline()
    logger.info("Fitting ensemble pipeline")
    pipeline.fit(price_data, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.save(MODEL_DIR)
    logger.info("Model saved to %s", MODEL_DIR / "ensemble.joblib")


if __name__ == "__main__":
    train()

