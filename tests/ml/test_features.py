from __future__ import annotations

import numpy as np
import pandas as pd

from bybit_bot.ml.features import FeatureConfig, compute_indicators


def test_compute_indicators_generates_columns():
    idx = pd.date_range("2024-01-01", periods=100, freq="15min")
    data = pd.DataFrame(
        {
            "open": np.linspace(100, 110, len(idx)),
            "high": np.linspace(101, 111, len(idx)),
            "low": np.linspace(99, 109, len(idx)),
            "close": np.linspace(100, 120, len(idx)),
            "volume": np.random.uniform(1000, 2000, len(idx)),
        },
        index=idx,
    )

    config = FeatureConfig(rsi_periods=(14,), ema_periods=(10,))
    features = compute_indicators(data, config=config, prefix="m15_")
    assert not features.empty
    assert "m15_rsi_14" in features.columns
    assert "m15_ema_10" in features.columns

