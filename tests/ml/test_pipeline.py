from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bybit_bot.ml.pipeline import EnsemblePipeline


def _generate_price_series(periods: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq)
    prices = np.cumsum(np.random.normal(0, 1, size=periods)) + 100
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.uniform(0, 1, periods),
            "low": prices - np.random.uniform(0, 1, periods),
            "close": prices + np.random.uniform(-0.5, 0.5, periods),
            "volume": np.random.uniform(1000, 2000, periods),
        },
        index=idx,
    )


@pytest.mark.slow
def test_pipeline_fit_and_predict(tmp_path):
    pipeline = EnsemblePipeline(test_size=0.3, random_state=42)

    price_data = {
        "m15": _generate_price_series(200, "15min"),
        "h1": _generate_price_series(200, "1h"),
    }
    labels = pd.Series(
        np.random.randint(0, 2, size=200),
        index=price_data["m15"].index,
    )

    artifacts = pipeline.fit(price_data, labels)
    assert artifacts.feature_columns
    assert artifacts.meta["learning_rule"] == "Disco57"

    predictions = pipeline.predict(price_data)
    assert predictions.shape[0] > 0

    model_dir = tmp_path / "model"
    pipeline.save(model_dir)
    loaded = EnsemblePipeline.load(model_dir)
    new_predictions = loaded.predict(price_data)
    assert new_predictions.shape == predictions.shape

