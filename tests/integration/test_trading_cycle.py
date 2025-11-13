from __future__ import annotations

import numpy as np
import pandas as pd

from bybit_bot.core.adaptation import AdaptationConfig
from bybit_bot.core.coordinator import TradingCoordinator


def _make_price_series(periods: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    prices = np.linspace(100, 110, periods)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices + 0.5,
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )


class StubPipeline:
    def __init__(self) -> None:
        self.weights = {
            "random_forest": 1.0,
            "lightgbm": 1.0,
            "svm": 1.0,
            "neural_net": 1.0,
        }

    def predict_with_components(self, price_data):
        ensemble = np.array([[0.2, 0.8]])
        component_probs = {name: np.array([[0.2, 0.8]]) for name in self.weights}
        return ensemble, component_probs

    def get_weights(self):
        return dict(self.weights)

    def set_weights(self, new_weights):
        for name, value in new_weights.items():
            self.weights[name] = float(value)


class StubClient:
    def __init__(self) -> None:
        self.orders = []

    def get_wallet_balance(self, account_type="UNIFIED"):
        return {"list": [{"totalEquity": 101.0, "availableBalance": 100.0}]}

    def get_positions(self, category="linear", symbol=None):
        return {"list": []}

    def get_history_orders(self, category="linear", limit=50, symbol=None):
        return {"list": []}

    def place_order(self, order):
        self.orders.append(order)
        return {"orderId": "ORDER123"}

    def set_leverage(self, symbol, buy_leverage, sell_leverage, category="linear"):
        return {"result": "ok"}


def test_trading_cycle_with_adaptation():
    pipeline = StubPipeline()
    client = StubClient()
    coordinator = TradingCoordinator(
        client=client,
        pipeline=pipeline,
        adaptation_config=AdaptationConfig(min_trades=1, weight_lr=0.2, threshold_lr=0.05),
    )
    coordinator.previous_total_equity = 100.0

    price_data = {
        "m15": _make_price_series(60, "15min"),
        "m30": _make_price_series(60, "30min"),
        "h1": _make_price_series(60, "1h"),
        "h4": _make_price_series(60, "4h"),
        "d1": _make_price_series(60, "1D"),
    }

    result = coordinator.run_cycle(price_data)
    assert result is not None
    assert result["orderId"] == "ORDER123"

    weights = pipeline.get_weights()
    assert weights["random_forest"] > 1.0
    assert coordinator.risk_manager.config.base_position_size_usd > 1.0

