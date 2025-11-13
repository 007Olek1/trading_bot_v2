"""Runtime adaptation of ensemble weights and signal thresholds."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable

from bybit_bot.core.risk import RiskManager
from bybit_bot.core.signals import SignalGenerator
from bybit_bot.ml.pipeline import EnsemblePipeline

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AdaptationConfig:
    min_trades: int = 3
    weight_lr: float = 0.1
    threshold_lr: float = 0.02
    threshold_bounds: tuple[float, float] = (0.55, 0.75)
    size_scale_bounds: tuple[float, float] = (0.8, 1.2)


class AdaptationManager:
    """Adapts model weights, signal thresholds, and risk parameters based on realised profit."""

    def __init__(
        self,
        pipeline: EnsemblePipeline,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        config: AdaptationConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.config = config or AdaptationConfig()
        self._buffer: Deque[tuple[float, str, Dict[str, float]]] = deque(maxlen=50)

    def record_trade(self, profit: float, signal: str, component_support: Dict[str, float]) -> None:
        """Store trade outcome; profit is realised change in equity."""
        if signal not in {"BUY", "SELL"}:
            return
        self._buffer.append((profit, signal, component_support))
        if len(self._buffer) >= self.config.min_trades:
            self._adapt()

    def _adapt(self) -> None:
        profits = [p for p, _, _ in self._buffer]
        if not profits:
            return
        avg_profit = sum(profits) / len(profits)
        self._adjust_threshold(avg_profit)
        self._adjust_weights(avg_profit)
        self._adjust_position_size(avg_profit)
        logger.info(
            "Adaptation applied | avg_profit=%.4f | threshold=%.3f | weights=%s | base_size=%.2f",
            avg_profit,
            self.signal_generator.config.probability_threshold,
            self.pipeline.get_weights(),
            self.risk_manager.config.base_position_size_usd,
        )
        self._buffer.clear()

    def _adjust_threshold(self, avg_profit: float) -> None:
        current = self.signal_generator.config.probability_threshold
        delta = -self.config.threshold_lr if avg_profit > 0 else self.config.threshold_lr
        new_threshold = max(self.config.threshold_bounds[0], min(self.config.threshold_bounds[1], current + delta))
        self.signal_generator.config.probability_threshold = new_threshold

    def _adjust_weights(self, avg_profit: float) -> None:
        weights = self.pipeline.get_weights()
        for profit, signal, support in self._buffer:
            direction = 1 if profit > 0 else -1
            for name, contribution in support.items():
                confidence = contribution if signal == "BUY" else 1 - contribution
                if confidence >= 0.5:
                    weights[name] = max(0.1, weights[name] * (1 + self.config.weight_lr * direction))
        self.pipeline.set_weights(weights)

    def _adjust_position_size(self, avg_profit: float) -> None:
        scale = self.config.size_scale_bounds[1] if avg_profit > 0 else self.config.size_scale_bounds[0]
        self.risk_manager.adjust_position_size(scale)

