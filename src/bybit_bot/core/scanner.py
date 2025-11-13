"""Market scanner that ranks symbols by ensemble confidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from bybit_bot.data.provider import MarketDataProvider
from bybit_bot.ml.pipeline import EnsemblePipeline

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScannerConfig:
    watchlist: Sequence[str] | None = field(default=None)
    min_confidence: float = 0.65  # Оптимизировано: 65% для фокуса на лучших возможностях
    top_n: int = 8

    def resolved_watchlist(self) -> Sequence[str]:
        if self.watchlist:
            return self.watchlist
        return (
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "AVAX/USDT",
            "LTC/USDT",
            "LINK/USDT",
            "DOT/USDT",
            "MATIC/USDT",
            "APT/USDT",
            "ARB/USDT",
            "OP/USDT",
        )


class MarketScanner:
    def __init__(
        self,
        data_provider: MarketDataProvider,
        pipeline: EnsemblePipeline,
        config: ScannerConfig | None = None,
    ) -> None:
        self.data_provider = data_provider
        self.pipeline = pipeline
        self.config = config or ScannerConfig()

    def rank(self, top_n: int | None = None) -> list[dict]:
        limit = top_n or self.config.top_n
        passed: list[dict] = []
        candidates: list[dict] = []
        for symbol in self.config.resolved_watchlist():
            try:
                price_data = self.data_provider.fetch(symbol=symbol)
                ensemble_probs, _ = self.pipeline.predict_with_components(price_data)
                buy_prob = float(ensemble_probs[-1][1])
                sell_prob = float(ensemble_probs[-1][0])
                confidence = max(buy_prob, sell_prob)
                direction = "LONG" if buy_prob >= sell_prob else "SHORT"
                candidate = {
                    "symbol": symbol.replace("/", ""),
                    "direction": direction,
                    "confidence": confidence,
                    "above_threshold": confidence >= self.config.min_confidence,
                }
                candidates.append(candidate)
                if candidate["above_threshold"]:
                    passed.append(candidate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to rank symbol %s: %s", symbol, exc)
        target = passed if passed else candidates
        target.sort(key=lambda item: item["confidence"], reverse=True)
        return target[:limit]

