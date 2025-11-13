"""Signal generation combining model probabilities and multi-timeframe filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SignalConfig:
    probability_threshold: float = 0.7  # Оптимизировано: 70% для более точных сигналов
    alignment_required: int = 3  # number of timeframes that must agree
    momentum_window: int = 5


class SignalGenerator:
    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()

    def _mtf_alignment(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        alignment = {}
        for timeframe, df in price_data.items():
            close = df["close"]
            short = close.rolling(window=self.config.momentum_window).mean()
            long = close.rolling(window=self.config.momentum_window * 3).mean()
            alignment[timeframe] = int(short.iloc[-1] > long.iloc[-1])
        return alignment

    def generate_signal(
        self,
        probabilities: np.ndarray,
        price_data: Dict[str, pd.DataFrame],
    ) -> str:
        """Return BUY, SELL or HOLD signal."""
        if probabilities.ndim == 2:
            bullish_prob = probabilities[:, 1][-1]
        else:
            bullish_prob = probabilities[-1]

        mtf_alignment = self._mtf_alignment(price_data)
        bullish_alignment = sum(mtf_alignment.values())
        bearish_alignment = len(mtf_alignment) - bullish_alignment

        if bullish_prob >= self.config.probability_threshold and bullish_alignment >= self.config.alignment_required:
            return "BUY"
        if (1 - bullish_prob) >= self.config.probability_threshold and bearish_alignment >= self.config.alignment_required:
            return "SELL"
        return "HOLD"

