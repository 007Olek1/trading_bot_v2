"""Risk management utilities for controlling exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class RiskConfig:
    max_concurrent_positions: int = 3
    leverage: int = 10
    base_position_size_usd: float = 1.0
    max_account_risk_pct: float = 2.0
    min_position_size_usd: float = 0.5
    max_position_size_usd: float = 5.0
    taker_fee_rate: float = 0.0006  # 0.06% per side


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def position_size(self, balance: float) -> float:
        account_risk = balance * (self.config.max_account_risk_pct / 100.0)
        gross_size = min(self.config.base_position_size_usd, account_risk)
        # Adjust for expected taker fee on entry + exit
        fee_multiplier = 1 + (self.config.taker_fee_rate * 2)
        return gross_size / fee_multiplier

    def can_open_position(self, open_positions: int) -> bool:
        return open_positions < self.config.max_concurrent_positions

    def leverage(self) -> int:
        return self.config.leverage

    def adjust_position_size(self, scale: float) -> float:
        proposed = self.config.base_position_size_usd * scale
        bounded = max(self.config.min_position_size_usd, min(self.config.max_position_size_usd, proposed))
        self.config.base_position_size_usd = bounded
        return self.config.base_position_size_usd

    def summary(self) -> Dict[str, float | int]:
        return {
            "max_concurrent_positions": self.config.max_concurrent_positions,
            "leverage": self.config.leverage,
            "base_position_size": self.config.base_position_size_usd,
            "max_account_risk_pct": self.config.max_account_risk_pct,
        }

