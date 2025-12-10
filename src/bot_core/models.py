from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple


Side = Literal["long", "short"]


@dataclass(slots=True)
class SignalSnapshot:
    symbol: str
    side: Side
    entry_price: float
    price_5m: float
    price_15m: float
    volume_ratio: float
    momentum: float
    momentum_norm: float
    atr: float
    atr_pct: float
    adx: float
    ema_fast_5m: float
    ema_slow_5m: float
    ema_fast_15m: float
    ema_slow_15m: float
    volatility: float
    spread_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)
    disco_decision: Optional[str] = None
    disco_confidence: Optional[float] = None
    ml_decision_id: Optional[int] = None


@dataclass(slots=True)
class Position:
    symbol: str
    side: Side
    entry_price: float
    quantity_total: float
    quantity_remaining: float
    leverage: int
    margin_usd: float
    notional_total: float
    notional_remaining: float
    stop_loss: float
    tp_prices: Tuple[float, float, float]
    breakeven_price: float
    trail_step: float
    trail_activate_at_r: float
    sl_distance: float
    risk_usd: float
    opened_at: float = field(default_factory=time.time)
    trailing_active: bool = False
    tp1_taken: bool = False
    tp2_taken: bool = False
    tp3_taken: bool = False
    realized_pnl_usd: float = 0.0
    trade_id: Optional[int] = None
    snapshot: Optional[SignalSnapshot] = None
    metadata: Dict[str, float] = field(default_factory=dict)
    partial_allocations: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    spread_buffer: float = 0.0
    last_trailing_price: Optional[float] = None

    def qty_from_notional(self, notional: float) -> float:
        return notional / self.entry_price if self.entry_price else 0.0

    def pnl_usd(self, current_price: float, qty: Optional[float] = None) -> float:
        qty = qty if qty is not None else self.quantity_remaining
        if self.side == "long":
            return (current_price - self.entry_price) * qty
        return (self.entry_price - current_price) * qty

    def r_multiple(self, current_price: float) -> float:
        if self.sl_distance <= 0:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry_price) / self.sl_distance
        return (self.entry_price - current_price) / self.sl_distance

    def price_to_r(self, target_r: float) -> float:
        if self.side == "long":
            return self.entry_price + self.sl_distance * target_r
        return self.entry_price - self.sl_distance * target_r


@dataclass(slots=True)
class StrategyDecision:
    symbol: str
    side: Side
    entry_price: float
    stop_loss: float
    tp_prices: Tuple[float, float, float]
    breakeven_price: float
    trail_step: float
    trail_activate_at_r: float
    quantity: float
    risk_usd: float
    sl_distance: float
    notional: float
    partial_allocations: Tuple[float, float, float]
    spread_buffer: float
    signal_strength: int = 0
    snapshot: Optional[SignalSnapshot] = None


__all__ = ["Position", "SignalSnapshot", "Side", "StrategyDecision"]
