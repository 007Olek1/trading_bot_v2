from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from trade_history_db import TradeHistoryDB

from .bybit_client import BybitClient
from .config import TradingConfig
from .models import Position, Side, SignalSnapshot, StrategyDecision

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingError(Exception):
    reason: str
    qty: float
    notional: float
    risk_usd: float


class PositionManager:
    """Хранит локальное состояние позиций и синхронизирует с TradeHistoryDB."""

    def __init__(
        self,
        client: BybitClient,
        trade_db: TradeHistoryDB,
        trading_cfg: TradingConfig,
    ):
        self._client = client
        self._trade_db = trade_db
        self._cfg = trading_cfg
        self._positions: Dict[str, Position] = {}

    @property
    def positions(self) -> Dict[str, Position]:
        return self._positions

    def has_capacity(self) -> bool:
        return len(self._positions) < self._cfg.max_positions

    def in_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def build_decision(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        signal_strength: int = 0,
        snapshot: Optional[SignalSnapshot] = None,
    ) -> StrategyDecision:
        """Построить решение с уровнем риска в R."""
        atr_value = 0.0
        if snapshot:
            atr_value = snapshot.atr or 0.0
        if atr_value <= 0:
            atr_value = entry_price * 0.015

        sl_distance = max(atr_value, entry_price * 0.005)
        if side == "long":
            stop_loss = entry_price - sl_distance
            tp_prices = (
                entry_price + sl_distance,
                entry_price + sl_distance * 2,
                entry_price + sl_distance * 3,
            )
        else:
            stop_loss = entry_price + sl_distance
            tp_prices = (
                entry_price - sl_distance,
                entry_price - sl_distance * 2,
                entry_price - sl_distance * 3,
            )

        trail_step = atr_value * self._cfg.trail_step_atr_mult
        if trail_step <= 0:
            trail_step = entry_price * 0.002

        spread_pct = snapshot.spread_pct if snapshot else 0.0
        spread_buffer = entry_price * (spread_pct / 100)

        decision = StrategyDecision(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp_prices=tp_prices,
            breakeven_price=entry_price,
            trail_step=trail_step,
            trail_activate_at_r=self._cfg.trail_activate_at_r,
            quantity=0.0,
            risk_usd=0.0,
            sl_distance=sl_distance,
            notional=0.0,
            partial_allocations=self._cfg.partial_takes,
            spread_buffer=spread_buffer,
            signal_strength=signal_strength,
            snapshot=snapshot,
        )
        return decision

    def register_position(self, decision: StrategyDecision, quantity: float) -> Position:
        notional = decision.notional if decision.notional else quantity * decision.entry_price
        pos = Position(
            symbol=decision.symbol,
            side=decision.side,
            entry_price=decision.entry_price,
            quantity_total=quantity,
            quantity_remaining=quantity,
            leverage=self._cfg.leverage,
            margin_usd=self._cfg.margin_usd,
            notional_total=notional,
            notional_remaining=notional,
            stop_loss=decision.stop_loss,
            tp_prices=decision.tp_prices,
            breakeven_price=decision.breakeven_price,
            trail_step=decision.trail_step,
            trail_activate_at_r=decision.trail_activate_at_r,
            sl_distance=decision.sl_distance,
            risk_usd=decision.risk_usd,
            trade_id=None,
            snapshot=decision.snapshot,
            metadata={
                "signal_strength": decision.signal_strength,
                "atr": decision.snapshot.atr if decision.snapshot else 0.0,
                "atr_pct": decision.snapshot.atr_pct if decision.snapshot else 0.0,
                "adx": decision.snapshot.adx if decision.snapshot else 0.0,
            },
            partial_allocations=decision.partial_allocations,
            spread_buffer=decision.spread_buffer,
        )
        self._positions[pos.symbol] = pos
        return pos

    def apply_partial_close(self, pos: Position, qty_closed: float) -> None:
        qty_closed = min(qty_closed, pos.quantity_remaining)
        pos.quantity_remaining = max(pos.quantity_remaining - qty_closed, 0.0)
        pos.notional_remaining = pos.quantity_remaining * pos.entry_price
        if pos.quantity_remaining <= 1e-8:
            pos.quantity_remaining = 0.0

    def finalize_position(self, symbol: str) -> Optional[Position]:
        return self._positions.pop(symbol, None)

    async def sync_open_from_exchange(self) -> None:
        """Загрузить актуальные позиции с биржи (использовать при запуске)."""
        positions = await self._client.fetch_positions()
        active = [p for p in positions if float(p.get("size", 0) or 0) > 0]
        for pos in active:
            symbol = pos["symbol"]
            if symbol in self._positions:
                continue
            side = "long" if pos.get("side") == "Buy" else "short"
            entry_price = float(pos.get("avgPrice", 0) or 0)
            quantity = float(pos.get("size", 0) or 0)
            logger.warning(
                "Найдена активная позиция %s %s с биржи. Добавите вручную через TradeHistoryDB.",
                symbol,
                side,
            )

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def record_open_trade(self, pos: Position) -> None:
        pos.trade_id = self._trade_db.add_trade_open(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            quantity=pos.quantity_total,
            risk_usd=pos.risk_usd,
            sl_distance=pos.sl_distance,
            sl_price=pos.stop_loss,
            tp_price=pos.tp_prices[0] if pos.tp_prices else 0.0,
            signal_strength=int(pos.metadata.get("signal_strength", 0)),
            disco_confidence=pos.metadata.get("disco_confidence", 0.0),
        )

    def record_close_trade(self, pos: Position, exit_price: float, reason: str, trailing: bool) -> float:
        pnl_remaining = pos.pnl_usd(exit_price)
        total_pnl = pos.realized_pnl_usd + pnl_remaining
        entry_notional = pos.entry_price * pos.quantity_total if pos.quantity_total else 1.0
        pnl_pct = (total_pnl / entry_notional) * 100
        pnl_r = total_pnl / pos.risk_usd if pos.risk_usd else 0.0
        self._trade_db.close_trade(
            symbol=pos.symbol,
            exit_price=exit_price,
            pnl_usd=total_pnl,
            pnl_pct=pnl_pct,
            pnl_r=pnl_r,
            reason=reason,
            trailing_activated=trailing,
        )
        return total_pnl

    def record_partial_trade(
        self,
        pos: Position,
        exit_price: float,
        qty_closed: float,
        reason: str,
    ) -> float:
        pnl_usd = pos.pnl_usd(exit_price, qty=qty_closed)
        entry_notional = pos.entry_price * qty_closed if qty_closed else 1.0
        pnl_pct = (pnl_usd / entry_notional) * 100
        if pos.trade_id:
            self._trade_db.add_trade_partial(
                trade_id=pos.trade_id,
                exit_price=exit_price,
                quantity=qty_closed,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                reason=reason,
            )
        pos.realized_pnl_usd += pnl_usd
        return pnl_usd


__all__ = ["PositionManager"]
