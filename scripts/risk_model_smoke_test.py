#!/usr/bin/env python3
"""Smoke-test the R-based risk model with fake market data."""
from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trade_history_db import TradeHistoryDB

from src.bot_core.config import (
    BybitConfig,
    Config,
    PathsConfig,
    TelegramConfig,
    TradingConfig,
)
from src.bot_core.models import SignalSnapshot
from src.bot_core.trader import SwingBot

LOG = logging.getLogger("risk_model_smoke_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class MemoryNotifier:
    """Collects bot notifications in-memory for assertions."""

    def __init__(self) -> None:
        self.events: List[Dict[str, object]] = []

    async def send_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        tp1_price: float,
        risk_usd: float,
        sl_distance: float,
    ) -> None:
        self.events.append(
            {
                "type": "open",
                "symbol": symbol,
                "side": side,
                "entry": entry_price,
                "qty": quantity,
                "sl": stop_loss,
                "tp1": tp1_price,
                "risk": risk_usd,
                "sl_distance": sl_distance,
            }
        )

    async def send_partial_take(
        self,
        symbol: str,
        side: str,
        tp_index: int,
        target_r: float,
        price: float,
        qty_closed: float,
        allocation: float,
        pnl_usd: float,
        r_multiple: float = 0.0,
    ) -> None:
        self.events.append(
            {
                "type": "partial",
                "stage": tp_index + 1,
                "target_r": target_r,
                "price": price,
                "qty": qty_closed,
                "allocation": allocation,
                "pnl": pnl_usd,
                "r": r_multiple,
            }
        )

    async def send_trailing_update(self, symbol: str, side: str, new_sl: float, r_multiple: float) -> None:
        self.events.append(
            {
                "type": "trailing",
                "symbol": symbol,
                "side": side,
                "sl": new_sl,
                "r": r_multiple,
            }
        )

    async def send_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        reason: str,
        duration_min: int,
        daily_pnl: float,
    ) -> None:
        self.events.append(
            {
                "type": "closed",
                "symbol": symbol,
                "side": side,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": pnl_usd,
                "reason": reason,
            }
        )

    async def send_startup(self) -> None:  # pragma: no cover - not used in smoke test
        return

    async def send_error(self, message: str) -> None:  # pragma: no cover - not used in smoke test
        self.events.append({"type": "error", "message": message})


class FakeBybitClient:
    """Minimal async stub for BybitClient used in the smoke test."""

    def __init__(self, starting_price: float) -> None:
        self.price = starting_price
        self.trading_stops: List[Dict[str, float]] = []
        self.orders: List[Dict[str, object]] = []
        self.equity = 1000.0

    def set_price(self, new_price: float) -> None:
        self.price = new_price

    async def fetch_account_balance(self) -> Dict[str, float]:
        return {"totalEquity": self.equity, "walletBalance": self.equity}

    async def set_leverage(self, symbol: str, leverage: int) -> None:  # pragma: no cover - noop
        return

    async def create_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        position_idx: int,
        reduce_only: bool = False,
        order_type: str = "Market",
    ) -> Dict[str, object]:
        self.orders.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "reduce_only": reduce_only,
            }
        )
        return {"orderId": f"fake_{len(self.orders)}"}

    async def set_trading_stop(
        self,
        symbol: str,
        position_idx: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> Dict[str, object]:
        self.trading_stops.append(
            {
                "symbol": symbol,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing": trailing_stop,
            }
        )
        return {"result": "ok"}

    async def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        return {
            "lastPrice": self.price,
            "bid1Price": self.price * 0.9999,
            "ask1Price": self.price * 1.0001,
        }

    async def normalize_quantity(self, symbol: str, qty: float):
        """Return qty unchanged with mock filters to satisfy bot calls."""
        step = 0.001
        min_qty = 0.001
        rounded = max(min_qty, round(qty / step) * step)
        filters = {"qty_step": step, "min_qty": min_qty, "min_notional": 0.0}
        return rounded, filters

    async def get_symbol_filters(self, symbol: str) -> Dict[str, float]:
        """Return static exchange filters used by SwingBot."""
        return {
            "price_tick": 0.01,
            "qty_step": 0.001,
            "min_qty": 0.001,
            "min_notional": 0.0,
        }

    async def close_position(self, symbol: str, side: str, qty: float, position_idx: int) -> Dict[str, object]:
        self.orders.append(
            {
                "symbol": symbol,
                "side": "Sell" if side == "long" else "Buy",
                "qty": qty,
                "reduce_only": True,
                "close": True,
            }
        )
        return {"result": "closed"}

    async def fetch_positions(self) -> List[Dict[str, object]]:  # pragma: no cover - not used here
        return []

    async def fetch_ohlcv(self, *_, **__):  # pragma: no cover - not used
        return []


async def run_smoke_test() -> None:
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)
    db_path = tmp_dir / "risk_model_test.db"
    if db_path.exists():
        db_path.unlink()

    config = Config(
        bybit=BybitConfig(api_key="fake", api_secret="fake", testnet=True),
        telegram=TelegramConfig(token="", chat_id=""),
        paths=PathsConfig(trade_db_path=db_path, disco_model_path=Path("./data/disco57_model.pkl")),
        trading=TradingConfig(
            margin_usd=10.0,
            leverage=1,
            max_positions=1,
            scan_interval_minutes=(1, 1),
            risk_fraction=0.01,
            min_notional_usd=5.0,
            min_qty=0.001,
            partial_takes=(0.4, 0.3, 0.3),
            trail_activate_at_r=2.0,
            trail_step_atr_mult=0.5,
        ),
    )

    trade_db = TradeHistoryDB(db_path=str(db_path))
    notifier = MemoryNotifier()
    fake_client = FakeBybitClient(starting_price=100.0)

    bot = SwingBot(
        config=config,
        client=fake_client,
        trade_db=trade_db,
        notifier=notifier,
        ml=None,
        symbols=["TESTUSDT"],
    )

    snapshot = SignalSnapshot(
        symbol="TESTUSDT",
        side="long",
        entry_price=100.0,
        price_5m=100.0,
        price_15m=100.0,
        volume_ratio=2.0,
        momentum=1.5,
        momentum_norm=1.5,
        atr=2.0,
        atr_pct=0.02,
        adx=25.0,
        ema_fast_5m=101.0,
        ema_slow_5m=99.0,
        ema_fast_15m=101.5,
        ema_slow_15m=99.5,
        volatility=0.02,
        spread_pct=0.02,
    )

    decision = bot.positions.build_decision(
        symbol=snapshot.symbol,
        side=snapshot.side,
        entry_price=snapshot.entry_price,
        snapshot=snapshot,
    )
    await bot._open_position(decision)

    sl_distance = decision.sl_distance
    price_path = [
        snapshot.entry_price + sl_distance * 1.05,  # TP1
        snapshot.entry_price + sl_distance * 2.05,  # TP2 + trailing activation
        snapshot.entry_price + sl_distance * 3.10,  # TP3
        snapshot.entry_price + sl_distance * 2.50,  # drift lower
        snapshot.entry_price + sl_distance * 1.50,  # hit trailing stop
    ]

    for price in price_path:
        fake_client.set_price(price)
        await bot._update_positions()

    # After the drop the trailing stop should close the leftover position
    fake_client.set_price(snapshot.entry_price + sl_distance * 1.0)
    await bot._update_positions()

    conn = trade_db._get_connection()
    trade_row = conn.execute(
        "SELECT risk_usd, sl_distance, pnl_usd, pnl_r, status FROM trades ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if trade_row:
        risk_usd, sl_distance, pnl_usd, pnl_r, status = trade_row
        if pnl_usd is None or pnl_r is None:
            LOG.info(
                "Final trade row (open): risk_usd=%.2f sl_distance=%.4f status=%s",
                risk_usd,
                sl_distance,
                status,
            )
        else:
            LOG.info(
                "Final trade row: risk_usd=%.2f sl_distance=%.4f pnl_usd=%.2f pnl_r=%.2f status=%s",
                risk_usd,
                sl_distance,
                pnl_usd,
                pnl_r,
                status,
            )
    else:
        LOG.warning("No trades recorded")
    partial_rows = conn.execute(
        "SELECT id, exit_price, quantity, pnl_usd FROM trade_partials ORDER BY id"
    ).fetchall()
    for idx, row in enumerate(partial_rows, start=1):
        LOG.info("Partial %d -> price=%.4f qty=%.6f pnl=%.2f", idx, row[1], row[2], row[3])

    LOG.info("Captured %d notifier events", len(notifier.events))
    for event in notifier.events:
        LOG.info("Notifier event: %s", event)

    # Clean up DB after inspection
    trade_db._conn.close()
    shutil.copy(db_path, tmp_dir / "risk_model_test_snapshot.db")  # keep snapshot for manual review


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
