from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from SYMBOLS_145 import TRADING_SYMBOLS_145
from trade_history_db import TradeHistoryDB
from .bybit_client import BybitClient
from .config import Config
from .cooldown import CooldownManager
from .models import Position, SignalSnapshot, StrategyDecision
from .positions import PositionManager
from .risk_settings import FIXED_SL_USD, FIXED_TP_USD, RR_RATIO, SL_TOLERANCE, TP_TOLERANCE
from .scanner import StrategyScanner
from telegram_notifier import TelegramNotifier
from .ml import Disco57Wrapper

logger = logging.getLogger(__name__)

POSITION_IDX = 0  # regular mode
COOLDOWN_SECONDS = 90 * 60
POSITION_CHECK_INTERVAL = 10
MIN_ADX = 20.0
MIN_ATR_PCT = 0.0015  # 0.15%
MIN_SPREAD_PCT = 0.07
MAX_ATR_PCT = 0.05  # 5%
SL_MIN_GAP_PCT = 0.001  # 0.1%
SL_MIN_GAP_ABS = 0.0002
SL_TICK_BUFFER = 3
MAX_ENTRY_DRIFT_R = 0.35  # skip if price moved >35% of SL distance


class SwingBot:
    """Основной класс бота: сканирование, открытия и сопровождение позиций."""

    def __init__(
        self,
        config: Config,
        client: BybitClient,
        trade_db: TradeHistoryDB,
        notifier: Optional[TelegramNotifier] = None,
        ml: Optional[Disco57Wrapper] = None,
        symbols: Optional[Iterable[str]] = None,
    ):
        self.config = config
        self.client = client
        self.trade_db = trade_db
        self.notifier = notifier
        self.ml = ml
        self.cooldowns = CooldownManager(COOLDOWN_SECONDS)
        self.positions = PositionManager(client, trade_db, config.trading)
        self.scanner = StrategyScanner(client, config.trading, symbols or TRADING_SYMBOLS_145)

        self.trading_enabled = True
        self.daily_pnl = 0.0
        self._scan_task: Optional[asyncio.Task] = None
        self._update_task: Optional[asyncio.Task] = None
        self._sl_retry_attempts = 3
        self._sl_retry_delay = 1.0
        self._stop_event = asyncio.Event()
        self._last_trade_ts: float = time.time()
        self._inactivity_alert_sent = False
        self._started_at: float = time.time()
        self._stopped = False
        logger.info(
            "Scanner configured for %sm/%sm, max concurrent positions %d",
            self.config.trading.scan_fast_timeframe,
            self.config.trading.scan_slow_timeframe,
            self.config.trading.max_positions,
        )

    async def start(self):
        self._started_at = time.time()
        await self.positions.sync_open_from_exchange()
        self._scan_task = asyncio.create_task(self._scan_loop())
        self._update_task = asyncio.create_task(self._update_loop())

    async def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.trading_enabled = False
        if self._scan_task:
            self._scan_task.cancel()
        if self._update_task:
            self._update_task.cancel()
        self._stop_event.set()

    async def _scan_loop(self):
        min_delay, max_delay = self.config.trading.scan_interval_minutes
        while True:
            try:
                if self.trading_enabled and self.positions.has_capacity():
                    logger.info("Запуск сканирования рынка (%d символов)", len(TRADING_SYMBOLS_145))
                    signals = await self.scanner.run_scan()
                    for snapshot in signals:
                        await self._handle_signal(snapshot)
                        if not self.positions.has_capacity():
                            break
                delay = random.randint(min_delay * 60, max_delay * 60)
                logger.info("Пауза до следующего скана: %d сек", delay)
                await asyncio.sleep(delay)
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Ошибка в цикле сканирования: %s", exc)
                await asyncio.sleep(30)

    async def _handle_signal(self, snapshot: SignalSnapshot):
        if not self.trading_enabled:
            return
        if self.positions.in_position(snapshot.symbol):
            return
        if self.cooldowns.is_on_cooldown(snapshot.symbol):
            logger.info("%s на кулдауне ещё %ds", snapshot.symbol, self.cooldowns.remaining(snapshot.symbol))
            return

        if not self.positions.has_capacity():
            logger.info("Нет свободных слотов для %s", snapshot.symbol)
            return

        if not self._passes_snapshot_filters(snapshot):
            return

        if self.ml and not self.ml.allow_signal(snapshot):
            logger.info("Disco57 заблокировал %s", snapshot.symbol)
            return

        decision = self.positions.build_decision(
            symbol=snapshot.symbol,
            side=snapshot.side,
            entry_price=snapshot.entry_price,
            snapshot=snapshot,
        )
        await self._open_position(decision)

    async def _open_position(self, decision: StrategyDecision):
        try:
            spread_pct = await self._ensure_spread_within_limits(decision.symbol, decision)
            if spread_pct is None:
                logger.warning("Не удалось определить спред для %s, пропускаем вход", decision.symbol)
                return
            allowed_spread = max(
                MIN_SPREAD_PCT,
                decision.spread_buffer / decision.entry_price * 100 if decision.entry_price else MIN_SPREAD_PCT,
            )
            if spread_pct > allowed_spread:
                logger.info(
                    "%s пропущен: спред %.4f%% выше лимита %.4f%%",
                    decision.symbol,
                    spread_pct,
                    allowed_spread,
                )
                return

            last_price = await self._get_last_price(decision.symbol)
            if last_price is None:
                logger.warning("Нет last_price для %s, пропускаем вход", decision.symbol)
                return
            if decision.sl_distance > 0:
                drift_r = abs(last_price - decision.entry_price) / decision.sl_distance
                if drift_r > MAX_ENTRY_DRIFT_R:
                    logger.info(
                        "%s пропущен: цена ушла %.2fR против snapshot (лимит %.2fR)",
                        decision.symbol,
                        drift_r,
                        MAX_ENTRY_DRIFT_R,
                    )
                    return
            self._shift_decision_levels(decision, last_price)

            await self._populate_quantity(decision)
            if decision.quantity <= 0:
                logger.info("%s пропущен: не удалось вычислить объём", decision.symbol)
                return

            await self.client.set_leverage(decision.symbol, self.config.trading.leverage)

            filters = await self.client.get_symbol_filters(decision.symbol)
            price_tick = filters.get("price_tick") or 0.0

            await self.client.create_order(
                symbol=decision.symbol,
                side="Buy" if decision.side == "long" else "Sell",
                qty=decision.quantity,
                position_idx=POSITION_IDX,
                reduce_only=False,
                order_type="Market",
            )

            pos = self.positions.register_position(decision, decision.quantity)
            if isinstance(pos.metadata, dict):
                pos.metadata["price_tick"] = price_tick
            safe_sl = self._adjust_stop_loss(pos, pos.stop_loss)
            pos.stop_loss = safe_sl
            decision.stop_loss = safe_sl
            self.positions.record_open_trade(pos)

            sl_applied = await self._set_trading_stop_with_retry(
                symbol=decision.symbol,
                stop_loss=safe_sl,
                take_profit=pos.tp_prices[0] if pos.tp_prices else None,
                context="initial SLTP",
            )
            if not sl_applied:
                logger.error("%s: не удалось установить стартовый SL после %s попыток", decision.symbol, self._sl_retry_attempts)
                if self.notifier:
                    await self.notifier.send_error(f"{decision.symbol}: не удалось установить стартовый SL, позиция без защиты!")
            else:
                self._sync_trade_levels(pos.symbol, sl_price=safe_sl, tp_price=pos.tp_prices[0] if pos.tp_prices else None)

            if self.notifier:
                await self.notifier.send_trade_opened(
                    symbol=pos.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    quantity=pos.quantity_total,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.tp_prices[0],
                    leverage=self.config.trading.leverage,
                    margin_usd=self.config.trading.margin_usd,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Не удалось открыть позицию %s: %s", decision.symbol, exc)

    async def _populate_quantity(self, decision: StrategyDecision) -> None:
        leverage = self.config.trading.leverage
        margin = self.config.trading.margin_usd
        exposure = margin * leverage
        sl_distance = self._fixed_sl_distance(decision.entry_price)
        tp_distance = self._fixed_tp_distance(decision.entry_price)
        decision.sl_distance = sl_distance
        decision.stop_loss = decision.entry_price - sl_distance if decision.side == "long" else decision.entry_price + sl_distance
        decision.tp_prices = (
            decision.entry_price + tp_distance if decision.side == "long" else decision.entry_price - tp_distance,
            decision.entry_price + tp_distance if decision.side == "long" else decision.entry_price - tp_distance,
            decision.entry_price + tp_distance if decision.side == "long" else decision.entry_price - tp_distance,
        )
        decision.breakeven_price = decision.entry_price
        risk_usd = FIXED_SL_USD
        qty = exposure / decision.entry_price
        notional = qty * decision.entry_price
        if qty < self.config.trading.min_qty:
            self._log_position_rejection(
                decision,
                reason="MIN_QTY",
                qty=qty,
                notional=notional,
                risk_usd=risk_usd,
                balance=exposure,
            )
            decision.quantity = 0.0
            return
        if notional < self.config.trading.min_notional_usd:
            self._log_position_rejection(
                decision,
                reason="MIN_NOTIONAL",
                qty=qty,
                notional=notional,
                risk_usd=risk_usd,
                balance=exposure,
            )
            decision.quantity = 0.0
            return

        normalized_qty, filters = await self.client.normalize_quantity(decision.symbol, qty)
        if normalized_qty <= 0:
            self._log_position_rejection(
                decision,
                reason="LOT_FILTER",
                qty=qty,
                notional=notional,
                risk_usd=risk_usd,
                balance=exposure,
            )
            decision.quantity = 0.0
            return
        if abs(normalized_qty - qty) > 1e-8:
            logger.info(
                "%s объём скорректирован: raw=%.8f -> rounded=%.8f (step=%.8f, min=%.8f)",
                decision.symbol,
                qty,
                normalized_qty,
                filters.get("qty_step", 0.0),
                filters.get("min_qty", 0.0),
            )

        notional = normalized_qty * decision.entry_price
        min_notional = filters.get("min_notional") or 0.0
        if min_notional and notional < min_notional:
            self._log_position_rejection(
                decision,
                reason="EXCHANGE_MIN_NOTIONAL",
                qty=normalized_qty,
                notional=notional,
                risk_usd=risk_usd,
                balance=exposure,
            )
            decision.quantity = 0.0
            return

        decision.quantity = normalized_qty
        decision.notional = notional
        decision.risk_usd = risk_usd
        logger.info(
            "%s план входа %s: qty=%.6f notional=%.2f margin=$%.2f lev=%dx SL=%s TP=%s R/R=1:%.1f",
            decision.symbol,
            decision.side.upper(),
            decision.quantity,
            decision.notional,
            margin,
            leverage,
            f"{decision.stop_loss:.6f}",
            f"{decision.tp_prices[0]:.6f}",
            RR_RATIO,
        )

    @staticmethod
    def _log_position_rejection(
        decision: StrategyDecision,
        reason: str,
        qty: float,
        notional: float,
        risk_usd: float,
        balance: float,
    ) -> None:
        logger.info(
            "%s пропущен (%s): qty=%.6f notional=%.2f risk=%.2f balance=%.2f sl=%.6f",
            decision.symbol,
            reason,
            qty,
            notional,
            risk_usd,
            balance,
            decision.sl_distance,
        )

    def _passes_snapshot_filters(self, snapshot: SignalSnapshot) -> bool:
        if snapshot.adx <= MIN_ADX:
            logger.info("%s отклонён: ADX=%.2f ниже %.2f", snapshot.symbol, snapshot.adx, MIN_ADX)
            return False
        if snapshot.atr_pct <= MIN_ATR_PCT:
            logger.info("%s отклонён: ATR%%=%.4f ниже %.4f", snapshot.symbol, snapshot.atr_pct, MIN_ATR_PCT)
            return False
        if snapshot.atr_pct >= MAX_ATR_PCT:
            logger.info("%s отклонён: ATR%%=%.4f выше %.4f", snapshot.symbol, snapshot.atr_pct, MAX_ATR_PCT)
            return False
        return True

    async def _ensure_spread_within_limits(self, symbol: str, decision: StrategyDecision) -> Optional[float]:
        try:
            ticker = await self.client.fetch_ticker(symbol)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Не удалось получить тикер %s: %s", symbol, exc)
            return None

        bid = self._extract_price(ticker, ("bid1Price", "bidPrice", "bestBidPrice"))
        ask = self._extract_price(ticker, ("ask1Price", "askPrice", "bestAskPrice"))
        if not bid or not ask:
            return None
        mid = (bid + ask) / 2
        if mid <= 0:
            return None
        spread_pct = ((ask - bid) / mid) * 100
        return spread_pct

    async def _get_last_price(self, symbol: str) -> Optional[float]:
        """Безопасно получить последнюю цену инструмента."""
        try:
            ticker = await self.client.fetch_ticker(symbol)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Не удалось получить last price %s: %s", symbol, exc)
            return None

        price = self._extract_price(
            ticker,
            ("lastPrice", "last", "markPrice", "indexPrice", "bid1Price", "ask1Price"),
        )
        if price is None or price <= 0:
            logger.warning("Ticker %s не содержит валидного last price: %s", symbol, ticker)
            return None
        return price

    def _shift_decision_levels(self, decision: StrategyDecision, last_price: float) -> None:
        """
        Смещает уровни решения к актуальной цене, сохраняя расстояния SL/TP.

        Snapshot мог быть собран несколькими секундами ранее, поэтому перед размещением
        ордера мы подтягиваем entry/SL/TP к последней цене, чтобы не открываться с дрейфом.
        """
        if last_price <= 0:
            return
        sl_distance = self._fixed_sl_distance(last_price)
        tp_distance = self._fixed_tp_distance(last_price)
        decision.entry_price = last_price
        if decision.side == "long":
            decision.stop_loss = last_price - sl_distance
            decision.tp_prices = (
                last_price + tp_distance,
                last_price + tp_distance,
                last_price + tp_distance,
            )
        else:
            decision.stop_loss = last_price + sl_distance
            decision.tp_prices = (
                last_price - tp_distance,
                last_price - tp_distance,
                last_price - tp_distance,
            )
        decision.breakeven_price = last_price

    def _sync_trade_levels(self, symbol: str, sl_price: Optional[float], tp_price: Optional[float]) -> None:
        if not self.trade_db:
            return
        self.trade_db.update_trade_levels(symbol, sl_price=sl_price, tp_price=tp_price)

    @staticmethod
    def _extract_price(ticker: dict, keys: Iterable[str]) -> Optional[float]:
        for key in keys:
            value = ticker.get(key)
            if value:
                try:
                    price = float(value)
                    if price > 0:
                        return price
                except (TypeError, ValueError):
                    continue
        return None

    async def _update_loop(self):
        while True:
            try:
                await self._update_positions()
                await asyncio.sleep(POSITION_CHECK_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Ошибка обновления позиций: %s", exc)
                await asyncio.sleep(10)

    async def _update_positions(self):
        for symbol, pos in list(self.positions.positions.items()):
            ticker = await self.client.fetch_ticker(symbol)
            last_price = float(ticker.get("lastPrice") or ticker.get("last") or 0)
            if last_price <= 0:
                continue
            r_multiple = pos.r_multiple(last_price)
            logger.debug("%s R=%.2f", symbol, r_multiple)

            await self._maybe_take_partials(pos, last_price, r_multiple)
            await self._maybe_activate_trailing(pos, last_price, r_multiple)

            if (pos.side == "long" and last_price <= pos.stop_loss) or (
                pos.side == "short" and last_price >= pos.stop_loss
            ):
                await self._close_position(pos, last_price, reason="STOP_LOSS")

    async def _maybe_take_partials(self, pos: Position, price: float, r_multiple: float):
        tp_targets = (1.0, 2.0, 3.0)
        for idx, target_r in enumerate(tp_targets):
            taken_flag = [pos.tp1_taken, pos.tp2_taken, pos.tp3_taken][idx]
            if taken_flag:
                continue
            if (pos.side == "long" and price >= pos.price_to_r(target_r)) or (
                pos.side == "short" and price <= pos.price_to_r(target_r)
            ):
                allocation = pos.partial_allocations[idx]
                qty_to_close = pos.quantity_total * allocation
                await self._execute_partial(pos, price, qty_to_close, idx, target_r, allocation, r_multiple)
                if idx == 0:
                    new_sl = pos.breakeven_price + (pos.spread_buffer if pos.side == "long" else -pos.spread_buffer)
                    new_sl = self._adjust_stop_loss(pos, new_sl)
                    updated = await self._set_trading_stop_with_retry(
                        symbol=pos.symbol,
                        stop_loss=new_sl,
                        take_profit=None,
                        context="breakeven after TP1",
                    )
                    if updated:
                        pos.stop_loss = new_sl
                        self._sync_trade_levels(pos.symbol, sl_price=new_sl, tp_price=None)
                    else:
                        logger.error("%s: не удалось подтянуть SL к breakeven после TP1", pos.symbol)
                        if self.notifier:
                            await self.notifier.send_error(f"{pos.symbol}: не удалось подтянуть SL после TP1, проверьте позицию.")
                setattr(pos, ["tp1_taken", "tp2_taken", "tp3_taken"][idx], True)

    async def _execute_partial(
        self,
        pos: Position,
        price: float,
        qty_to_close: float,
        tp_index: int,
        target_r: float,
        allocation: float,
        r_multiple: float,
    ):
        close_side = "Sell" if pos.side == "long" else "Buy"
        try:
            normalized_qty, filters = await self.client.normalize_quantity(
                pos.symbol,
                min(qty_to_close, pos.quantity_remaining),
            )
            if normalized_qty <= 0:
                logger.info(
                    "%s: TP%s пропущен — объём %.8f ниже min %.8f (step %.8f)",
                    pos.symbol,
                    tp_index + 1,
                    qty_to_close,
                    filters.get("min_qty", 0.0),
                    filters.get("qty_step", 0.0),
                )
                return
            if abs(normalized_qty - qty_to_close) > 1e-8:
                logger.info(
                    "%s: TP%s объём скорректирован %.8f -> %.8f (step %.8f)",
                    pos.symbol,
                    tp_index + 1,
                    qty_to_close,
                    normalized_qty,
                    filters.get("qty_step", 0.0),
                )
            await self.client.create_order(
                symbol=pos.symbol,
                side=close_side,
                qty=normalized_qty,
                position_idx=POSITION_IDX,
                reduce_only=True,
            )
            pnl = self.positions.record_partial_trade(
                pos,
                price,
                normalized_qty,
                reason=f"TP{tp_index+1}_{target_r}R",
            )
            self.positions.apply_partial_close(pos, normalized_qty)
            self.daily_pnl += pnl
            logger.info("%s: TP%s %.1fR выполнен, PnL $%.2f", pos.symbol, tp_index + 1, target_r, pnl)
            if self.notifier:
                await self.notifier.send_partial_take(
                    symbol=pos.symbol,
                    side=pos.side,
                    tp_index=tp_index,
                    target_r=target_r,
                    price=price,
                    qty_closed=normalized_qty,
                    allocation=allocation,
                    pnl_usd=pnl,
                    r_multiple=r_multiple,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Ошибка частичного закрытия %s: %s", pos.symbol, exc)

    async def _maybe_activate_trailing(self, pos: Position, price: float, r_multiple: float):
        if r_multiple < pos.trail_activate_at_r:
            return
        if not pos.trailing_active:
            pos.trailing_active = True
            pos.last_trailing_price = price
        if pos.last_trailing_price is not None:
            move = price - pos.last_trailing_price if pos.side == "long" else pos.last_trailing_price - price
            if move < pos.trail_step:
                return
        new_sl = price - pos.trail_step if pos.side == "long" else price + pos.trail_step
        if pos.side == "long":
            new_sl = max(new_sl, pos.stop_loss)
        else:
            new_sl = min(new_sl, pos.stop_loss)
        new_sl = self._adjust_stop_loss(pos, new_sl)
        updated = await self._set_trading_stop_with_retry(
            symbol=pos.symbol,
            stop_loss=new_sl,
            take_profit=None,
            context="trailing update",
        )
        if not updated:
            logger.error("%s: не удалось обновить трейлинг SL (цель %.6f)", pos.symbol, new_sl)
            if self.notifier:
                await self.notifier.send_error(f"{pos.symbol}: не удалось обновить трейлинг SL, проверьте позицию вручную.")
            return
        pos.stop_loss = new_sl
        pos.last_trailing_price = price
        self._sync_trade_levels(pos.symbol, sl_price=new_sl, tp_price=None)
        if self.notifier:
            await self.notifier.send_trailing_update(
                symbol=pos.symbol,
                side=pos.side,
                new_sl=new_sl,
                r_multiple=r_multiple,
            )
        logger.info("%s: трейлинг обновлён -> %.6f (R=%.2f)", pos.symbol, new_sl, r_multiple)

    async def _close_position(self, pos: Position, price: float, reason: str):
        already_closed = False
        try:
            await self.client.close_position(
                symbol=pos.symbol,
                side=pos.side,
                qty=pos.quantity_remaining,
                position_idx=POSITION_IDX,
            )
        except Exception as exc:  # pylint: disable=broad-except
            message = str(exc)
            if "current position is zero" in message or "ErrCode: 110017" in message:
                logger.warning("%s: позиция уже закрыта на бирже, завершаем локально", pos.symbol)
                already_closed = True
            else:
                logger.error("Не удалось закрыть позицию %s через API: %s", pos.symbol, exc)
                return

        pnl = self.positions.record_close_trade(pos, exit_price=price, reason=reason, trailing=pos.trailing_active)
        self.daily_pnl += pnl
        self.positions.finalize_position(pos.symbol)
        self.cooldowns.set_cooldown(pos.symbol)
        if self.ml:
            self.ml.learn_from_position(pos, success=pnl > 0)
        if self.notifier:
            duration_min = int((time.time() - pos.opened_at) / 60)
            await self.notifier.send_trade_closed(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=price,
                pnl_usd=pnl,
                reason=reason,
                duration_min=duration_min,
                daily_pnl=self.daily_pnl,
            )
        logger.info("%s: позиция закрыта (%s)", pos.symbol, reason)
        self._last_trade_ts = time.time()
        self._inactivity_alert_sent = False

    async def _health_loop(self):
        while True:
            try:
                await self._send_health_report()
                await asyncio.sleep(HEALTH_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Health loop error: %s", exc)
                await asyncio.sleep(HEALTH_INTERVAL)

    async def _send_health_report(self):
        now = time.time()
        age_hours = (now - self._last_trade_ts) / 3600
        open_positions = len(self.positions.positions)
        trading_status = "🟢 ON" if self.trading_enabled else "🔴 OFF"
        ml_stats = self.ml.stats() if self.ml else {}
        started_at = datetime.fromtimestamp(self._started_at).strftime("%Y-%m-%d %H:%M")
        message = (
            "🩺 <b>GoldTrigger Health</b>\n"
            f"Status: {trading_status}\n"
            f"Open positions: {open_positions}/{self.config.trading.max_positions}\n"
            f"Daily PnL: ${self.daily_pnl:+.2f}\n"
            f"Last trade: {age_hours:.1f}h назад\n"
            f"Started: {started_at}\n"
            f"Disco57 WR: {ml_stats.get('win_rate', 0):.1f}% ({ml_stats.get('total_trades', 0)} trades)"
        )
        if self.notifier:
            await self.notifier.send_message(message)
        logger.info(
            "Health report — trading=%s, open=%d, pnl=%.2f, last_trade=%.1fh",
            trading_status,
            open_positions,
            self.daily_pnl,
            age_hours,
        )
        if age_hours * 3600 > INACTIVITY_ALERT_SECONDS and not self._inactivity_alert_sent:
            if self.notifier:
                await self.notifier.send_error(
                    f"⚠️ Нет сделок уже {age_hours:.1f}ч. Проверь бота / рынок."
                )
            self._inactivity_alert_sent = True
        if age_hours * 3600 <= INACTIVITY_ALERT_SECONDS:
            self._inactivity_alert_sent = False

    def _adjust_stop_loss(self, pos: Position, target: float) -> float:
        if target <= 0 or pos.entry_price <= 0:
            return target
        price_tick = 0.0
        if isinstance(pos.metadata, dict):
            price_tick = float(pos.metadata.get("price_tick") or 0.0)
        tick_gap = price_tick * SL_TICK_BUFFER if price_tick > 0 else 0.0
        min_gap = max(pos.entry_price * SL_MIN_GAP_PCT, SL_MIN_GAP_ABS, tick_gap)
        if pos.side == "long":
            safe = min(target, pos.entry_price - min_gap)
            if safe <= 0:
                safe = pos.entry_price * 0.5
        else:
            safe = max(target, pos.entry_price + min_gap)
        if abs(safe - target) > 1e-8:
            logger.info(
                "%s SL скорректирован: raw=%.6f -> safe=%.6f (side=%s)",
                pos.symbol,
                target,
                safe,
                pos.side,
            )
        return safe

    async def _set_trading_stop_with_retry(
        self,
        symbol: str,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        context: str,
    ) -> bool:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._sl_retry_attempts + 1):
            try:
                await self.client.set_trading_stop(
                    symbol=symbol,
                    position_idx=POSITION_IDX,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                return True
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                logger.warning(
                    "%s: не удалось обновить SL/TP (попытка %d/%d, контекст %s): %s",
                    symbol,
                    attempt,
                    self._sl_retry_attempts,
                    context,
                    exc,
                )
                if attempt < self._sl_retry_attempts:
                    await asyncio.sleep(self._sl_retry_delay)
        logger.error(
            "%s: SL/TP обновление провалено после %d попыток (контекст %s): %s",
            symbol,
            self._sl_retry_attempts,
            context,
            last_error,
        )
        return False


__all__ = ["SwingBot"]
