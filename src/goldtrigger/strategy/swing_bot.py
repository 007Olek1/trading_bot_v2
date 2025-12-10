import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from goldtrigger.selector import Selector, SymbolCandidate
from goldtrigger.risk import PositionSizeInputs, SltpInputs, calc_position_size, calc_sl_tp_targets
from goldtrigger.strategy.indicators import (
    ema,
    ema_series,
    compute_atr,
    rsi,
    sma,
    ema_crossed,
)
from goldtrigger.utils.logging import get_child_logger, setup_logging
from goldtrigger.disco import Disco57Learner, TradeFeatures


@dataclass
class SwingSignal:
    symbol: str
    direction: str
    reason: str = ""
    snapshot: Optional[dict] = None


@dataclass
class SwingPosition:
    symbol: str
    side: str
    size: float
    entry_price: float
    sl_distance_pct: float
    sl_price: float
    tp_price: float
    entry_time: float
    trade_id: Optional[int] = None
    current_pnl: float = 0.0
    trailing_active: bool = False
    tp1_triggered: bool = False
    max_roi: float = 0.0
    locked_sl_roi: float = 0.0
    disco_features: Optional[TradeFeatures] = None


class SwingBot:
    def __init__(
        self,
        api,
        selector: Selector,
        notifier,
        trade_db,
        disco,
        disco_learner: Optional[Disco57Learner] = None,
        mode="paper",
    ):
        self.api = api
        self.selector = selector
        self.notifier = notifier
        self.trade_db = trade_db
        self.disco = disco
        self.disco_learner = disco_learner
        self.disco57 = disco_learner or disco  # for Telegram commands compatibility
        self.mode = mode
        self.logger = get_child_logger(setup_logging(), "swing_bot")
        self.positions: Dict[str, SwingPosition] = {}
        self.last_trade_timestamp = 0
        self.symbol_cooldowns: Dict[str, float] = {}
        self.symbol_cooldown_sec = int(os.getenv("SYMBOL_COOLDOWN_SEC", 5400))
        self.max_positions = int(os.getenv("MAX_POSITIONS", 3))
        self.risk_pct = float(os.getenv("RISK_PCT", 0.006))
        self.leverage = float(os.getenv("LEVERAGE", 20.0))
        self.tp1_roi_target = float(os.getenv("TP1_ROI_TARGET", 0.30))  # ROI (0.30 = 30%)
        self.trailing_roi_step = float(os.getenv("TRAILING_ROI_STEP", 0.10))  # ROI gap for trailing
        self.selector_alert_hours = float(os.getenv("SELECTOR_ALERT_HOURS", 5))
        self._selector_alert_sent = False
        self.trading_enabled = True
        self.daily_pnl = 0.0
        self.daily_trades = 0

    async def start(self):
        self.logger.info("SwingBot ready — waiting for selector scans")
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_startup_message()

    async def run_scan(self):
        if not self.trading_enabled:
            self.logger.debug("Trading paused, skipping scan")
            return
        prioritized = self.selector.get_prioritized()
        if not prioritized:
            self.logger.info("Selector did not provide candidates yet")
            await self._maybe_send_selector_alert(has_candidates=False)
            return
        for symbol in prioritized:
            await self.evaluate_symbol(symbol)
        await self._maybe_send_selector_alert(has_candidates=True)

    async def evaluate_symbol(self, symbol: str):
        if symbol in self.positions:
            self.logger.debug("%s already has open position", symbol)
            return
        if self.is_in_cooldown(symbol):
            self.logger.debug("%s in cooldown", symbol)
            return
        if self.get_open_positions_count() >= self.max_positions:
            self.logger.info("Max positions reached, skipping %s", symbol)
            return
        signal = await self._build_signal(symbol)
        if not signal:
            return
        await self._attempt_entry(signal)

    async def _maybe_send_selector_alert(self, has_candidates: bool):
        metrics = getattr(self.selector, "scan_metrics", None)
        if not metrics or not self.notifier or not self.notifier.enabled:
            return
        if has_candidates:
            self._selector_alert_sent = False
            return
        last_candidate = metrics.get("last_candidate_ts") or 0
        hours_since = (time.time() - last_candidate) / 3600
        if hours_since < self.selector_alert_hours:
            self._selector_alert_sent = False
            return
        if self._selector_alert_sent:
            return
        self._selector_alert_sent = True
        total_scans = metrics.get("total_scans", 0)
        checked = metrics.get("symbols_checked", 0)
        found = metrics.get("candidates_found", 0)
        message = (
            "⚠️ <b>Selector heartbeat</b>\n"
            f"Нет кандидатов уже {hours_since:.1f}ч.\n"
            f"Сканов: {total_scans}, проверено символов: {checked}, найдено за последнюю попытку: {found}.\n"
            "Проверь фильтры или подключение к бирже."
        )
        await self.notifier.send_message(message)

    async def _build_signal(self, symbol: str) -> Optional[SwingSignal]:
        try:
            candles_1h = await self.api.fetch_ohlcv(symbol, timeframe="1h", limit=200)
            candles_30m = await self.api.fetch_ohlcv(symbol, timeframe="30m", limit=200)
        except Exception as exc:
            self.logger.warning("Failed to fetch candles for %s: %s", symbol, exc)
            return None

        if len(candles_1h) < 120 or len(candles_30m) < 120:
            return None

        closes_1h = [c[4] for c in candles_1h]
        volumes_1h = [c[5] for c in candles_1h]
        last_close_1h = closes_1h[-1]
        ema50 = ema(closes_1h[-120:], 50)
        ema200 = ema(closes_1h[-200:], 200)
        ema9_series = ema_series(closes_1h[-60:], 9)
        ema21_series = ema_series(closes_1h[-60:], 21)
        rsi_1h = rsi(closes_1h, 14)
        vol_ratio = (sum(volumes_1h[-2:]) / 2) / max(sma(volumes_1h, 20), 1e-9)

        direction: Optional[str] = None
        cross_bull = ema_crossed(ema9_series, ema21_series, lookback=3, direction="bullish")
        cross_bear = ema_crossed(ema9_series, ema21_series, lookback=3, direction="bearish")

        if (
            last_close_1h > max(ema50, ema200)
            and cross_bull
            and 45 <= rsi_1h <= 70
            and vol_ratio >= 1.2
        ):
            direction = "long"
        elif (
            last_close_1h < min(ema50, ema200)
            and cross_bear
            and 30 <= rsi_1h <= 55
            and vol_ratio >= 1.2
        ):
            direction = "short"
        else:
            return None

        # 30m confirmation: pullback to EMA50/200 with green/red candle
        closes_30m = [c[4] for c in candles_30m]
        ema50_30m = ema(closes_30m[-120:], 50)
        last_open_30m = candles_30m[-1][1]
        last_close_30m = candles_30m[-1][4]
        body_pct = abs(last_close_30m - last_open_30m) / last_open_30m
        if direction == "long":
            pullback_ok = (
                last_close_30m >= ema50_30m and body_pct >= 0.005 and last_close_30m > last_open_30m
            )
        else:
            pullback_ok = (
                last_close_30m <= ema50_30m and body_pct >= 0.005 and last_close_30m < last_open_30m
            )
        if not pullback_ok:
            return None

        atr_value = compute_atr(candles_1h, 14)
        atr_pct = atr_value / last_close_1h if last_close_1h else 0

        candidate = next(
            (c for c in self.selector._prioritized if c.symbol == symbol), None
        )
        selector_reason = candidate.reason if candidate else "manual_scan"

        snapshot = {
            "close_1h": last_close_1h,
            "ema50": ema50,
            "ema200": ema200,
            "rsi": rsi_1h,
            "vol_ratio": vol_ratio,
            "atr_pct": atr_pct,
            "selector_reason": selector_reason,
        }

        return SwingSignal(
            symbol=symbol,
            direction=direction,
            reason=f"strategy_{selector_reason}",
            snapshot=snapshot,
        )

    async def _collect_trade_features(self, symbol: str, ticker: Dict) -> Optional[TradeFeatures]:
        if not self.disco_learner:
            return None
        try:
            candles = await self.api.fetch_ohlcv(symbol, timeframe="1h", limit=200)
        except Exception as exc:
            self.logger.debug("Failed to fetch candles for Disco57 features %s: %s", symbol, exc)
            return None
        if len(candles) < 40:
            return None
        try:
            return self.disco_learner.extract_features(candles, ticker)
        except Exception as exc:
            self.logger.debug("Feature extraction failed for %s: %s", symbol, exc)
            return None

    async def _attempt_entry(self, signal: SwingSignal):
        if not self.trading_enabled:
            self.logger.info("Trading disabled, skip entry for %s", signal.symbol)
            return
        balance_info = await self.api.get_account_balance()
        usdt_total = float(balance_info.get("total", {}).get("USDT", 0))
        if usdt_total <= 0:
            self.logger.warning("Balance unavailable, cannot size %s", signal.symbol)
            return
        ticker = await self.api.fetch_ticker(signal.symbol)
        entry_price = float(ticker.get("last") or 0)
        if entry_price <= 0:
            return
        sl_distance_pct = max(signal.snapshot.get("atr_pct", 0.015) * 1.8, 0.01)
        pos_inputs = PositionSizeInputs(
            balance=usdt_total,
            risk_pct=self.risk_pct,
            sl_distance_pct=sl_distance_pct,
            entry_price=entry_price,
        )
        size_contracts = calc_position_size(pos_inputs)
        if size_contracts <= 0:
            self.logger.info("Size zero for %s (balance=%s)", signal.symbol, usdt_total)
            return
        sltp = calc_sl_tp_targets(
            SltpInputs(
                entry_price=entry_price,
                sl_distance_pct=sl_distance_pct,
                rr_multiplier=6.0,
                direction=signal.direction,
            )
        )
        risk_amount = usdt_total * self.risk_pct
        tp_target_usd = risk_amount * 6.0
        order = await self._execute_entry(signal, size_contracts, entry_price)
        if not order:
            self.logger.warning("Entry order failed for %s", signal.symbol)
            return
        fill_price = float(
            order.get("average")
            or order.get("price")
            or ((order.get("info") or {}).get("avgPrice"))
            or entry_price
        )
        trade_features: Optional[TradeFeatures] = None
        disco_confidence_pct = float(signal.snapshot.get("disco_confidence", 0) if signal.snapshot else 0)
        if self.disco_learner:
            trade_features = await self._collect_trade_features(signal.symbol, ticker)
            if trade_features:
                allow, confidence = self.disco_learner.predict(trade_features, signal.direction)
                disco_confidence_pct = confidence * 100
                self.logger.info(
                    "Disco57 confidence %.1f%% for %s %s (allow=%s)",
                    disco_confidence_pct,
                    signal.symbol,
                    signal.direction,
                    allow,
                )
        trade_id = None
        if self.trade_db:
            trade_id = self.trade_db.add_trade_open(
                symbol=signal.symbol,
                side=signal.direction,
                entry_price=fill_price,
                quantity=size_contracts,
                signal_strength=int(signal.snapshot.get("signal_strength", 0) if signal.snapshot else 0),
                disco_confidence=disco_confidence_pct,
            )
        sl_ok = await self.api.set_stop_loss(signal.symbol, signal.direction, sltp["sl"])
        tp_ok = await self.api.set_take_profit(signal.symbol, signal.direction, sltp["tp_final"])
        if not (sl_ok and tp_ok):
            self.logger.warning("Failed to set SL/TP for %s (SL ok=%s TP ok=%s)", signal.symbol, sl_ok, tp_ok)
        self.positions[signal.symbol] = SwingPosition(
            symbol=signal.symbol,
            side=signal.direction,
            size=size_contracts,
            entry_price=fill_price,
            sl_distance_pct=sl_distance_pct,
            sl_price=sltp["sl"],
            tp_price=sltp["tp_final"],
            entry_time=time.time(),
            trade_id=trade_id,
            disco_features=trade_features,
        )
        self.daily_trades += 1
        self.last_trade_timestamp = time.time()
        self.set_cooldown(signal.symbol)
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_trade_opened(
                symbol=signal.symbol,
                side=signal.direction,
                entry_price=fill_price,
                sl_usd=risk_amount,
                tp_usd=tp_target_usd,
                sl_price=sltp["sl"],
                signal_strength=int(signal.snapshot.get("signal_strength", 0) if signal.snapshot else 0),
                disco_confidence=disco_confidence_pct,
            )

    def get_open_positions_count(self) -> int:
        return len(self.positions)

    def is_in_cooldown(self, symbol: str) -> bool:
        return self.symbol_cooldowns.get(symbol, 0) > time.time()

    def set_cooldown(self, symbol: str, seconds: Optional[int] = None):
        self.symbol_cooldowns[symbol] = time.time() + (seconds or self.symbol_cooldown_sec)

    async def update_positions(self):
        """
        Sync open positions with exchange and close if SL/TP hit.
        """
        try:
            exchange_positions = await self.api.fetch_positions()
        except Exception as exc:
            self.logger.error("Failed to fetch exchange positions: %s", exc)
            return
        exchange_map = {p["symbol"]: p for p in exchange_positions}
        # Close positions that no longer exist on exchange
        for symbol in list(self.positions.keys()):
            if symbol not in exchange_map:
                self.logger.info("Exchange no longer has %s, removing local state", symbol)
                self.positions.pop(symbol, None)
        # Update existing positions
        for symbol, position in self.positions.items():
            ex_pos = exchange_map.get(symbol)
            if not ex_pos:
                continue
            try:
                ticker = await self.api.fetch_ticker(symbol)
                current_price = float(ticker.get("last") or 0)
            except Exception as exc:
                self.logger.warning("Failed to fetch ticker for %s: %s", symbol, exc)
                continue
            if current_price <= 0:
                continue
            if position.side == "long":
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                hit_sl = current_price <= position.sl_price * 0.999
                hit_tp = current_price >= position.tp_price * 0.999
            else:
                pnl_pct = (position.entry_price - current_price) / position.entry_price
                hit_sl = current_price >= position.sl_price * 1.001
                hit_tp = current_price <= position.tp_price * 1.001
            position.current_pnl = pnl_pct * position.size * position.entry_price
            await self._maybe_activate_trailing(position, current_price)
            if hit_sl:
                await self._close_position(symbol, current_price, "SL")
                continue
            if hit_tp:
                await self._close_position(symbol, current_price, "TP")
                continue
            # Determine if position closed remotely
            contracts = float(ex_pos.get("contracts") or 0)
            if contracts <= 0:
                self.logger.info("%s appears closed on exchange, reconciling", symbol)
                await self._close_position(symbol, current_price, "EXTERNAL")

    async def _execute_entry(self, signal: SwingSignal, size_contracts: float, reference_price: float):
        """
        Place entry order with fallback to market if limit fails.
        """
        side = "buy" if signal.direction == "long" else "sell"
        limit_price = reference_price
        try:
            order = await self.api.create_order(signal.symbol, side, size_contracts, price=limit_price)
            if order and order.get("status") in {"closed", "filled"}:
                return order
        except Exception as exc:
            self.logger.warning("Limit order failed for %s: %s", signal.symbol, exc)
        # fallback to market
        try:
            order = await self.api.create_order(signal.symbol, side, size_contracts, price=None)
            return order
        except Exception as exc:
            self.logger.error("Market order failed for %s: %s", signal.symbol, exc)
            if self.notifier:
                await self.notifier.send_error(f"Не удалось открыть {signal.symbol}: {exc}")
            return None

    async def _maybe_activate_trailing(self, position: SwingPosition, current_price: float):
        if self.leverage <= 0:
            return
        if position.side == "long":
            move_pct = (current_price - position.entry_price) / position.entry_price
        else:
            move_pct = (position.entry_price - current_price) / position.entry_price
        if move_pct <= 0:
            return
        roi_decimal = move_pct * self.leverage
        position.max_roi = max(position.max_roi, roi_decimal)
        pnl_usd = move_pct * position.size * position.entry_price
        if not position.tp1_triggered and roi_decimal >= self.tp1_roi_target:
            success = await self._set_sl_to_roi(position, self.tp1_roi_target)
            if success:
                position.tp1_triggered = True
                position.trailing_active = True
                position.locked_sl_roi = self.tp1_roi_target
                self.logger.info(
                    "%s TP1 reached (%.1f%% ROI). SL locked at %.1f%% ROI and trailing enabled",
                    position.symbol,
                    roi_decimal * 100,
                    self.tp1_roi_target * 100,
                )
                if self.notifier and self.notifier.enabled:
                    await self.notifier.send_trailing_activated(
                        symbol=position.symbol,
                        side=position.side,
                        profit_usd=pnl_usd,
                        roi_pct=roi_decimal * 100,
                        locked_roi_pct=self.tp1_roi_target * 100,
                    )
            return
        if not position.trailing_active or self.trailing_roi_step <= 0:
            return
        delta = roi_decimal - position.locked_sl_roi
        if delta < self.trailing_roi_step - 1e-6:
            return
        increments = int(delta // self.trailing_roi_step)
        if increments <= 0:
            return
        new_lock = position.locked_sl_roi + increments * self.trailing_roi_step
        success = await self._set_sl_to_roi(position, new_lock)
        if success:
            position.locked_sl_roi = new_lock
            self.logger.info(
                "%s trailing SL advanced to %.1f%% ROI (step +%.1f%%)",
                position.symbol,
                position.locked_sl_roi * 100,
                self.trailing_roi_step * 100,
            )

    async def _close_position(self, symbol: str, exit_price: float, reason: str):
        position = self.positions.get(symbol)
        if not position:
            return
        if reason != "EXTERNAL":
            try:
                await self.api.close_position(symbol)
            except Exception as exc:
                self.logger.error("Failed to close %s: %s", symbol, exc)
                return
        if position.side == "long":
            pnl_usd = (exit_price - position.entry_price) * position.size
        else:
            pnl_usd = (position.entry_price - exit_price) * position.size
        duration_min = int((time.time() - position.entry_time) / 60)
        if self.disco_learner and position.disco_features:
            try:
                self.disco_learner.learn(position.disco_features, position.side, pnl_usd)
            except Exception as exc:
                self.logger.warning("Disco57 learn failed for %s: %s", symbol, exc)
        self.daily_pnl += pnl_usd
        self.positions.pop(symbol, None)
        if self.trade_db:
            self.trade_db.close_trade(
                symbol=symbol,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                reason=reason,
                trailing_activated=position.trailing_active,
            )
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_trade_closed(
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                reason=reason,
                daily_pnl=self.daily_pnl,
                duration_min=duration_min,
            )

    async def _set_sl_to_roi(self, position: SwingPosition, roi_decimal: float) -> bool:
        price = self._price_for_roi(position, roi_decimal)
        if price <= 0:
            return False
        try:
            updated = await self.api.set_stop_loss(position.symbol, position.side, price)
        except Exception as exc:
            self.logger.warning("Failed to update SL for %s: %s", position.symbol, exc)
            return False
        if updated:
            position.sl_price = price
        return updated

    def _price_for_roi(self, position: SwingPosition, roi_decimal: float) -> float:
        move_pct = roi_decimal / self.leverage
        if position.side == "long":
            return position.entry_price * (1 + move_pct)
        return position.entry_price * (1 - move_pct)
