from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

from .bybit_client import BybitClient
from .config import TradingConfig
from .models import SignalSnapshot

logger = logging.getLogger(__name__)

CANDLE_LIMIT = 200
CACHE_TTL_SECONDS = 20.0
SCAN_CONCURRENCY = 8
HOT_SYMBOL_LIMIT = 40


class StrategyScanner:
    """Сканирование монет и построение сигналов (каркас)."""

    def __init__(
        self,
        client: BybitClient,
        trading_cfg: TradingConfig,
        symbols: Iterable[str],
    ):
        self._client = client
        self._cfg = trading_cfg
        self._symbols = list(symbols)
        self._candle_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._cache_ttl = CACHE_TTL_SECONDS
        self._symbol_scores: Dict[str, float] = {}
        self._fast_tf = str(trading_cfg.scan_fast_timeframe or "5")
        self._slow_tf = str(trading_cfg.scan_slow_timeframe or "15")
        concurrency = min(max(1, SCAN_CONCURRENCY), max(1, len(self._symbols)))
        self._fetch_semaphore = asyncio.Semaphore(concurrency)

    async def run_scan(self) -> List[SignalSnapshot]:
        results: List[SignalSnapshot] = []
        symbol_order = self._ordered_symbols()
        tasks = [asyncio.create_task(self._scan_symbol(symbol)) for symbol in symbol_order]
        stop_collection = False
        try:
            for task in asyncio.as_completed(tasks):
                symbol, snapshot = await task
                if snapshot:
                    self._register_symbol_hit(symbol, snapshot)
                    results.append(snapshot)
                    if len(results) >= self._cfg.max_positions:
                        stop_collection = True
                        break
                if stop_collection:
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _evaluate_symbol(self, symbol: str) -> Optional[SignalSnapshot]:
        candles_fast, candles_slow = await asyncio.gather(
            self._fetch_cached_candles(symbol, self._fast_tf),
            self._fetch_cached_candles(symbol, self._slow_tf),
        )
        if len(candles_fast) < 60 or len(candles_slow) < 60:
            return None

        closes_fast = [float(c[4]) for c in candles_fast]
        closes_slow = [float(c[4]) for c in candles_slow]
        last_fast = closes_fast[-1]
        last_slow = closes_slow[-1]

        volume_ratio = self._volume_ratio(candles_fast)
        momentum = closes_fast[-1] - closes_fast[-15]
        atr = self._atr(candles_fast)
        atr_pct = atr / last_fast if last_fast else 0
        momentum_norm = momentum / atr if atr else 0
        adx = self._adx(candles_fast)
        ema_fast_fast = self._ema(closes_fast, 9)
        ema_slow_fast = self._ema(closes_fast, 21)
        ema_fast_slow = self._ema(closes_slow, 9)
        ema_slow_slow = self._ema(closes_slow, 21)

        direction: Optional[str] = None
        vol_condition = volume_ratio >= 1.5
        trend_condition = adx > 20 and atr_pct > 0.001

        if (
            vol_condition
            and trend_condition
            and momentum_norm > 0.8
            and ema_fast_fast > ema_slow_fast
            and ema_fast_slow > ema_slow_slow
            and last_fast > ema_slow_fast
        ):
            direction = "long"
        elif (
            vol_condition
            and trend_condition
            and momentum_norm < -0.8
            and ema_fast_fast < ema_slow_fast
            and ema_fast_slow < ema_slow_slow
            and last_fast < ema_slow_fast
        ):
            direction = "short"

        if not direction:
            return None

        return SignalSnapshot(
            symbol=symbol,
            side=direction,
            entry_price=last_fast,
            price_5m=last_fast,
            price_15m=last_slow,
            volume_ratio=volume_ratio,
            momentum=momentum,
            momentum_norm=momentum_norm,
            atr=atr,
            atr_pct=atr_pct,
            adx=adx,
            ema_fast_5m=ema_fast_fast,
            ema_slow_5m=ema_slow_fast,
            ema_fast_15m=ema_fast_slow,
            ema_slow_15m=ema_slow_slow,
            volatility=atr_pct,
        )

    async def _scan_symbol(self, symbol: str) -> Tuple[str, Optional[SignalSnapshot]]:
        async with self._fetch_semaphore:
            try:
                snapshot = await self._evaluate_symbol(symbol)
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Ошибка анализа %s: %s", symbol, exc)
                snapshot = None
        return symbol, snapshot

    def _ordered_symbols(self) -> List[str]:
        if not self._symbol_scores:
            symbols = list(self._symbols)
            random.shuffle(symbols)
            return symbols
        hot_candidates = sorted(
            self._symbol_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        hot = [symbol for symbol, _ in hot_candidates[:HOT_SYMBOL_LIMIT]]
        cold = [symbol for symbol in self._symbols if symbol not in hot]
        random.shuffle(hot)
        random.shuffle(cold)
        return hot + cold

    def _register_symbol_hit(self, symbol: str, snapshot: SignalSnapshot) -> None:
        score = snapshot.volume_ratio * max(snapshot.atr_pct, 1e-6)
        score += snapshot.adx * 0.01
        self._symbol_scores[symbol] = score

    async def _fetch_cached_candles(self, symbol: str, interval: str) -> List:
        key = (symbol, interval)
        entry = self._candle_cache.get(key)
        now = time.time()
        if entry:
            fetched_at = entry["ts"]
            if isinstance(fetched_at, (int, float)) and now - fetched_at <= self._cache_ttl:
                data = entry["data"]
                if isinstance(data, list):
                    return data
        candles = await self._client.fetch_ohlcv(symbol, interval=interval, limit=CANDLE_LIMIT)
        if candles:
            trimmed = candles[-CANDLE_LIMIT:]
            self._candle_cache[key] = {"ts": now, "data": trimmed}
            return trimmed
        return candles

    @staticmethod
    def _volume_ratio(candles) -> float:
        volumes = [float(c[5]) for c in candles[-20:]]
        avg = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
        return volumes[-1] / avg if avg else 0

    @staticmethod
    def _atr(candles, period: int = 14) -> float:
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        closes = [float(c[4]) for c in candles]
        trs = []
        for i in range(1, len(highs)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        if len(trs) < period:
            return 0.0
        return sum(trs[-period:]) / period

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        if len(values) < period:
            return 0.0
        k = 2 / (period + 1)
        ema = sum(values[:period]) / period
        for value in values[period:]:
            ema = value * k + ema * (1 - k)
        return ema

    @staticmethod
    def _adx(candles, period: int = 14) -> float:
        if len(candles) <= period:
            return 0.0
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        closes = [float(c[4]) for c in candles]

        tr_list = []
        plus_dm = []
        minus_dm = []
        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
            tr_list.append(tr)

        if len(tr_list) < period:
            return 0.0

        tr_smooth = sum(tr_list[:period])
        plus_smooth = sum(plus_dm[:period])
        minus_smooth = sum(minus_dm[:period])

        dx_values: List[float] = []
        for i in range(period, len(tr_list)):
            if i > period:
                tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[i]
                plus_smooth = plus_smooth - (plus_smooth / period) + plus_dm[i]
                minus_smooth = minus_smooth - (minus_smooth / period) + minus_dm[i]

            plus_di = 100 * plus_smooth / tr_smooth if tr_smooth else 0.0
            minus_di = 100 * minus_smooth / tr_smooth if tr_smooth else 0.0
            denom = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / denom if denom else 0.0
            dx_values.append(dx)

        if not dx_values:
            return 0.0

        adx = sum(dx_values[:period]) / min(len(dx_values), period)
        for dx in dx_values[period:]:
            adx = ((adx * (period - 1)) + dx) / period
        return adx


__all__ = ["StrategyScanner"]
