import asyncio
import math
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple

from goldtrigger.utils.logging import get_child_logger, setup_logging
from goldtrigger.disco import Disco57Learner, TradeFeatures

try:
    from SYMBOLS_145 import TRADING_SYMBOLS_145 as UNIVERSAL_SYMBOLS
except Exception:  # pragma: no cover - fallback
    UNIVERSAL_SYMBOLS = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
    ]


ATR_LOOKBACK = 14
EMA_LONG = 200
EMA_MEDIUM = 50
MAX_TOP_SYMBOLS = 10


@dataclass
class SymbolCandidate:
    symbol: str
    score: float = 0.0
    direction: str = "long"
    reason: str = ""
    features: Dict[str, float] = field(default_factory=dict)


def ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2 / (period + 1)
    ema_value = values[0]
    for value in values[1:]:
        ema_value = value * k + ema_value * (1 - k)
    return ema_value


def compute_atr(candles: List[List[float]], period: int = ATR_LOOKBACK) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1][4]
        high = candles[i][2]
        low = candles[i][3]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    return mean(trs[-period:])


class Selector:
    """
    Swing selector scans the 145-symbol universe, applies ATR/volume/EMA filters,
    and consults Disco57-Swing before prioritizing candidates.
    """

    def __init__(self, exchange, disco_interface, config: Optional[Dict] = None):
        self.exchange = exchange
        self.disco = disco_interface
        self.config = config or {}
        self.disco_learner: Optional[Disco57Learner] = config.get("disco_learner")
        self.logger = get_child_logger(setup_logging(), "selector")
        self._prioritized: List[SymbolCandidate] = []
        self._lock = asyncio.Lock()
        self.symbols = self.config.get("symbols", UNIVERSAL_SYMBOLS)
        self.top_limit = self.config.get("top_limit", MAX_TOP_SYMBOLS)
        self.min_volume_ratio = self.config.get("min_volume_ratio", 1.2)
        self.min_atr_ratio = self.config.get("min_atr_ratio", 1.5)
        self.scan_metrics = {
            "total_scans": 0,
            "last_candidate_ts": time.time(),
            "symbols_checked": 0,
            "candidates_found": 0,
        }

    async def initialize(self):
        self.logger.info("Selector initialized (%d symbols)", len(self.symbols))
        await self.perform_full_rescan()

    async def perform_full_rescan(self):
        async with self._lock:
            self.logger.info("Full rescan across %d symbols", len(self.symbols))
            self._prioritized = await self._scan_symbols(self.symbols)

    async def scan_pool(self) -> List[SymbolCandidate]:
        async with self._lock:
            refresh_symbols = self._pick_subset()
            self.logger.info("Scanning subset of %d symbols", len(refresh_symbols))
            subset_candidates = await self._scan_symbols(refresh_symbols)
            merged = {c.symbol: c for c in self._prioritized}
            for candidate in subset_candidates:
                merged[candidate.symbol] = candidate
            ranked = sorted(
                merged.values(), key=lambda c: c.score, reverse=True
            )[: self.top_limit]
            self._prioritized = ranked
            return list(self._prioritized)

    def get_prioritized(self) -> List[str]:
        return [candidate.symbol for candidate in self._prioritized]

    def _pick_subset(self) -> List[str]:
        # Simple rotation: first N symbols chunked by top_limit*2
        step = self.top_limit * 2
        offset = getattr(self, "_subset_offset", 0)
        subset = self.symbols[offset : offset + step]
        if not subset:
            offset = 0
            subset = self.symbols[:step]
        self._subset_offset = offset + step
        return subset

    async def _scan_symbols(self, symbols: List[str]) -> List[SymbolCandidate]:
        candidates: List[SymbolCandidate] = []
        self.scan_metrics["total_scans"] += 1
        scan_candidates = 0
        for symbol in symbols:
            result = await self._evaluate_symbol(symbol)
            if result:
                candidates.append(result)
                scan_candidates += 1
                self.scan_metrics["last_candidate_ts"] = time.time()
        self.scan_metrics["symbols_checked"] = len(symbols)
        self.scan_metrics["candidates_found"] = scan_candidates
        return sorted(candidates, key=lambda c: c.score, reverse=True)[: self.top_limit]

    async def _evaluate_symbol(self, symbol: str) -> Optional[SymbolCandidate]:
        try:
            candles_1h = await self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=200)
        except Exception as exc:
            self.logger.debug("Failed to fetch 1h candles for %s: %s", symbol, exc)
            return None

        if len(candles_1h) < 60:
            return None

        closes = [c[4] for c in candles_1h]
        volumes = [c[5] for c in candles_1h]
        last_close = closes[-1]
        ema50 = ema(closes[-100:], 50)
        ema200 = ema(closes[-EMA_LONG:], EMA_LONG)

        atr_current = compute_atr(candles_1h[-(ATR_LOOKBACK + 30) :])
        atr_mean = compute_atr(candles_1h[-(ATR_LOOKBACK * 7) :])
        if atr_current <= 0 or atr_mean <= 0:
            return None
        atr_ratio = atr_current / atr_mean
        if atr_ratio < self.min_atr_ratio:
            return None

        vol_recent = mean(volumes[-2:])
        vol_mean = mean(volumes[-48:])
        if vol_mean == 0 or (vol_recent / vol_mean) < self.min_volume_ratio:
            return None

        direction = "long" if last_close > ema200 else "short"
        if direction == "long" and last_close <= ema50:
            return None
        if direction == "short" and last_close >= ema50:
            return None

        features = {
            "symbol": symbol,
            "atr_ratio": atr_ratio,
            "volume_ratio": vol_recent / vol_mean,
            "ema50_dist": (last_close - ema50) / ema50 if ema50 else 0,
            "ema200_dist": (last_close - ema200) / ema200 if ema200 else 0,
        }

        disco_confidence = None
        if self.disco_learner:
            ticker = await self.exchange.fetch_ticker(symbol)
            trade_features = self.disco_learner.extract_features(candles_1h, ticker)
            allow, confidence = self.disco_learner.predict(trade_features, direction)
            disco_confidence = confidence
            if not allow:
                self.logger.info("Disco57 filtered %s (confidence %.1f%%)", symbol, confidence * 100)
                return None
            features["disco_confidence"] = confidence * 100
            features["trade_features"] = trade_features

        score = atr_ratio * 0.6 + features["volume_ratio"] * 0.4
        candidate = SymbolCandidate(
            symbol=symbol,
            score=score,
            direction=direction,
            reason="atr_volume_trend",
            features=features,
        )
        self.logger.info(
            "Selector candidate %s dir=%s score=%.2f disco=%s",
            symbol,
            direction,
            score,
            reason,
        )
        return candidate
