from statistics import mean
from typing import List, Optional


def ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2 / (period + 1)
    ema_value = values[0]
    for value in values[1:]:
        ema_value = value * k + ema_value * (1 - k)
    return ema_value


def ema_series(values: List[float], period: int) -> List[Optional[float]]:
    if not values:
        return []
    k = 2 / (period + 1)
    ema_value = values[0]
    series: List[Optional[float]] = []
    for value in values:
        ema_value = value * k + ema_value * (1 - k)
        series.append(ema_value)
    return series


def sma(values: List[float], period: int) -> float:
    if len(values) < period or period <= 0:
        return 0.0
    return mean(values[-period:])


def rsi(values: List[float], period: int = 14) -> float:
    if len(values) <= period:
        return 50.0
    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = values[-i] - values[-i - 1]
        if delta >= 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss if avg_loss else 0
    return 100 - (100 / (1 + rs))


def compute_atr(candles: List[List[float]], period: int = 14) -> float:
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


def ema_crossed(
    fast_series: List[Optional[float]],
    slow_series: List[Optional[float]],
    lookback: int = 3,
    direction: str = "bullish",
) -> bool:
    if not fast_series or not slow_series:
        return False
    length = min(len(fast_series), len(slow_series))
    if length < lookback + 1:
        return False
    rng = range(1, lookback + 1)
    for offset in rng:
        fast_now = fast_series[-offset]
        slow_now = slow_series[-offset]
        fast_prev = fast_series[-offset - 1]
        slow_prev = slow_series[-offset - 1]
        if None in (fast_now, slow_now, fast_prev, slow_prev):
            continue
        if direction == "bullish" and fast_prev <= slow_prev and fast_now > slow_now:
            return True
        if direction == "bearish" and fast_prev >= slow_prev and fast_now < slow_now:
            return True
    return False
