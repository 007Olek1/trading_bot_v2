#!/usr/bin/env python3
"""Утилита для бэктеста фильтров StrategyScanner на исторических свечах Bybit."""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bot_core.bybit_client import BybitClient  # type: ignore  # noqa: E402
from bot_core.config import load_config  # type: ignore  # noqa: E402
from bot_core.scanner import StrategyScanner, CANDLE_LIMIT  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest фильтров Scanner на исторических свечах")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT",
        help="Список символов через запятую (например BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=600,
        help="Количество 5m свечей для загрузки (макс 1000)",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=48,
        help="Сколько 5m баров анализировать после сигнала для оценки результата",
    )
    parser.add_argument(
        "--tp-mult",
        type=float,
        default=3.0,
        help="Коэффициент ATR для цели (по умолчанию 3 ATR)",
    )
    parser.add_argument(
        "--sl-mult",
        type=float,
        default=1.0,
        help="Коэффициент ATR для стопа (по умолчанию 1 ATR)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Параллельное количество символов для backtest",
    )
    return parser.parse_args()


async def fetch_candles(client: BybitClient, symbol: str, interval: str, limit: int) -> List[List[float]]:
    candles = await client.fetch_ohlcv(symbol, interval=interval, limit=limit)
    return candles or []


def evaluate_window(window_5m: List[List[float]], window_15m: List[List[float]]):
    if len(window_5m) < 60 or len(window_15m) < 60:
        return None
    closes_5m = [float(c[4]) for c in window_5m]
    closes_15m = [float(c[4]) for c in window_15m]
    last_5m = closes_5m[-1]
    last_15m = closes_15m[-1]
    volume_ratio = StrategyScanner._volume_ratio(window_5m)  # type: ignore[attr-defined]
    atr = StrategyScanner._atr(window_5m)  # type: ignore[attr-defined]
    atr_pct = atr / last_5m if last_5m else 0
    momentum = closes_5m[-1] - closes_5m[-15]
    momentum_norm = momentum / atr if atr else 0
    adx = StrategyScanner._adx(window_5m)  # type: ignore[attr-defined]
    ema_fast_5m = StrategyScanner._ema(closes_5m, 9)  # type: ignore[attr-defined]
    ema_slow_5m = StrategyScanner._ema(closes_5m, 21)  # type: ignore[attr-defined]
    ema_fast_15m = StrategyScanner._ema(closes_15m, 9)  # type: ignore[attr-defined]
    ema_slow_15m = StrategyScanner._ema(closes_15m, 21)  # type: ignore[attr-defined]

    direction: Optional[str] = None
    vol_condition = volume_ratio >= 1.5
    trend_condition = adx > 20 and atr_pct > 0.001

    if (
        vol_condition
        and trend_condition
        and momentum_norm > 0.8
        and ema_fast_5m > ema_slow_5m
        and ema_fast_15m > ema_slow_15m
        and last_5m > ema_slow_5m
    ):
        direction = "long"
    elif (
        vol_condition
        and trend_condition
        and momentum_norm < -0.8
        and ema_fast_5m < ema_slow_5m
        and ema_fast_15m < ema_slow_15m
        and last_5m < ema_slow_5m
    ):
        direction = "short"

    if not direction:
        return None

    return {
        "direction": direction,
        "entry_price": last_5m,
        "atr": atr,
        "atr_pct": atr_pct,
        "adx": adx,
        "volume_ratio": volume_ratio,
        "momentum_norm": momentum_norm,
    }


def simulate_trade(
    candles_5m: List[List[float]],
    start_idx: int,
    direction: str,
    entry_price: float,
    atr: float,
    lookahead: int,
    tp_mult: float,
    sl_mult: float,
) -> float:
    if atr <= 0:
        return 0.0
    sign = 1 if direction == "long" else -1
    target = entry_price + sign * atr * tp_mult
    stop = entry_price - sign * atr * sl_mult
    end_idx = min(len(candles_5m), start_idx + 1 + lookahead)
    for idx in range(start_idx + 1, end_idx):
        high = float(candles_5m[idx][2])
        low = float(candles_5m[idx][3])
        if direction == "long":
            if high >= target:
                return tp_mult
            if low <= stop:
                return -sl_mult
        else:
            if low <= target:
                return tp_mult
            if high >= stop:
                return -sl_mult
    final_close = float(candles_5m[end_idx - 1][4])
    pnl_r = (final_close - entry_price) / atr * sign
    return max(min(pnl_r, tp_mult), -sl_mult * 2)


async def backtest_symbol(
    client: BybitClient,
    symbol: str,
    limit: int,
    lookahead: int,
    tp_mult: float,
    sl_mult: float,
) -> Dict[str, float]:
    candles_5m = await fetch_candles(client, symbol, "5", limit)
    if len(candles_5m) < 120:
        return {"symbol": symbol, "signals": 0}
    limit_15m = max(200, limit // 3 + 50)
    candles_15m = await fetch_candles(client, symbol, "15", limit_15m)
    if len(candles_15m) < 80:
        return {"symbol": symbol, "signals": 0}

    ptr_15 = 0
    r_values: List[float] = []
    snapshots = 0

    for idx in range(60, len(candles_5m) - lookahead):
        ts = float(candles_5m[idx][0])
        while ptr_15 < len(candles_15m) and float(candles_15m[ptr_15][0]) <= ts:
            ptr_15 += 1
        window_5m = candles_5m[max(0, idx + 1 - CANDLE_LIMIT) : idx + 1]
        if ptr_15 == 0:
            continue
        window_15m = candles_15m[max(0, ptr_15 - CANDLE_LIMIT) : ptr_15]
        snapshot = evaluate_window(window_5m, window_15m)
        if not snapshot:
            continue
        snapshots += 1
        r = simulate_trade(
            candles_5m,
            idx,
            snapshot["direction"],
            snapshot["entry_price"],
            snapshot["atr"],
            lookahead,
            tp_mult,
            sl_mult,
        )
        r_values.append(r)

    if snapshots == 0:
        return {"symbol": symbol, "signals": 0}

    wins = sum(1 for r in r_values if r > 0)
    losses = sum(1 for r in r_values if r < 0)
    avg_r = statistics.mean(r_values) if r_values else 0.0
    return {
        "symbol": symbol,
        "signals": snapshots,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / snapshots * 100,
        "avg_r": avg_r,
        "best_r": max(r_values) if r_values else 0.0,
        "worst_r": min(r_values) if r_values else 0.0,
    }


async def run_backtest(args: argparse.Namespace):
    config = load_config()
    client = BybitClient(config.bybit)
    symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def worker(symbol: str):
        async with semaphore:
            logger = logging.getLogger("backtest")
            logger.info("Запускаем бэктест %s", symbol)
            result = await backtest_symbol(
                client,
                symbol,
                limit=args.limit,
                lookahead=args.lookahead,
                tp_mult=args.tp_mult,
                sl_mult=args.sl_mult,
            )
            return result

    results = await asyncio.gather(*[worker(sym) for sym in symbols])

    print("\n===== Backtest Summary =====")
    for res in results:
        symbol = res.get("symbol")
        signals = res.get("signals", 0)
        if not signals:
            print(f"{symbol}: недостаточно данных или нет сигналов")
            continue
        print(
            f"{symbol:<10} signals={signals:>4} wins={res['wins']:>3} losses={res['losses']:>3} "
            f"win_rate={res['win_rate']:.1f}% avg_R={res['avg_r']:.2f} "
            f"best={res['best_r']:.2f} worst={res['worst_r']:.2f}"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    arguments = parse_args()
    try:
        asyncio.run(run_backtest(arguments))
    except KeyboardInterrupt:
        print("Остановлено пользователем")
