#!/usr/bin/env python3
"""
Bootstrap Disco57 model using historical trades.

Loads recent closed trades from TradeHistoryDB, reconstructs feature snapshots
from current Bybit market data, and feeds them into Disco57Learner.learn().

Usage:
    python scripts/bootstrap_disco57.py --hours 720 --limit 200
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from goldtrigger.bybit_api import BybitAPI
from goldtrigger.db import TradeHistoryDB
from goldtrigger.disco import Disco57Learner, TradeFeatures


logger = logging.getLogger("bootstrap_disco57")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


async def _collect_features(
    api: BybitAPI,
    learner: Disco57Learner,
    symbol: str,
    semaphore: asyncio.Semaphore,
    cache: Optional[Dict[str, Dict]] = None,
    cache_updates: Optional[Dict[str, Dict]] = None,
):
    """Fetch candles/ticker and extract features for Disco57 (bounded concurrency)."""
    cached_entry = cache.get(symbol) if cache else None
    if cached_entry and cached_entry.get("candles") and cached_entry.get("ticker"):
        candles = cached_entry["candles"]
        ticker = cached_entry["ticker"]
    else:
        async with semaphore:
            candles = await api.fetch_ohlcv(symbol, timeframe="1h", limit=200)
            if len(candles) < 40:
                raise RuntimeError("not enough candles")
            ticker = await api.fetch_ticker(symbol)
        if cache_updates is not None:
            cache_updates[symbol] = {
                "candles": candles,
                "ticker": ticker,
                "updated_at": time.time(),
            }

    return learner.extract_features(candles, ticker)


async def _preload_features(
    symbols: List[str],
    api: BybitAPI,
    learner: Disco57Learner,
    concurrency: int,
    cache: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Optional[TradeFeatures]]:
    cache: Dict[str, Optional[TradeFeatures]] = {}
    semaphore = asyncio.Semaphore(max(1, concurrency))
    cache_updates: Dict[str, Dict] = {}

    async def fetch(symbol: str):
        try:
            cache[symbol] = await _collect_features(
                api, learner, symbol, semaphore, cache=cache if cache else None, cache_updates=cache_updates
            )
            logger.info("Prepared features for %s", symbol)
        except Exception as exc:  # pragma: no cover - diagnostics
            logger.warning("Failed to preload %s: %s", symbol, exc)
            cache[symbol] = None

    await asyncio.gather(*(fetch(symbol) for symbol in symbols))
    return cache, cache_updates


def load_cache(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r") as fp:
            return json.load(fp)
    except Exception as exc:
        logger.warning("Failed to load cache %s: %s", path, exc)
        return {}


def save_cache(path: Optional[str], cache_updates: Dict[str, Dict], base_cache: Dict[str, Dict]):
    if not path or not cache_updates:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    base_cache.update(cache_updates)
    try:
        with cache_path.open("w") as fp:
            json.dump(base_cache, fp)
    except Exception as exc:
        logger.warning("Failed to write cache %s: %s", path, exc)


async def bootstrap(hours: int, limit: int, model_path: str, concurrency: int, cache_path: Optional[str]):
    db = TradeHistoryDB()
    trades: List[Dict] = db.get_recent_trades(hours=hours, limit=limit)
    closed_trades = [t for t in trades if t.get("status") == "closed"]
    if not closed_trades:
        logger.warning("No closed trades found in the selected window (hours=%s, limit=%s)", hours, limit)
        return

    api = BybitAPI()
    learner = Disco57Learner(model_path=model_path)

    cache_data = load_cache(cache_path)
    unique_symbols = sorted({trade["symbol"] for trade in closed_trades})
    logger.info(
        "Preloading features for %s symbols (concurrency=%s, cached=%s)",
        len(unique_symbols),
        concurrency,
        sum(1 for symbol in unique_symbols if symbol in cache_data),
    )
    feature_cache, cache_updates = await _preload_features(
        unique_symbols, api, learner, concurrency, cache=cache_data or None
    )
    save_cache(cache_path, cache_updates, cache_data)

    success = 0
    skipped = 0
    total = len(closed_trades)
    for idx, trade in enumerate(closed_trades, start=1):
        symbol = trade["symbol"]
        pnl_usd = float(trade.get("pnl_usd") or 0.0)
        direction = trade.get("side", "").lower()
        if direction not in ("long", "short"):
            logger.debug("Skip trade %s: unsupported side '%s'", symbol, direction)
            skipped += 1
            continue
        try:
            logger.info("(%s/%s) processing %s %s pnl=%s", idx, total, symbol, direction, pnl_usd)
            features = feature_cache.get(symbol)
            if not features:
                raise RuntimeError("features not available for symbol")
            learner.learn(features, direction, pnl_usd)
            success += 1
        except Exception as exc:  # pragma: no cover - diagnostics
            logger.warning("Skip %s due to error: %s", symbol, exc)
            skipped += 1

    await api.close()
    logger.info(
        "Bootstrap finished. Trades processed: %s (trained=%s, skipped=%s). Current win rate: %.1f%%",
        total,
        success,
        skipped,
        learner.get_win_rate(),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap Disco57 learner from historical trades")
    parser.add_argument("--hours", type=int, default=720, help="Number of hours back to fetch trades (default: 720)")
    parser.add_argument("--limit", type=int, default=200, help="Limit trades fetched from DB (default: 200)")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of parallel API requests (default: 4)")
    parser.add_argument(
        "--cache-path",
        type=str,
        default="data/bootstrap_candles.json",
        help="Optional path to store/reuse candle/ticker cache",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("DISCO57_MODEL_PATH", "/opt/GoldTrigger_bot/data/disco57_model.json"),
        help="Path to Disco57 model file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(bootstrap(args.hours, args.limit, args.model_path, args.concurrency, args.cache_path))


if __name__ == "__main__":
    main()
