#!/usr/bin/env python3
"""
Preload Bybit candles/tickers into bootstrap cache with retries.

Usage:
    PYTHONPATH=src python scripts/preload_candles.py --hours 720 --limit 200 \
        --cache-path data/bootstrap_candles.json --concurrency 4
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


logger = logging.getLogger("preload_candles")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MAX_RETRIES = 5
RETRY_DELAY = 2.0


async def fetch_with_retry(api_call, *args, **kwargs):
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await api_call(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - diagnostics
            if attempt == MAX_RETRIES:
                raise
            logger.warning("Retrying %s (attempt %s/%s): %s", api_call.__name__, attempt, MAX_RETRIES, exc)
            await asyncio.sleep(delay)
            delay *= 1.5


def load_cache(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        with path.open("r") as fp:
            return json.load(fp)
    except Exception as exc:
        logger.warning("Failed to read cache %s: %s", path, exc)
        return {}


def save_cache(path: Path, data: Dict[str, Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(data, fp)


async def preload(symbols: List[str], cache_path: str, concurrency: int):
    cache_file = Path(cache_path)
    cache = load_cache(cache_file)
    api = BybitAPI()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    updated = 0

    async def worker(symbol: str):
        nonlocal updated
        async with semaphore:
            candles = await fetch_with_retry(api.fetch_ohlcv, symbol, timeframe="1h", limit=200)
            ticker = await fetch_with_retry(api.fetch_ticker, symbol)
        cache[symbol] = {
            "candles": candles,
            "ticker": ticker,
            "updated_at": time.time(),
        }
        updated += 1
        logger.info("Cached candles for %s (%s/%s)", symbol, updated, len(symbols))

    await asyncio.gather(*(worker(symbol) for symbol in symbols))
    await api.close()
    save_cache(cache_file, cache)
    logger.info("Finished caching %s symbols into %s", updated, cache_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Preload candles/tickers for Disco57 bootstrap")
    parser.add_argument("--hours", type=int, default=720, help="Hours back for selecting trades")
    parser.add_argument("--limit", type=int, default=200, help="Limit of trades to inspect")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel fetchers")
    parser.add_argument(
        "--cache-path",
        type=str,
        default="data/bootstrap_candles.json",
        help="Path to cache JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = TradeHistoryDB()
    trades = db.get_recent_trades(hours=args.hours, limit=args.limit)
    symbols = sorted({trade["symbol"] for trade in trades})
    if not symbols:
        logger.warning("No symbols found in trade history.")
        return
    logger.info("Preloading %s symbols with concurrency=%s", len(symbols), args.concurrency)
    asyncio.run(preload(symbols, args.cache_path, args.concurrency))


if __name__ == "__main__":
    main()
