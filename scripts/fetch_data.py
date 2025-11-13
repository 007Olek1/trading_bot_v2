"""Fetch historical OHLCV data for a watchlist and store locally."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import ccxt

from config.settings import settings

OUTPUT_DIR = Path("data/historical")
TIMEFRAMES = ("15m", "30m", "1h", "4h", "1d")
LIMIT = 1000


def fetch_symbol(exchange: ccxt.Exchange, symbol: str) -> dict:
    result = {}
    for timeframe in TIMEFRAMES:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=LIMIT)
        result[timeframe] = ohlcv
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.bybit(
        {
            "apiKey": settings.bybit_api_key,
            "secret": settings.bybit_api_secret,
            "enableRateLimit": True,
        }
    )
    exchange.options["defaultType"] = "linear"
    watchlist = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for symbol in watchlist:
        data = fetch_symbol(exchange, symbol)
        sanitized = symbol.replace("/", "")
        file_path = OUTPUT_DIR / f"{sanitized}_{timestamp}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        print(f"Saved {symbol} data to {file_path}")


if __name__ == "__main__":
    main()

