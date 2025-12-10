#!/usr/bin/env python3
"""Run StrategyScanner once and print live signals."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.bot_core.config import load_config
from src.bot_core.bybit_client import BybitClient
from src.bot_core.scanner import StrategyScanner
from SYMBOLS_145 import TRADING_SYMBOLS_145


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%" if abs(value) < 1 else f"{value:.2f}%"


def describe_signal(signal):
    return (
        f"{signal.symbol} {signal.side.upper()} @ {signal.entry_price:.6f} "
        f"ADX={signal.adx:.1f} ATR%={signal.atr_pct:.3f} "
        f"vol_ratio={signal.volume_ratio:.2f} momentum={signal.momentum:.4f} "
        f"spread={signal.spread_pct:.3f}%"
    )


def ensure_env():
    dotenv_path = os.environ.get("GOLDTRIGGER_DOTENV", ROOT / ".env")
    load_dotenv(dotenv_path)


async def main():
    ensure_env()
    config = load_config()
    client = BybitClient(config.bybit)
    scanner = StrategyScanner(client, config.trading, TRADING_SYMBOLS_145)
    print(f"Running scan across {len(TRADING_SYMBOLS_145)} symbols...")
    signals = await scanner.run_scan()
    if not signals:
        print("No signals matched filters right now.")
        return
    print(f"Signals found: {len(signals)}")
    for signal in signals:
        print(describe_signal(signal))


if __name__ == "__main__":
    asyncio.run(main())
