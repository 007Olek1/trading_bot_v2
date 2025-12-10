#!/usr/bin/env python3
"""
Debug selector per symbol to understand why no candidates are produced.

Usage:
    PYTHONPATH=src python scripts/debug_selector.py --symbols 20
"""

import argparse
import asyncio
import os
from typing import List

from goldtrigger.main import GoldTriggerSwingApp


async def run_debug(limit: int, verbose: bool):
    app = GoldTriggerSwingApp(mode=os.getenv("MODE", "live"))
    await app.selector.initialize()
    symbols: List[str] = app.selector.symbols[:limit] if limit > 0 else app.selector.symbols
    problems = {"atr": 0, "volume": 0, "ema": 0, "disco": 0, "other": 0, "ok": 0}

    for symbol in symbols:
        try:
            result = await app.selector._evaluate_symbol(symbol)  # type: ignore
            if result:
                problems["ok"] += 1
                if verbose:
                    print(f"OK {symbol} score={result.score:.2f} dir={result.direction} features={result.features}")
            else:
                problems["other"] += 1
                if verbose:
                    print(f"SKIP {symbol}: generic rejection (no details)")
        except Exception as exc:
            if verbose:
                print(f"ERROR {symbol}: {exc}")
            reason = "other"
            msg = str(exc).lower()
            if "atr" in msg:
                reason = "atr"
            elif "volume" in msg:
                reason = "volume"
            elif "ema" in msg:
                reason = "ema"
            elif "disco" in msg:
                reason = "disco"
            problems[reason] += 1

    print("Summary:", problems)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug selector per symbol")
    parser.add_argument("--symbols", type=int, default=40, help="Number of symbols to probe (0=all)")
    parser.add_argument("--verbose", action="store_true", help="Print per-symbol decisions")
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_debug(args.symbols, args.verbose))


if __name__ == "__main__":
    main()
