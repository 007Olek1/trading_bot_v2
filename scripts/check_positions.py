#!/usr/bin/env python3
"""
Utility script to print current Bybit positions with SL/TP info.
"""

import asyncio
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from goldtrigger.bybit_api import BybitAPI


async def main():
    api = BybitAPI()
    try:
        positions = await api.get_positions_with_sl_tp()
    finally:
        await api.close()

    rows = []
    for pos in positions:
        size = float(pos.get("size") or pos.get("contracts") or 0)
        if size <= 0:
            continue
        rows.append(
            {
                "symbol": pos.get("symbol"),
                "side": pos.get("side"),
                "size": size,
                "avgPrice": pos.get("avgPrice"),
                "stopLoss": pos.get("stopLoss"),
                "takeProfit": pos.get("takeProfit"),
                "lastPrice": pos.get("lastPrice"),
            }
        )

    print(f"Total open positions: {len(rows)}")
    for row in rows:
        pprint(row)


if __name__ == "__main__":
    asyncio.run(main())
