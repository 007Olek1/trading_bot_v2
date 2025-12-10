#!/usr/bin/env python3
"""
Inspect trades stored in TradeHistoryDB.
Usage examples:
    python3 scripts/inspect_trades.py --symbol BCH/USDT:USDT --limit 5
    python3 scripts/inspect_trades.py --status open
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goldtrigger.trade_history_db import TradeHistoryDB


def fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Inspect trades table")
    parser.add_argument("--symbol", help="Symbol to filter, e.g. BCH/USDT:USDT")
    parser.add_argument("--status", choices=["open", "closed"], help="Filter by trade status")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to show")
    args = parser.parse_args()

    db = TradeHistoryDB()
    cur = db._conn.cursor()

    base_query = (
        "SELECT id, symbol, side, entry_price, exit_price, pnl_usd, status, timestamp, closed_at "
        "FROM trades"
    )
    clauses = []
    params: list = []

    if args.symbol:
        clauses.append("symbol = %s")
        params.append(args.symbol)
    if args.status:
        clauses.append("status = %s")
        params.append(args.status)

    if clauses:
        base_query += " WHERE " + " AND ".join(clauses)

    base_query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(args.limit)

    cur.execute(db._prepare_query(base_query), tuple(params))
    rows = cur.fetchall()

    print(f"Rows fetched: {len(rows)}")
    for row in rows:
        row_id, symbol, side, entry, exit_price, pnl_usd, status, ts, closed_at = row
        print(
            f"[{row_id}] {symbol} {side} status={status} "
            f"entry={entry:.6f} exit={exit_price or 0:.6f} pnl={pnl_usd or 0:.2f} "
            f"opened={fmt_ts(ts)} closed={fmt_ts(closed_at)}"
        )


if __name__ == "__main__":
    main()
