#!/usr/bin/env python3
"""
Restore missing BCH trade entry in TradeHistoryDB.
"""

from trade_history_db import TradeHistoryDB


def main():
    symbol = "BCH/USDT:USDT"
    db = TradeHistoryDB()
    cur = db._conn.cursor()
    cur.execute(
        db._prepare_query(
            "SELECT COUNT(*) FROM trades WHERE symbol = %s AND status = 'open'"
        ),
        (symbol,),
    )
    count = cur.fetchone()[0] or 0
    print(f"Existing open trades: {count}")
    if count == 0:
        trade_id = db.add_trade_open(
            symbol=symbol,
            side="long",
            entry_price=582.4,
            quantity=0.02,
            signal_strength=0,
            disco_confidence=0,
        )
        print(f"Inserted trade ID {trade_id}")
    else:
        print("Skipped insert")


if __name__ == "__main__":
    main()
