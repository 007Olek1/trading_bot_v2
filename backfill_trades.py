import os
import sqlite3
import time

import ccxt
from dotenv import load_dotenv

DB_PATH = 'data/trade_history.db'


def normalize_symbol(symbol: str) -> str:
    return symbol[:-4] + '/USDT:USDT' if symbol.endswith('USDT') else symbol


def close_trade(cur, row, exit_price, pnl_usd, reason, closed_at):
    trade_id, symbol, side, entry_price, quantity, opened_at = row
    if entry_price:
        if side == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
    else:
        pnl_pct = 0.0

    pnl_value = pnl_usd if pnl_usd is not None else pnl_pct * 25
    duration = closed_at - opened_at if opened_at and closed_at else 0

    cur.execute(
        """
        UPDATE trades
        SET exit_price = ?,
            pnl_usd = ?,
            pnl_pct = ?,
            reason = ?,
            trailing_activated = ?,
            duration_sec = ?,
            status = 'closed',
            closed_at = ?
        WHERE id = ?
        """,
        (exit_price, pnl_value, pnl_pct * 100, reason, 0, duration, closed_at, trade_id)
    )


def main():
    load_dotenv('.env')
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, symbol, side, entry_price, quantity, timestamp FROM trades WHERE status='open' ORDER BY timestamp ASC")
    open_map = {}
    for row in cur.fetchall():
        open_map.setdefault(row[1], []).append(row)

    api = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_API_SECRET'),
    })

    start_ms = int((time.time() - 24 * 3600) * 1000)
    response = api.private_get_v5_position_closed_pnl({
        'category': 'linear',
        'startTime': start_ms,
        'limit': 200,
    })
    entries = response.get('result', {}).get('list', [])

    filled = 0
    for entry in entries:
        symbol = normalize_symbol(entry['symbol'])
        if symbol not in open_map or not open_map[symbol]:
            continue

        row = open_map[symbol].pop(0)
        exit_price = float(entry.get('avgExitPrice') or entry.get('orderPrice') or row[3])
        pnl_value = float(entry.get('closedPnl', 0))
        closed_at = float(entry.get('updatedTime', 0)) / 1000
        close_trade(cur, row, exit_price, pnl_value, 'HIST_BACKFILL', closed_at)
        filled += 1

    conn.commit()
    conn.close()
    print(f"Backfilled {filled} trades")


if __name__ == '__main__':
    main()
