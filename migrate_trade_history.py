#!/usr/bin/env python3
"""
Миграция истории сделок из SQLite в PostgreSQL.

Запускать на сервере после настройки переменных окружения:
TRADE_DB_PROVIDER=postgres, PG_* и SQLITE_DB_PATH.
"""

import os
import sqlite3
from pathlib import Path

import psycopg
from dotenv import load_dotenv


def fetch_sqlite_rows(db_path: str, table: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table}")
    rows = cursor.fetchall()
    conn.close()
    return rows


def migrate_trades(pg_cursor, trades):
    inserted = 0
    for row in trades:
        pg_cursor.execute(
            """
            INSERT INTO trades (
                id, timestamp, symbol, side, entry_price, exit_price,
                quantity, pnl_usd, pnl_pct, reason, signal_strength,
                disco_confidence, trailing_activated, duration_sec,
                status, created_at, closed_at
            )
            VALUES (
                %(id)s, %(timestamp)s, %(symbol)s, %(side)s, %(entry_price)s, %(exit_price)s,
                %(quantity)s, %(pnl_usd)s, %(pnl_pct)s, %(reason)s, %(signal_strength)s,
                %(disco_confidence)s, %(trailing_activated)s, %(duration_sec)s,
                %(status)s, %(created_at)s, %(closed_at)s
            )
            ON CONFLICT (id) DO NOTHING
            """,
            dict(row)
        )
        if pg_cursor.rowcount:
            inserted += 1
    return inserted


def migrate_daily_stats(pg_cursor, stats_rows):
    inserted = 0
    for row in stats_rows:
        pg_cursor.execute(
            """
            INSERT INTO daily_stats (
                id, date, total_trades, winning_trades, losing_trades,
                total_pnl, max_drawdown, best_trade, worst_trade, avg_duration_sec
            )
            VALUES (
                %(id)s, %(date)s, %(total_trades)s, %(winning_trades)s, %(losing_trades)s,
                %(total_pnl)s, %(max_drawdown)s, %(best_trade)s, %(worst_trade)s, %(avg_duration_sec)s
            )
            ON CONFLICT (id) DO NOTHING
            """,
            dict(row)
        )
        if pg_cursor.rowcount:
            inserted += 1
    return inserted


def reset_sequences(pg_cursor):
    pg_cursor.execute("SELECT setval('trades_id_seq', (SELECT COALESCE(MAX(id), 0) + 1 FROM trades))")
    pg_cursor.execute("SELECT setval('daily_stats_id_seq', (SELECT COALESCE(MAX(id), 0) + 1 FROM daily_stats))")


def main():
    load_dotenv()
    sqlite_path = os.getenv('SQLITE_DB_PATH', '/opt/bot/data/trade_history.db')
    if not sqlite_path or not Path(sqlite_path).exists():
        raise SystemExit(f"SQLite DB not found at {sqlite_path}")

    pg_params = {
        'host': os.getenv('PG_HOST', '127.0.0.1'),
        'port': int(os.getenv('PG_PORT', '5432')),
        'dbname': os.getenv('PG_NAME'),
        'user': os.getenv('PG_USER'),
        'password': os.getenv('PG_PASSWORD'),
        'sslmode': os.getenv('PG_SSLMODE', 'disable'),
    }
    if not all([pg_params['dbname'], pg_params['user'], pg_params['password']]):
        raise SystemExit("PG_NAME, PG_USER, PG_PASSWORD must be set in environment")

    trades = fetch_sqlite_rows(sqlite_path, 'trades')
    stats_rows = fetch_sqlite_rows(sqlite_path, 'daily_stats')

    with psycopg.connect(**pg_params) as pg_conn:
        with pg_conn.cursor() as cursor:
            inserted_trades = migrate_trades(cursor, trades)
            inserted_stats = migrate_daily_stats(cursor, stats_rows)
            reset_sequences(cursor)
        pg_conn.commit()

    print(f"Trades inserted: {inserted_trades}")
    print(f"Daily stats inserted: {inserted_stats}")


if __name__ == '__main__':
    main()
