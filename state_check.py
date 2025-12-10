#!/usr/bin/env python3
"""
Utility to inspect current bot state:
- Recent closed trades from TradeHistoryDB
- Daily stats snapshot
- Disco57 training metrics
"""
import os
from datetime import datetime, timezone

from bybit_api import BybitAPI  # ensures env/env vars loaded
from trade_history_db import TradeHistoryDB
from disco57_learner import Disco57Learner


def format_ts(timestamp: float) -> str:
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_recent_trades(db: TradeHistoryDB, hours: int = 24, limit: int = 10):
    print(f"\n📜 Последние сделки (<= {hours}ч, limit {limit}):")
    trades = db.get_recent_trades(hours=hours, limit=limit) or []
    if not trades:
        print("  Нет записей")
        return
    for trade in trades:
        status = trade.get("status", "unknown")
        pnl = trade.get("pnl_usd")
        pnl_val = pnl if pnl is not None else 0.0
        reason = trade.get("reason")
        opened = format_ts(trade.get("timestamp"))
        closed = format_ts(trade.get("closed_at"))
        print(
            f"  • {trade['symbol']} {trade['side'].upper()} | {status.upper()} | "
            f"PnL {pnl_val:+.4f} USD | reason={reason} | open={opened} | close={closed}"
        )


def print_daily_stats(db: TradeHistoryDB, days: int = 1):
    print(f"\n📊 Статистика за {days} день/дня:")
    stats = db.get_stats(hours=days * 24)
    if not stats:
        print("  Нет данных")
        return
    print(
        f"  Trades: {stats['total_trades']} (W {stats['winning_trades']} / "
        f"L {stats['losing_trades']}) | Win Rate {stats['win_rate']:.1f}% | "
        f"PnL {stats['total_pnl']:+.2f} USD | "
        f"Best {stats['best_trade']:+.2f} / Worst {stats['worst_trade']:+.2f}"
    )


def print_disco57_stats():
    try:
        disco = Disco57Learner()
    except Exception as e:
        print(f"\n🤖 Disco57: не удалось инициализировать ({e})")
        return
    stats = disco.get_stats()
    print("\n🤖 Disco57 состояние:")
    print(
        f"  Trades trained on: {stats['total_trades']} | "
        f"Win Rate: {stats['win_rate']:.1f}% | Total PnL: {stats['total_pnl']:+.2f}"
    )


def main():
    db = TradeHistoryDB()
    try:
        print_recent_trades(db, hours=48, limit=10)
        print_daily_stats(db, days=2)
    finally:
        db.close()
    print_disco57_stats()


if __name__ == "__main__":
    main()
