#!/usr/bin/env python3
"""
Synchronize open trades in TradeHistoryDB with Bybit closed PnL records.
Use this when positions were force-closed outside the bot and DB still marks them as open.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

from bybit_api import BybitAPI
from trade_history_db import TradeHistoryDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sync_open_trades")


def normalize_symbol(symbol: str) -> str:
    """
    Convert ccxt format (e.g. BTC/USDT:USDT) to Bybit raw (BTCUSDT) and vice versa.
    """
    if symbol.endswith("/USDT:USDT"):
        return symbol.replace("/USDT:USDT", "USDT")
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT:USDT"
    return symbol


def side_from_closed_entry(entry_side: str) -> str:
    if not entry_side:
        return ""
    entry_side = entry_side.lower()
    if entry_side in ("buy", "long"):
        return "long"
    if entry_side in ("sell", "short"):
        return "short"
    return entry_side


async def sync_open_trades():
    db = TradeHistoryDB()
    open_trades = db.get_open_trades()
    if not open_trades:
        logger.info("База данных не содержит открытых сделок.")
        return

    # Determine earliest open timestamp
    earliest_ts = min(t["timestamp"] for t in open_trades if t.get("timestamp"))
    buffer_sec = 12 * 3600  # 12 часов буфер
    start_ms = int((earliest_ts - buffer_sec) * 1000) if earliest_ts else int((time.time() - 72 * 3600) * 1000)

    logger.info("Загружаем закрытые сделки с Bybit начиная с %s",
                datetime.fromtimestamp(start_ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    api = BybitAPI()
    closed_entries = await api.fetch_closed_pnl(start_time_ms=start_ms, limit=200)
    await api.close()

    if not closed_entries:
        logger.warning("Bybit не вернул закрытых сделок за выбранный период.")
        return

    # Sort entries by update time ascending
    closed_entries.sort(key=lambda e: int(e.get("updatedTime", 0)))
    used = [False] * len(closed_entries)
    normalized_entries: List[Dict] = []
    for entry in closed_entries:
        normalized_entries.append(
            {
                "raw_symbol": entry.get("symbol"),
                "symbol": normalize_symbol(entry.get("symbol", "")),
                "side": side_from_closed_entry(entry.get("side", "")),
                "updated": int(entry.get("updatedTime", 0)),
                "exit_price": float(entry.get("avgExitPrice") or entry.get("execPrice") or entry.get("orderPrice") or 0),
                "pnl": float(entry.get("closedPnl", 0) or 0),
            }
        )

    synced = 0
    for trade in sorted(open_trades, key=lambda t: t.get("timestamp", 0)):
        trade_symbol = trade["symbol"]
        trade_side = trade["side"]
        trade_ts_ms = int((trade.get("timestamp") or 0) * 1000)

        match_idx = None
        for idx, entry in enumerate(normalized_entries):
            if used[idx]:
                continue
            if entry["symbol"] != trade_symbol:
                continue
            if entry["updated"] < trade_ts_ms:
                continue
            match_idx = idx
            break

        if match_idx is None:
            logger.warning("Не найден закрытый PnL для %s %s (открыта %s)",
                           trade_symbol, trade_side,
                           datetime.fromtimestamp(trade["timestamp"], timezone.utc).isoformat())
            continue

        entry = normalized_entries[match_idx]
        exit_price = entry["exit_price"] or trade["entry_price"]
        pnl_usd = entry["pnl"]

        db.close_trade(
            symbol=trade_symbol,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            reason="SYNCED_FROM_BYBIT",
            trailing_activated=False
        )
        used[match_idx] = True
        synced += 1
        logger.info("Сделка %s (%s) синхронизирована: exit %.6f | PnL %+0.4f",
                    trade_symbol, trade_side, exit_price, pnl_usd)

    logger.info("Синхронизация завершена. Обновлено %d из %d открытых сделок.", synced, len(open_trades))


if __name__ == "__main__":
    asyncio.run(sync_open_trades())
