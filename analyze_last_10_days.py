#!/usr/bin/env python3
"""
Анализ торговых сделок за последние 10 дней
"""
import sys
sys.path.insert(0, "/opt/bot")
from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta
import pytz

load_dotenv(Path("/opt/bot/.env"))
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
session = HTTP(api_key=api_key, api_secret=api_secret, testnet=False)

# Получаем текущий баланс
wallet = session.get_wallet_balance(accountType="UNIFIED")
usdt_balance = 0.0
if wallet and wallet.get("retCode") == 0:
    for coin in wallet.get("result", {}).get("list", [{}])[0].get("coin", []):
        if coin.get("coin") == "USDT":
            usdt_balance = float(coin.get("walletBalance", 0))
            break

print(f"Текущий баланс USDT: {usdt_balance:.2f}")

# Получаем историю закрытых позиций за последние 10 дней
warsaw_tz = pytz.timezone("Europe/Warsaw")
end_time = datetime.now(warsaw_tz)
start_time = end_time - timedelta(days=10)

# Bybit ограничивает запрос до 7 дней, поэтому разбиваем на части
trades = []
total_pnl = 0.0

# Разбиваем период на части по 7 дней
current_end = end_time
while current_end > start_time:
    current_start = max(current_end - timedelta(days=7), start_time)
    
    start_timestamp = int(current_start.timestamp() * 1000)
    end_timestamp = int(current_end.timestamp() * 1000)
    
    print(f"Запрашиваю данные: {current_start.strftime('%Y-%m-%d')} - {current_end.strftime('%Y-%m-%d')}")
    
    closed_pnl = session.get_closed_pnl(
        category="linear",
        startTime=start_timestamp,
        endTime=end_timestamp,
        limit=200
    )
    
    if closed_pnl and closed_pnl.get("retCode") == 0:
        result_list = closed_pnl.get("result", {}).get("list", [])
        for item in result_list:
            symbol = item.get("symbol", "N/A")
            side = item.get("side", "N/A")
            closed_pnl_val = float(item.get("closedPnl", 0))
            created_time = int(item.get("createdTime", 0))
            updated_time = int(item.get("updatedTime", 0))
            
            created_dt = datetime.fromtimestamp(created_time / 1000, tz=warsaw_tz)
            updated_dt = datetime.fromtimestamp(updated_time / 1000, tz=warsaw_tz)
            
            # Избегаем дубликатов
            trade_key = (symbol, side, created_time)
            if not any(t.get("key") == trade_key for t in trades):
                trades.append({
                    "key": trade_key,
                    "symbol": symbol,
                    "side": side,
                    "pnl": closed_pnl_val,
                    "created": created_dt,
                    "updated": updated_dt
                })
                total_pnl += closed_pnl_val
    
    current_end = current_start

separator = "=" * 80
print(f"\n{separator}")
print("АНАЛИЗ СДЕЛОК ЗА ПОСЛЕДНИЕ 10 ДНЕЙ")
print(separator)
print(f"Период: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}")
print(f"Всего сделок: {len(trades)}")
print(f"Общий PnL: {total_pnl:.2f} USDT")
initial_balance = usdt_balance - total_pnl
print(f"\nНачальный баланс (10 дней назад): {initial_balance:.2f} USDT")
print(f"Текущий баланс: {usdt_balance:.2f} USDT")
if initial_balance > 0:
    change_pct = (total_pnl / initial_balance) * 100
    print(f"Изменение баланса: {total_pnl:+.2f} USDT ({change_pct:+.2f}%)")

# Анализ по сделкам
if trades:
    profitable = [t for t in trades if t["pnl"] > 0]
    losing = [t for t in trades if t["pnl"] < 0]
    breakeven = [t for t in trades if t["pnl"] == 0]
    
    print(f"\n{separator}")
    print("СТАТИСТИКА:")
    print(separator)
    print(f"Прибыльных сделок: {len(profitable)} ({len(profitable)/len(trades)*100:.1f}%)")
    print(f"Убыточных сделок: {len(losing)} ({len(losing)/len(trades)*100:.1f}%)")
    print(f"Без изменения: {len(breakeven)}")
    
    if profitable:
        avg_profit = sum(t["pnl"] for t in profitable) / len(profitable)
        max_profit = max(t["pnl"] for t in profitable)
        total_profit = sum(t["pnl"] for t in profitable)
        best_trade = max(profitable, key=lambda x: x["pnl"])
        print(f"\nПрибыльные сделки:")
        print(f"  Средняя прибыль: {avg_profit:.2f} USDT")
        print(f"  Максимальная прибыль: {max_profit:.2f} USDT ({best_trade['symbol']})")
        print(f"  Общая прибыль: {total_profit:.2f} USDT")
    
    if losing:
        avg_loss = sum(t["pnl"] for t in losing) / len(losing)
        max_loss = min(t["pnl"] for t in losing)
        total_loss = sum(t["pnl"] for t in losing)
        worst_trade = min(losing, key=lambda x: x["pnl"])
        print(f"\nУбыточные сделки:")
        print(f"  Средний убыток: {avg_loss:.2f} USDT")
        print(f"  Максимальный убыток: {max_loss:.2f} USDT ({worst_trade['symbol']})")
        print(f"  Общий убыток: {total_loss:.2f} USDT")
    
    print(f"\n{separator}")
    print("ДЕТАЛИЗАЦИЯ ПО СДЕЛКАМ (все сделки):")
    print(separator)
    trades_sorted = sorted(trades, key=lambda x: x["created"], reverse=True)
    for i, trade in enumerate(trades_sorted, 1):
        pnl_sign = "+" if trade["pnl"] > 0 else ""
        print(f"{i:3d}. {trade['symbol']:15s} {trade['side']:6s} | PnL: {pnl_sign}{trade['pnl']:7.2f} USDT | {trade['created'].strftime('%Y-%m-%d %H:%M')}")

