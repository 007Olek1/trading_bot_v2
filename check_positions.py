#!/usr/bin/env python3
import ccxt
import os
from pathlib import Path

env_file = Path("/opt/bot/.env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value.strip().strip("\"\'")

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

if api_key and api_secret:
    try:
        exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "linear"}
        })
        
        positions = exchange.fetch_positions()
        open_pos = [p for p in positions if (p.get("contracts", 0) or p.get("size", 0)) > 0]
        
        print(f"📊 ОТКРЫТЫХ ПОЗИЦИЙ: {len(open_pos)}")
        if open_pos:
            for p in open_pos:
                symbol = p.get("symbol")
                side = p.get("side", "unknown")
                size = p.get("size", 0) or p.get("contracts", 0)
                entry = p.get("entryPrice", 0) or p.get("averagePrice", 0)
                mark = p.get("markPrice", 0)
                pnl = p.get("unrealizedPnl", 0)
                pnl_pct = p.get("percentage", 0)
                print(f"  {symbol} {side.upper()}")
                print(f"    Размер: {size}")
                print(f"    Вход: ${entry:.6f} | Текущая: ${mark:.6f}")
                print(f"    PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            print("  (нет открытых позиций)")
        
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        print(f"\n💰 БАЛАНС: ${usdt.get('total', 0):.2f} (свободно: ${usdt.get('free', 0):.2f})")
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")
else:
    print("⚠️ API ключи не найдены")










