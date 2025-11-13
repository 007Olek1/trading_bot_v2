#!/usr/bin/env python3
import sys
sys.path.insert(0, "/opt/bot")

import asyncio
import ccxt
import os
from pathlib import Path

from smart_coin_selector import SmartCoinSelector

# Загружаем .env
env_file = Path("/opt/bot/.env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value.strip().strip("\"\'")

async def check():
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
            
            selector = SmartCoinSelector()
            
            print("📊 ПРОВЕРКА КОЛИЧЕСТВА МОНЕТ:")
            print("="*60)
            
            conditions = ['normal', 'bullish', 'bearish', 'volatile']
            for condition in conditions:
                symbols = await selector.get_smart_symbols(exchange, condition)
                count = len(symbols) if symbols else 0
                print(f"{condition.upper()}: {count} монет")
            
            # Текущее условие рынка
            symbols_normal = await selector.get_smart_symbols(exchange, 'normal')
            print(f"\n✅ Текущий выбор (normal): {len(symbols_normal) if symbols_normal else 0} монет")
            if symbols_normal:
                print(f"   Примеры: {', '.join(symbols_normal[:10])}...")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

asyncio.run(check())








