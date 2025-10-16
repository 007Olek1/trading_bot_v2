#!/usr/bin/env python3
"""Триггер анализа вручную для тестирования"""

import asyncio
import sys
sys.path.insert(0, "/root/trading_bot_v2")

from trading_bot_v3_main import TradingBotV2

async def main():
    print("🔄 Запускаю торговый цикл вручную...")
    bot = TradingBotV2()
    
    # Подключаемся
    await bot._startup()
    
    # Запускаем ОДИН торговый цикл
    await bot.trading_loop()
    
    print("✅ Цикл завершен")

if __name__ == "__main__":
    asyncio.run(main())



