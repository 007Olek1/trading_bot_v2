#!/usr/bin/env python3
"""Простой тест подключения к API"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(".env")
load_dotenv("api.env", override=False)

from super_bot_v4_mtf import SuperBotV4MTF

async def test():
    print("📊 Инициализация бота...")
    bot = SuperBotV4MTF()
    await bot.initialize()
    
    print("\n🔍 Проверка API ключей:")
    has_key = bot.api_key is not None and len(bot.api_key) > 0
    has_secret = bot.api_secret is not None and len(bot.api_secret) > 0
    print(f"   API Key: {'✅ Есть' if has_key else '❌ НЕТ'}")
    print(f"   API Secret: {'✅ Есть' if has_secret else '❌ НЕТ'}")
    print(f"   Exchange: {'✅ Подключен' if bot.exchange else '❌ НЕТ'}")
    
    print("\n📈 Тест получения данных BTCUSDT 15m:")
    try:
        df = await bot._fetch_ohlcv("BTCUSDT", "15m", 10)
        if df is not None and not df.empty:
            print(f"   ✅ Получено {len(df)} свечей")
            print(f"   Последняя цена: {df['close'].iloc[-1]:.2f}")
        else:
            print("   ❌ Данные пустые")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n⏰ Тест MTF данных:")
    try:
        mtf_data = await bot._fetch_multi_timeframe_data("BTCUSDT")
        print(f"   Получено таймфреймов: {len(mtf_data)}")
        for tf in ['15m', '30m', '45m', '1h', '4h']:
            if tf in mtf_data:
                indicators_count = len(mtf_data[tf])
                print(f"   ✅ {tf}: {indicators_count} индикаторов")
            else:
                print(f"   ⚠️ {tf}: нет данных")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    if bot.exchange:
        await bot.exchange.close()

if __name__ == "__main__":
    asyncio.run(test())







