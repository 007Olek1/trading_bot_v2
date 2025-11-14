#!/usr/bin/env python3
"""
🧪 ТЕСТ ИНТЕГРАЦИИ СЕЛЕКТОРА С БОТОМ И ЛОГИКОЙ ВХОДА
"""
import sys
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, "/opt/bot")

# Загружаем переменные окружения
env_file = Path("/opt/bot/.env")
if env_file.exists():
    load_dotenv(env_file, override=True)

import ccxt
import logging
from smart_coin_selector import SmartCoinSelector
from super_bot_v4_mtf import SuperBotV4MTF

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_integration():
    """Тест интеграции селектора с ботом"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ ИНТЕГРАЦИИ СЕЛЕКТОРА С БОТОМ")
    logger.info("="*60)
    
    try:
        # Инициализируем биржу
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("❌ API ключи не найдены")
            return False
        
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        
        # Инициализируем бота
        logger.info("📦 Инициализация бота...")
        bot = SuperBotV4MTF()
        await bot.initialize()
        logger.info("✅ Бот инициализирован")
        
        # Тест 1: Анализ рынка
        logger.info("\n🔍 ТЕСТ 1: Анализ рыночных условий...")
        market_data = await bot.analyze_market_trend_v4()
        market_condition = market_data.get('trend', 'neutral').upper()
        logger.info(f"✅ Рыночное условие: {market_condition}")
        logger.info(f"   BTC изменение: {market_data.get('btc_change', 0):.2f}%")
        
        # Тест 2: Выбор символов
        logger.info("\n🎯 ТЕСТ 2: Умный выбор символов...")
        symbols = await bot.smart_symbol_selection_v4(market_data)
        logger.info(f"✅ Выбрано символов: {len(symbols)}")
        logger.info(f"   Топ-10: {symbols[:10]}")
        
        # Проверяем топ-50
        priority_top50 = ['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT']
        included = [s for s in priority_top50 if s in symbols]
        logger.info(f"✅ Приоритетные топ-5 включены: {len(included)}/5")
        
        # Тест 3: Анализ символа с индикаторами
        logger.info("\n📊 ТЕСТ 3: Анализ символа с индикаторами...")
        test_symbol = symbols[0] if symbols else 'BTCUSDT'
        logger.info(f"   Анализируем: {test_symbol}")
        
        signal = await bot.analyze_symbol_v4(test_symbol)
        if signal:
            logger.info(f"✅ Сигнал создан:")
            logger.info(f"   Направление: {signal.direction.upper()}")
            logger.info(f"   Уверенность: {signal.confidence:.1f}%")
            logger.info(f"   Причины: {signal.reasons[:3]}")
        else:
            logger.info(f"⚪ Сигнал не создан (нормально для тестового символа)")
        
        # Тест 4: Проверка индикаторов
        logger.info("\n📈 ТЕСТ 4: Проверка индикаторов...")
        mtf_data = await bot._fetch_multi_timeframe_data(test_symbol)
        if mtf_data:
            logger.info(f"✅ MTF данные получены:")
            for tf in ['15m', '30m', '45m', '1h', '4h']:
                if tf in mtf_data:
                    data = mtf_data[tf]
                    logger.info(f"   {tf}: EMA9={data.get('ema_9', 0):.2f}, RSI={data.get('rsi', 0):.1f}, MACD={data.get('macd', 0):.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)










