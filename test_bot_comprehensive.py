#!/usr/bin/env python3
"""
🧪 Комплексное тестирование бота перед ночным запуском
"""

import asyncio
import sys
from datetime import datetime
from bot_v2_exchange import ExchangeManager
from bot_v2_config import Config

async def test_all_systems():
    """Полное тестирование всех систем"""
    
    print("\n" + "="*70)
    print("🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ БОТА V2.0")
    print("="*70)
    
    results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    # ============================================
    # ТЕСТ 1: Конфигурация
    # ============================================
    print("\n📋 ТЕСТ 1: Проверка конфигурации...")
    try:
        assert Config.BYBIT_API_KEY, "API ключ пустой"
        assert Config.BYBIT_API_SECRET, "API секрет пустой"
        assert Config.TELEGRAM_BOT_TOKEN, "Telegram токен пустой"
        assert Config.TELEGRAM_CHAT_ID > 0, "Chat ID не установлен"
        
        print(f"   ✅ API ключи: Настроены")
        print(f"   ✅ Telegram: Настроен")
        print(f"   ✅ Размер позиции: ${Config.POSITION_SIZE_USD}")
        print(f"   ✅ Макс позиций: {Config.MAX_POSITIONS}")
        print(f"   ✅ Leverage: {Config.LEVERAGE}x")
        print(f"   ✅ Stop Loss: {Config.MAX_LOSS_PER_TRADE_PERCENT}%")
        
        results['passed'].append("Конфигурация")
    except AssertionError as e:
        print(f"   ❌ ОШИБКА: {e}")
        results['failed'].append(f"Конфигурация: {e}")
    
    # ============================================
    # ТЕСТ 2: Подключение к Bybit
    # ============================================
    print("\n🏦 ТЕСТ 2: Подключение к Bybit...")
    try:
        em = ExchangeManager()
        await em.connect()
        print("   ✅ Подключение успешно")
        
        # Проверка баланса
        balance = await em.get_balance()
        print(f"   ✅ Баланс: ${balance:.2f} USDT")
        
        if balance < 10:
            results['warnings'].append(f"Низкий баланс: ${balance:.2f}")
            print(f"   ⚠️ ПРЕДУПРЕЖДЕНИЕ: Баланс низкий!")
        
        results['passed'].append("Подключение к Bybit")
        
    except Exception as e:
        print(f"   ❌ ОШИБКА: {e}")
        results['failed'].append(f"Подключение: {e}")
        await em.disconnect()
        return results
    
    # ============================================
    # ТЕСТ 3: Открытые позиции
    # ============================================
    print("\n📊 ТЕСТ 3: Проверка открытых позиций...")
    try:
        positions = await em.fetch_positions()
        open_pos = [p for p in positions if p['contracts'] > 0]
        
        print(f"   ✅ Открытых позиций: {len(open_pos)}/{Config.MAX_POSITIONS}")
        
        total_pnl = 0
        for p in open_pos:
            symbol = p['symbol']
            side = p['side']
            pnl = p['unrealizedPnl']
            total_pnl += pnl
            
            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"   {emoji} {symbol} | {side.upper()} | PnL: ${pnl:.2f}")
            
            # Проверка наличия SL
            sl = p.get('info', {}).get('stopLoss')
            if sl:
                print(f"      ✅ Stop Loss: ${float(sl):.4f}")
            else:
                results['warnings'].append(f"{symbol}: нет Stop Loss!")
                print(f"      ⚠️ Stop Loss не установлен!")
        
        print(f"   💵 TOTAL PnL: ${total_pnl:.2f}")
        
        results['passed'].append("Позиции и SL")
        
    except Exception as e:
        print(f"   ❌ ОШИБКА: {e}")
        results['failed'].append(f"Позиции: {e}")
    
    # ============================================
    # ТЕСТ 4: Анализ символов (быстрый тест)
    # ============================================
    print("\n🔍 ТЕСТ 4: Проверка анализа символов...")
    try:
        # Тест на одном символе
        test_symbol = 'BTC/USDT:USDT'
        candles = await em.fetch_ohlcv(test_symbol, '1m', limit=100)
        
        if len(candles) >= 100:
            print(f"   ✅ Загрузка свечей: OK ({len(candles)} свечей)")
            
            # Проверка данных
            last_candle = candles[-1]
            print(f"   ✅ Последняя цена {test_symbol}: ${last_candle['close']:.2f}")
            
            results['passed'].append("Анализ символов")
        else:
            results['warnings'].append("Недостаточно данных для анализа")
            
    except Exception as e:
        print(f"   ❌ ОШИБКА: {e}")
        results['failed'].append(f"Анализ: {e}")
    
    # ============================================
    # ТЕСТ 5: Проверка лимитов
    # ============================================
    print("\n🛡️ ТЕСТ 5: Проверка лимитов безопасности...")
    
    # Проверка что не превышаем лимиты
    if len(open_pos) <= Config.MAX_POSITIONS:
        print(f"   ✅ Лимит позиций: {len(open_pos)}/{Config.MAX_POSITIONS}")
        results['passed'].append("Лимиты позиций")
    else:
        print(f"   ❌ Превышен лимит позиций!")
        results['failed'].append("Лимит позиций превышен")
    
    # Проверка размера позиций
    for p in open_pos:
        size_usd = p.get('notional', 0)
        if size_usd > Config.POSITION_SIZE_USD * 1.2:  # +20% допуск
            results['warnings'].append(f"{p['symbol']}: размер ${size_usd:.2f} > ${Config.POSITION_SIZE_USD}")
    
    # ============================================
    # ТЕСТ 6: Trailing Stop проверка
    # ============================================
    print("\n🎯 ТЕСТ 6: Проверка Trailing Stop...")
    
    trailing_working = False
    for p in open_pos:
        entry = p['entryPrice']
        sl = p.get('info', {}).get('stopLoss')
        
        if sl:
            sl_float = float(sl)
            side = p['side']
            
            # Проверка что SL правильный для направления
            if side.lower() in ['buy', 'long']:
                if sl_float < entry:
                    print(f"   ✅ {p['symbol']} LONG: SL ${sl_float:.4f} < Entry ${entry:.4f}")
                    trailing_working = True
                else:
                    results['warnings'].append(f"{p['symbol']}: SL выше Entry (LONG)")
            else:  # SHORT
                if sl_float > entry:
                    print(f"   ✅ {p['symbol']} SHORT: SL ${sl_float:.4f} > Entry ${entry:.4f}")
                    trailing_working = True
                else:
                    results['warnings'].append(f"{p['symbol']}: SL ниже Entry (SHORT)")
    
    if trailing_working:
        results['passed'].append("Trailing Stop")
    
    # ============================================
    # Отключение
    # ============================================
    await em.disconnect()
    print("\n✅ Отключение от биржи")
    
    # ============================================
    # ИТОГИ
    # ============================================
    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*70)
    
    print(f"\n✅ ПРОЙДЕНО ({len(results['passed'])}):")
    for test in results['passed']:
        print(f"   ✅ {test}")
    
    if results['warnings']:
        print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ ({len(results['warnings'])}):")
        for warn in results['warnings']:
            print(f"   ⚠️ {warn}")
    
    if results['failed']:
        print(f"\n❌ ПРОВАЛЕНО ({len(results['failed'])}):")
        for fail in results['failed']:
            print(f"   ❌ {fail}")
    
    # Финальная оценка
    print("\n" + "="*70)
    
    if not results['failed'] and len(results['warnings']) < 3:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! БОТ ГОТОВ К НОЧНОЙ ТОРГОВЛЕ!")
        print("="*70)
        return True
    elif not results['failed']:
        print("⚠️ ЕСТЬ ПРЕДУПРЕЖДЕНИЯ, НО БОТ МОЖЕТ РАБОТАТЬ")
        print("="*70)
        return True
    else:
        print("❌ КРИТИЧЕСКИЕ ОШИБКИ! НЕ РЕКОМЕНДУЕТСЯ ЗАПУСК!")
        print("="*70)
        return False


if __name__ == "__main__":
    result = asyncio.run(test_all_systems())
    sys.exit(0 if result else 1)


