#!/usr/bin/env python3
"""
🔍 ТЕСТ ПОДКЛЮЧЕНИЯ К BYBIT API
Проверяет:
- Наличие API ключей в .env
- Валидность ключей
- Подключение к бирже
- Получение рыночных данных
- Получение баланса (если ключи валидны)
"""

import os
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import ccxt.async_support as ccxt
from datetime import datetime

# Цвета для вывода
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Загрузка переменных окружения
env_files = [
    Path(__file__).parent / "api.env",
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env"
]

loaded = False
for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file, override=False)
        if not loaded:
            print(f"{BLUE}✅ Переменные окружения загружены из {env_file}{RESET}")
        loaded = True

if not loaded:
    load_dotenv()
    if os.getenv('BYBIT_API_KEY'):
        print(f"{BLUE}✅ Переменные окружения загружены из системного .env{RESET}")
        loaded = True

async def test_bybit_connection():
    """Тестирование подключения к Bybit API"""
    print("\n" + "="*60)
    print(f"{BLUE}🔍 ПРОВЕРКА ПОДКЛЮЧЕНИЯ К BYBIT API{RESET}")
    print("="*60 + "\n")
    
    # Шаг 1: Проверка наличия API ключей
    print(f"{YELLOW}📋 Шаг 1: Проверка API ключей...{RESET}")
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key:
        print(f"{RED}❌ BYBIT_API_KEY не найден в переменных окружения{RESET}")
        print(f"{YELLOW}💡 Проверьте файлы: {', '.join([str(f) for f in env_files])}{RESET}")
        return False
    
    if not api_secret:
        print(f"{RED}❌ BYBIT_API_SECRET не найден в переменных окружения{RESET}")
        return False
    
    print(f"{GREEN}✅ API ключ найден: {api_key[:8]}...{RESET}")
    print(f"{GREEN}✅ API секрет найден: {'*' * len(api_secret[:8])}...{RESET}")
    
    # Шаг 2: Инициализация биржи
    print(f"\n{YELLOW}📋 Шаг 2: Инициализация биржи...{RESET}")
    try:
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'accountType': 'UNIFIED'  # Unified account для Bybit
            }
        })
        print(f"{GREEN}✅ Биржа инициализирована{RESET}")
    except Exception as e:
        print(f"{RED}❌ Ошибка инициализации: {e}{RESET}")
        return False
    
    # Шаг 3: Тест публичного API (получение тикера)
    print(f"\n{YELLOW}📋 Шаг 3: Тест публичного API (fetch_ticker)...{RESET}")
    try:
        ticker = await exchange.fetch_ticker('BTC/USDT:USDT')
        if ticker and ticker.get('last'):
            btc_price = ticker.get('last', 0)
            print(f"{GREEN}✅ Публичное API работает{RESET}")
            print(f"   BTC/USDT: ${btc_price:,.2f}")
        else:
            print(f"{RED}❌ Не удалось получить тикер{RESET}")
            await exchange.close()
            return False
    except Exception as e:
        print(f"{RED}❌ Ошибка получения тикера: {e}{RESET}")
        await exchange.close()
        return False
    
    # Шаг 4: Тест приватного API (получение баланса)
    print(f"\n{YELLOW}📋 Шаг 4: Тест приватного API (fetch_balance)...{RESET}")
    try:
        # Для Bybit Unified Account нужно указать accountType в params
        balance = await exchange.fetch_balance({'accountType': 'UNIFIED'})
        if balance:
            usdt_info = balance.get('USDT', {})
            usdt_balance = usdt_info.get('free', 0) if isinstance(usdt_info, dict) else 0
            usdt_total = usdt_info.get('total', 0) if isinstance(usdt_info, dict) else 0
            print(f"{GREEN}✅ Приватное API работает{RESET}")
            if usdt_balance is not None and usdt_total is not None:
                print(f"   USDT доступно: {usdt_balance:.2f}")
                print(f"   USDT всего: {usdt_total:.2f}")
            else:
                print(f"   Баланс: {balance}")
            
            # Выводим список валют с балансом > 0
            currencies_with_balance = []
            for currency, info in balance.items():
                if isinstance(info, dict):
                    total = info.get('total')
                    free = info.get('free', 0) or 0
                    used = info.get('used', 0) or 0
                    if (total and total > 0) or (free > 0) or (used > 0):
                        currencies_with_balance.append((currency, total or free or used))
            
            if currencies_with_balance:
                print(f"\n   💰 Валюты с балансом:")
                for currency, total in sorted(currencies_with_balance, key=lambda x: x[1], reverse=True):
                    print(f"      {currency}: {total:.4f}")
        else:
            print(f"{YELLOW}⚠️ Баланс пуст или недоступен{RESET}")
    except Exception as e:
        error_msg = str(e)
        if "API key is invalid" in error_msg or "10003" in error_msg:
            print(f"{RED}❌ API ключ невалиден (ошибка 10003){RESET}")
            print(f"{YELLOW}💡 Проверьте ключи в Bybit: https://www.bybit.com/app/user/api-management{RESET}")
        elif "IP" in error_msg or "whitelist" in error_msg.lower():
            print(f"{YELLOW}⚠️ IP адрес не в whitelist (но ключи работают){RESET}")
            print(f"   {error_msg}")
        else:
            print(f"{RED}❌ Ошибка получения баланса: {error_msg}{RESET}")
    
    # Шаг 5: Тест получения открытых позиций
    print(f"\n{YELLOW}📋 Шаг 5: Тест получения открытых позиций...{RESET}")
    try:
        positions = await exchange.fetch_positions(['BTC/USDT:USDT'])
        if positions:
            open_positions = [p for p in positions if p.get('contracts', 0) != 0]
            if open_positions:
                print(f"{GREEN}✅ Открытые позиции найдены: {len(open_positions)}{RESET}")
                for pos in open_positions[:5]:  # Показываем первые 5
                    symbol = pos.get('symbol', 'N/A')
                    size = pos.get('contracts', 0)
                    side = pos.get('side', 'N/A')
                    pnl = pos.get('unrealizedPnl', 0)
                    print(f"   {symbol}: {side} {abs(size)} контрактов, PnL: ${pnl:.2f}")
            else:
                print(f"{BLUE}ℹ️ Открытых позиций нет{RESET}")
        else:
            print(f"{BLUE}ℹ️ Позиции недоступны или пусты{RESET}")
    except Exception as e:
        error_msg = str(e)
        if "API key is invalid" in error_msg:
            print(f"{RED}❌ API ключ невалиден{RESET}")
        else:
            print(f"{YELLOW}⚠️ Ошибка получения позиций: {error_msg}{RESET}")
    
    # Шаг 6: Тест получения рынков
    print(f"\n{YELLOW}📋 Шаг 6: Тест получения списка рынков...{RESET}")
    try:
        markets = await exchange.load_markets(reload=False)
        if markets:
            symbols_count = len([s for s in markets.keys() if 'USDT' in s])
            print(f"{GREEN}✅ Рынки загружены: ~{symbols_count} USDT пар{RESET}")
            print(f"   Примеры: {', '.join(list(markets.keys())[:5])}")
        else:
            print(f"{YELLOW}⚠️ Рынки не загружены{RESET}")
    except Exception as e:
        error_msg = str(e)
        if "API key is invalid" in error_msg:
            print(f"{RED}❌ API ключ невалиден (не удалось загрузить markets){RESET}")
        else:
            print(f"{YELLOW}⚠️ Ошибка загрузки markets: {error_msg}{RESET}")
    
    # Закрываем соединение
    try:
        await exchange.close()
    except:
        pass
    
    # Итоговый результат
    print("\n" + "="*60)
    print(f"{GREEN}✅ ПРОВЕРКА ЗАВЕРШЕНА{RESET}")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_bybit_connection())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print(f"\n{RED}❌ Проверка прервана пользователем{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}❌ Критическая ошибка: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

