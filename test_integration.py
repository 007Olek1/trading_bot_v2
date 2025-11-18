#!/usr/bin/env python3
"""
🔗 ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ
Проверка работы с реальным API Bybit
"""

import os
import sys
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

def test_api_connection():
    """Тест подключения к API"""
    print("="*80)
    print("🧪 ТЕСТ #1: Подключение к Bybit API")
    print("="*80)
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ API ключи не найдены в .env")
        return False
    
    try:
        client = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Проверка баланса
        response = client.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"
        )
        
        if response['retCode'] == 0:
            print("✅ Подключение к API успешно")
            
            if response['result']['list']:
                balance = float(response['result']['list'][0]['coin'][0]['walletBalance'])
                print(f"💰 Баланс: ${balance:.2f} USDT")
            
            return True
        else:
            print(f"❌ Ошибка API: {response.get('retMsg')}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False


def test_symbol_validation():
    """Тест валидации символов"""
    print("\n" + "="*80)
    print("🧪 ТЕСТ #2: Валидация символов")
    print("="*80)
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    client = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
    
    # Тестовые символы
    test_symbols = [
        "BTCUSDT",   # Должен работать
        "ETHUSDT",   # Должен работать
        "ADAUSDT",   # Должен работать
        "XXXUSDT",   # Не должен существовать
    ]
    
    print("\n📊 Проверка символов:")
    valid_count = 0
    
    for symbol in test_symbols:
        try:
            response = client.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                info = response['result']['list'][0]
                status = info.get('status', 'Unknown')
                
                if status == 'Trading':
                    print(f"  ✅ {symbol}: {status}")
                    valid_count += 1
                else:
                    print(f"  ⚠️  {symbol}: {status} (не торгуется)")
            else:
                print(f"  ❌ {symbol}: Не найден")
                
        except Exception as e:
            print(f"  ❌ {symbol}: Ошибка - {str(e)[:50]}")
    
    print(f"\n✅ Валидных символов: {valid_count}/{len(test_symbols)}")
    return valid_count > 0


def test_market_scanner():
    """Тест динамического сканера рынка"""
    print("\n" + "="*80)
    print("🧪 ТЕСТ #3: Динамический сканер рынка")
    print("="*80)
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    client = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
    
    try:
        # Получаем тикеры
        response = client.get_tickers(category="linear")
        
        if response['retCode'] != 0:
            print(f"❌ Ошибка получения тикеров: {response.get('retMsg')}")
            return False
        
        tickers = response['result']['list']
        
        # Фильтруем USDT пары с объёмом > $1M
        min_volume = 1000000
        liquid_pairs = []
        
        for ticker in tickers:
            symbol = ticker['symbol']
            
            if not symbol.endswith('USDT'):
                continue
            
            volume_24h = float(ticker.get('turnover24h', 0))
            if volume_24h < min_volume:
                continue
            
            last_price = float(ticker.get('lastPrice', 0))
            if last_price <= 0:
                continue
            
            liquid_pairs.append({
                'symbol': symbol,
                'volume_24h': volume_24h,
                'last_price': last_price
            })
        
        # Сортируем по объёму
        liquid_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        print(f"\n📊 Найдено {len(liquid_pairs)} ликвидных пар (объём > ${min_volume:,})")
        print(f"\n🏆 Топ-10 по ликвидности:")
        
        for i, pair in enumerate(liquid_pairs[:10], 1):
            print(f"  {i}. {pair['symbol']}: ${pair['volume_24h']:,.0f} (цена: ${pair['last_price']:.4f})")
        
        print(f"\n✅ Сканер работает корректно")
        return len(liquid_pairs) > 0
        
    except Exception as e:
        print(f"❌ Ошибка сканера: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_order_parameters():
    """Тест параметров ордера"""
    print("\n" + "="*80)
    print("🧪 ТЕСТ #4: Параметры ордера")
    print("="*80)
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    client = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
    
    test_symbol = "ADAUSDT"
    
    try:
        response = client.get_instruments_info(
            category="linear",
            symbol=test_symbol
        )
        
        if response['retCode'] != 0:
            print(f"❌ Ошибка: {response.get('retMsg')}")
            return False
        
        info = response['result']['list'][0]
        
        # Параметры лота
        min_qty = float(info['lotSizeFilter']['minOrderQty'])
        max_qty = float(info['lotSizeFilter']['maxOrderQty'])
        qty_step = float(info['lotSizeFilter']['qtyStep'])
        
        # Параметры цены
        min_price = float(info['priceFilter']['minPrice'])
        max_price = float(info['priceFilter']['maxPrice'])
        tick_size = float(info['priceFilter']['tickSize'])
        
        print(f"\n📊 Параметры для {test_symbol}:")
        print(f"\n📦 Количество:")
        print(f"  Минимум: {min_qty}")
        print(f"  Максимум: {max_qty}")
        print(f"  Шаг: {qty_step}")
        
        print(f"\n💰 Цена:")
        print(f"  Минимум: ${min_price}")
        print(f"  Максимум: ${max_price}")
        print(f"  Тик: ${tick_size}")
        
        # Тест расчёта количества
        position_size_usd = 1.0
        leverage = 10
        current_price = 0.49
        
        quantity = (position_size_usd * leverage) / current_price
        quantity = round(quantity / qty_step) * qty_step
        
        print(f"\n🧮 Расчёт позиции:")
        print(f"  Размер: ${position_size_usd} x{leverage}")
        print(f"  Цена: ${current_price}")
        print(f"  Количество: {quantity} (округлено до шага)")
        
        # Проверка
        if quantity < min_qty:
            print(f"  ⚠️  Количество меньше минимума!")
        elif quantity > max_qty:
            print(f"  ⚠️  Количество больше максимума!")
        else:
            print(f"  ✅ Количество в допустимых пределах")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def run_integration_tests():
    """Запуск всех интеграционных тестов"""
    print("\n🔗 ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ")
    print("="*80)
    
    results = []
    
    results.append(("API Connection", test_api_connection()))
    results.append(("Symbol Validation", test_symbol_validation()))
    results.append(("Market Scanner", test_market_scanner()))
    results.append(("Order Parameters", test_order_parameters()))
    
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✅ ВСЕ ИНТЕГРАЦИОННЫЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ!")
    
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
