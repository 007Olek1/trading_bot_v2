#!/usr/bin/env python3
"""
Скрипт для проверки открытых позиций на Bybit
"""

import os
import sys
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# Загружаем переменные окружения
load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() == "true"

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    print("❌ BYBIT_API_KEY и BYBIT_API_SECRET должны быть установлены в .env")
    sys.exit(1)

# Создаём клиент
client = HTTP(
    testnet=USE_TESTNET,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
)

print("="*80)
print("🔍 ПРОВЕРКА ОТКРЫТЫХ ПОЗИЦИЙ НА BYBIT")
print("="*80)
print()

try:
    # Получаем все открытые позиции
    response = client.get_positions(
        category="linear",
        settleCoin="USDT"
    )
    
    if response['retCode'] != 0:
        print(f"❌ Ошибка получения позиций: {response.get('retMsg', 'Unknown')}")
        sys.exit(1)
    
    positions = response['result']['list']
    
    # Фильтруем только открытые позиции
    open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
    
    if not open_positions:
        print("📭 Нет открытых позиций")
        print()
    else:
        print(f"📊 Найдено открытых позиций: {len(open_positions)}")
        print()
        
        for pos in open_positions:
            symbol = pos['symbol']
            side = pos['side']
            size = float(pos['size'])
            entry_price = float(pos['avgPrice'])
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unrealisedPnl'])
            leverage = pos['leverage']
            
            # TP/SL информация
            take_profit = pos.get('takeProfit', '')
            stop_loss = pos.get('stopLoss', '')
            
            print(f"{'🟢' if side == 'Buy' else '🔴'} {symbol} - {side.upper()}")
            print(f"  📦 Размер: {size}")
            print(f"  💰 Цена входа: ${entry_price}")
            print(f"  📊 Текущая цена: ${mark_price}")
            print(f"  ⚡ Плечо: x{leverage}")
            print(f"  💵 Нереализованный PnL: ${unrealized_pnl:.2f}")
            
            if take_profit:
                print(f"  🎯 Take Profit: ${take_profit}")
            else:
                print(f"  🎯 Take Profit: ❌ НЕ УСТАНОВЛЕН")
            
            if stop_loss:
                print(f"  🛑 Stop Loss: ${stop_loss}")
            else:
                print(f"  🛑 Stop Loss: ❌ НЕ УСТАНОВЛЕН")
            
            print()
    
    # Проверяем активные ордера
    print("="*80)
    print("📋 ПРОВЕРКА АКТИВНЫХ ОРДЕРОВ")
    print("="*80)
    print()
    
    orders_response = client.get_open_orders(
        category="linear",
        settleCoin="USDT"
    )
    
    if orders_response['retCode'] == 0:
        orders = orders_response['result']['list']
        
        if not orders:
            print("📭 Нет активных ордеров")
        else:
            print(f"📊 Найдено активных ордеров: {len(orders)}")
            print()
            
            for order in orders:
                symbol = order['symbol']
                side = order['side']
                order_type = order['orderType']
                qty = order['qty']
                price = order.get('price', 'Market')
                
                print(f"  {symbol} - {side} {order_type}")
                print(f"    Количество: {qty}")
                print(f"    Цена: {price}")
                print()
    
    print("="*80)
    print("✅ Проверка завершена")
    print("="*80)

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
