#!/usr/bin/env python3
"""
Скрипт для установки SL/TP к существующим позициям на Bybit
Использует правильный API метод для установки SL/TP
"""

import asyncio
import sys
import os
import hmac
import hashlib
import time
import json
import requests

sys.path.insert(0, '/opt/bot')
os.chdir('/opt/bot')
from dotenv import load_dotenv
load_dotenv('/opt/bot/api.env')

# Параметры
POSITION_SIZE = 5.0
LEVERAGE = 5
MAX_STOP_LOSS_USD = 5.0
TP_PERCENT = 20.0

# API ключи
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
BASE_URL = "https://api.bybit.com"


def generate_signature(timestamp: str, recv_window: str, secret: str) -> str:
    """Генерирует подпись для Bybit API v5 (POST запрос)"""
    # Для POST запроса Bybit v5 подпись = HMAC-SHA256(timestamp + apiKey + recvWindow)
    # БЕЗ JSON body в подписи!
    param_str = f"{timestamp}{API_KEY}{recv_window}"
    signature = hmac.new(
        secret.encode('utf-8'),
        param_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature


async def set_sl_tp_bybit_direct(symbol: str, stop_loss: float = None, take_profit: float = None):
    """
    Прямая установка SL/TP через Bybit API v5
    Использует POST /v5/position/trading-stop
    """
    try:
        bybit_symbol = symbol.replace('/', '').replace(':USDT', '').replace(':USDT', '')
        
        # Параметры для тела запроса
        body_params = {
            'category': 'linear',
            'symbol': bybit_symbol,
            'positionIdx': 0  # 0 - one-way mode
        }
        
        # Форматируем значения (без лишних знаков)
        if stop_loss:
            body_params['stopLoss'] = f"{stop_loss:.8f}".rstrip('0').rstrip('.')
        
        if take_profit:
            body_params['takeProfit'] = f"{take_profit:.8f}".rstrip('0').rstrip('.')
        
        # Timestamp для подписи
        timestamp = str(int(time.time() * 1000))
        recv_window = '5000'
        
        # Генерируем подпись (только timestamp + apiKey + recvWindow, БЕЗ body!)
        signature = generate_signature(timestamp, recv_window, API_SECRET)
        
        # Отправляем запрос с правильными заголовками
        url = f"{BASE_URL}/v5/position/trading-stop"
        headers = {
            'Content-Type': 'application/json',
            'X-BAPI-API-KEY': API_KEY,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000'
        }
        
        response = requests.post(url, json=body_params, headers=headers, timeout=10)
        result = response.json()
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка установки SL/TP для {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return {'retCode': -1, 'retMsg': str(e)}


async def main():
    try:
        import ccxt.async_support as ccxt
        
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear', 'accountType': 'UNIFIED'}
        })
        
        print("🔍 Получаем открытые позиции...")
        print("=" * 70)
        
        positions = await exchange.fetch_positions(params={'category': 'linear'})
        open_positions = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
        
        print(f"📊 Найдено открытых позиций: {len(open_positions)}\n")
        
        if not open_positions:
            print("✅ Нет открытых позиций")
            await exchange.close()
            return
        
        for pos in open_positions:
            symbol = pos.get('symbol', '')
            side = pos.get('side', '').lower()
            entry_price = float(pos.get('entryPrice', 0) or pos.get('entry', 0))
            size = float(pos.get('contracts', 0) or pos.get('size', 0))
            
            print(f"\n📌 Обрабатываем {symbol} {side.upper()}:")
            print(f"   Вход: ${entry_price:.8f}")
            print(f"   Размер: {size}")
            
            # Рассчитываем SL и TP
            position_notional = POSITION_SIZE * LEVERAGE
            stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100
            
            if side in ['long', 'buy']:
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + TP_PERCENT / 100.0)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - TP_PERCENT / 100.0)
            
            print(f"   🛑 SL: ${stop_loss_price:.8f} (-${MAX_STOP_LOSS_USD:.2f} максимум)")
            print(f"   🎯 TP: ${tp_price:.8f} (+{TP_PERCENT:.0f}%)")
            
            # Устанавливаем SL/TP через прямой API
            print(f"\n   ⚙️ Устанавливаем Stop Loss и Take Profit...")
            result = await set_sl_tp_bybit_direct(symbol, stop_loss_price, tp_price)
            
            if result.get('retCode') == 0:
                print(f"      ✅ SL и TP успешно установлены!")
            else:
                ret_code = result.get('retCode', -1)
                ret_msg = result.get('retMsg', 'Unknown error')
                print(f"      ❌ Ошибка: retCode={ret_code}, msg={ret_msg}")
                
                # Пробуем установить отдельно
                if ret_code != 0:
                    # Только SL
                    result_sl = await set_sl_tp_bybit_direct(symbol, stop_loss_price, None)
                    if result_sl.get('retCode') == 0:
                        print(f"      ✅ SL установлен отдельно")
                    else:
                        print(f"      ⚠️ SL не установлен: {result_sl.get('retMsg', 'N/A')}")
                    
                    # Только TP
                    result_tp = await set_sl_tp_bybit_direct(symbol, None, tp_price)
                    if result_tp.get('retCode') == 0:
                        print(f"      ✅ TP установлен отдельно")
                    else:
                        print(f"      ⚠️ TP не установлен: {result_tp.get('retMsg', 'N/A')}")
        
        await exchange.close()
        print("\n" + "=" * 70)
        print("✅ Обработка завершена!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

