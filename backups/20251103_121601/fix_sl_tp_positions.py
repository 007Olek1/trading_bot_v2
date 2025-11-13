#!/usr/bin/env python3
"""
Рабочий скрипт для установки SL/TP к существующим позициям на Bybit
Использует правильную подпись для Bybit API v5
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

API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
BASE_URL = "https://api.bybit.com"


def sign_request(timestamp: str, recv_window: str, secret: str) -> str:
    """
    Генерирует подпись для Bybit API v5 POST запроса
    Для Bybit v5: подпись = HMAC-SHA256(timestamp + apiKey + recvWindow)
    """
    # Для Bybit v5 POST запросов подпись создается из:
    # timestamp + apiKey + recvWindow (БЕЗ JSON body!)
    param_str = f"{timestamp}{API_KEY}{recv_window}"
    signature = hmac.new(
        secret.encode('utf-8'),
        param_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature


def set_trading_stop(symbol: str, stop_loss: float = None, take_profit: float = None):
    """
    Устанавливает SL/TP для позиции через Bybit API v5
    """
    try:
        bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
        
        # Тело запроса
        body = {
            'category': 'linear',
            'symbol': bybit_symbol,
            'positionIdx': 0
        }
        
        if stop_loss:
            body['stopLoss'] = f"{stop_loss:.8f}".rstrip('0').rstrip('.')
        if take_profit:
            body['takeProfit'] = f"{take_profit:.8f}".rstrip('0').rstrip('.')
        
        # Timestamp и recvWindow для подписи
        timestamp = str(int(time.time() * 1000))
        recv_window = '5000'
        
        # Генерируем подпись (ТОЛЬКО timestamp + apiKey + recvWindow, БЕЗ JSON body!)
        signature = sign_request(timestamp, recv_window, API_SECRET)
        
        # Заголовки
        headers = {
            'X-BAPI-API-KEY': API_KEY,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000',
            'Content-Type': 'application/json'
        }
        
        # Отправляем запрос
        url = f"{BASE_URL}/v5/position/trading-stop"
        response = requests.post(url, json=body, headers=headers, timeout=10)
        result = response.json()
        
        return result
        
    except Exception as e:
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
            
            print(f"\n📌 Обрабатываем {symbol} {side.upper()}:")
            print(f"   Вход: ${entry_price:.8f}")
            
            # Рассчитываем SL и TP
            position_notional = POSITION_SIZE * LEVERAGE
            stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100
            
            if side in ['long', 'buy']:
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + TP_PERCENT / 100.0)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - TP_PERCENT / 100.0)
            
            print(f"   🛑 SL: ${stop_loss_price:.8f} (-${MAX_STOP_LOSS_USD:.2f})")
            print(f"   🎯 TP: ${tp_price:.8f} (+{TP_PERCENT:.0f}%)")
            
            # Устанавливаем SL/TP
            print(f"\n   ⚙️ Устанавливаем SL и TP...")
            result = set_trading_stop(symbol, stop_loss_price, tp_price)
            
            if result.get('retCode') == 0:
                print(f"      ✅ SL и TP успешно установлены!")
            else:
                ret_code = result.get('retCode', -1)
                ret_msg = result.get('retMsg', 'Unknown')
                print(f"      ⚠️ Ошибка: retCode={ret_code}, msg={ret_msg}")
                
                # Пробуем отдельно SL
                if ret_code != 0:
                    print(f"      Пробуем установить только SL...")
                    result_sl = set_trading_stop(symbol, stop_loss_price, None)
                    if result_sl.get('retCode') == 0:
                        print(f"      ✅ SL установлен!")
                    else:
                        print(f"      ⚠️ SL: {result_sl.get('retMsg', 'N/A')}")
                    
                    # Пробуем отдельно TP
                    print(f"      Пробуем установить только TP...")
                    result_tp = set_trading_stop(symbol, None, tp_price)
                    if result_tp.get('retCode') == 0:
                        print(f"      ✅ TP установлен!")
                    else:
                        print(f"      ⚠️ TP: {result_tp.get('retMsg', 'N/A')}")
        
        await exchange.close()
        
        print("\n" + "=" * 70)
        print("✅ Обработка завершена!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

