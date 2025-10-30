#!/usr/bin/env python3
"""
Установка SL/TP используя ccxt напрямую через правильный метод
"""

import asyncio
import sys
import os

sys.path.insert(0, '/opt/bot')
os.chdir('/opt/bot')
from dotenv import load_dotenv
load_dotenv('/opt/bot/api.env')

import ccxt.async_support as ccxt

POSITION_SIZE = 5.0
LEVERAGE = 5
MAX_STOP_LOSS_USD = 5.0
TP_PERCENT = 20.0


async def set_sl_tp_via_request(exchange, symbol: str, stop_loss: float = None, take_profit: float = None):
    """
    Устанавливает SL/TP через ccxt request метод
    """
    try:
        bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
        
        params = {
            'category': 'linear',
            'symbol': bybit_symbol,
            'positionIdx': 0
        }
        
        if stop_loss:
            params['stopLoss'] = str(stop_loss)
        if take_profit:
            params['takeProfit'] = str(take_profit)
        
        # Используем встроенный метод ccxt для правильной подписи
        # Метод: exchange.market().request() или exchange.request()
        try:
            response = await exchange.request('v5', 'position', 'trading-stop', 'post', params)
            return response
        except Exception as e1:
            # Пробуем другой вариант
            try:
                response = await exchange.request('private', 'post', '/v5/position/trading-stop', params)
                return response
            except Exception as e2:
                # Пробуем через прямой доступ к методу
                try:
                    if hasattr(exchange, 'request'):
                        response = await exchange.request('position', 'trading-stop', params, None, 'post')
                        return response
                except Exception as e3:
                    raise e3
        
    except Exception as e:
        return {'retCode': -1, 'retMsg': str(e)}


async def main():
    try:
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear', 'accountType': 'UNIFIED'}
        })
        
        print("🔍 Получаем открытые позиции...")
        positions = await exchange.fetch_positions(params={'category': 'linear'})
        open_positions = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
        
        print(f"📊 Найдено позиций: {len(open_positions)}\n")
        
        for pos in open_positions:
            symbol = pos.get('symbol', '')
            side = pos.get('side', '').lower()
            entry_price = float(pos.get('entryPrice', 0))
            
            print(f"📌 {symbol} {side.upper()}: вход={entry_price:.8f}")
            
            # Рассчитываем SL и TP
            position_notional = POSITION_SIZE * LEVERAGE
            stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100
            
            if side in ['long', 'buy']:
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + TP_PERCENT / 100.0)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - TP_PERCENT / 100.0)
            
            print(f"   🛑 SL: {stop_loss_price:.8f}")
            print(f"   🎯 TP: {tp_price:.8f}")
            
            # Устанавливаем через ccxt request
            print(f"   ⚙️ Устанавливаем...")
            result = await set_sl_tp_via_request(exchange, symbol, stop_loss_price, tp_price)
            
            if result and result.get('retCode') == 0:
                print(f"      ✅ SL и TP установлены!")
            else:
                ret_code = result.get('retCode', -1) if result else -1
                ret_msg = result.get('retMsg', str(result)) if result else 'Unknown'
                print(f"      ⚠️ Ошибка: retCode={ret_code}, msg={ret_msg}")
        
        await exchange.close()
        print("\n✅ Готово!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())





