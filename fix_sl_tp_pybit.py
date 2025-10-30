#!/usr/bin/env python3
"""
Установка SL/TP используя официальную библиотеку pybit от Bybit
"""

import asyncio
import sys
import os

sys.path.insert(0, '/opt/bot')
os.chdir('/opt/bot')
from dotenv import load_dotenv
load_dotenv('/opt/bot/api.env')

try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    print("⚠️ pybit не установлена. Устанавливаем...")

import ccxt.async_support as ccxt

POSITION_SIZE = 5.0
LEVERAGE = 5
MAX_STOP_LOSS_USD = 5.0
TP_PERCENT = 20.0


async def set_sl_tp_pybit(session, symbol: str, stop_loss: float = None, take_profit: float = None):
    """
    Устанавливает SL/TP через pybit библиотеку
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
        
        response = session.set_trading_stop(**params)
        return response.get('retCode'), response.get('retMsg', ''), response
        
    except Exception as e:
        return -1, str(e), None


async def main():
    try:
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        # Проверяем наличие pybit
        if not PYBIT_AVAILABLE:
            print("❌ pybit не доступна. Используем альтернативный метод.")
            return
        
        # Создаем сессию pybit
        session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Получаем позиции через ccxt
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
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
            
            # Устанавливаем через pybit
            print(f"   ⚙️ Устанавливаем...")
            ret_code, ret_msg, response = await set_sl_tp_pybit(session, symbol, stop_loss_price, tp_price)
            
            if ret_code == 0:
                print(f"      ✅ SL и TP установлены!")
            else:
                print(f"      ⚠️ Ошибка: retCode={ret_code}, msg={ret_msg}")
                
                # Пробуем отдельно
                if ret_code != 0:
                    ret_sl, msg_sl, _ = await set_sl_tp_pybit(session, symbol, stop_loss_price, None)
                    if ret_sl == 0:
                        print(f"      ✅ SL установлен!")
                    else:
                        print(f"      ⚠️ SL: {msg_sl}")
                    
                    ret_tp, msg_tp, _ = await set_sl_tp_pybit(session, symbol, None, tp_price)
                    if ret_tp == 0:
                        print(f"      ✅ TP установлен!")
                    else:
                        print(f"      ⚠️ TP: {msg_tp}")
        
        await exchange.close()
        print("\n✅ Готово!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())





