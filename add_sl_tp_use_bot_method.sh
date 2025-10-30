#!/bin/bash
# Скрипт для добавления SL/TP используя метод самого бота

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

echo "=================================="
echo "🔄 ДОБАВЛЕНИЕ SL/TP К ОТКРЫТЫМ ПОЗИЦИЯМ"
echo "=================================="
echo ""

ssh -i "$SSH_KEY" root@"$SERVER_IP" "python3 << 'PYADD'
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

async def add_sl_tp():
    try:
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear', 'accountType': 'UNIFIED'}
        })
        
        print('🔍 Получаем открытые позиции...')
        positions = await exchange.fetch_positions(params={'category': 'linear'})
        open_pos = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
        
        print(f'📊 Найдено позиций: {len(open_pos)}\\n')
        
        for pos in open_pos:
            symbol = pos.get('symbol', '')
            side = pos.get('side', '').lower()
            entry_price = float(pos.get('entryPrice', 0))
            size = float(pos.get('contracts', 0) or pos.get('size', 0))
            
            print(f'📌 {symbol} {side.upper()}: вход={entry_price:.8f}, размер={size}')
            
            # Рассчитываем SL и TP
            position_notional = POSITION_SIZE * LEVERAGE
            stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100
            
            if side in ['long', 'buy']:
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + TP_PERCENT / 100.0)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - TP_PERCENT / 100.0)
            
            print(f'   🛑 SL: {stop_loss_price:.8f}')
            print(f'   🎯 TP: {tp_price:.8f}')
            
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            # Используем прямой метод как в боте через API call
            print(f'   ⚙️ Устанавливаем SL/TP...')
            
            # Метод 1: Пробуем через edit_order или update ордеров
            # Но для существующих позиций нужно использовать position trading stop
            
            # Используем встроенный метод ccxt для позиций если доступен
            try:
                # Пробуем через set_leverage как тест доступности методов
                # А затем используем правильный эндпоинт
                import json
                
                # Используем внутренний метод exchange для прямого вызова
                # Bybit v5 требует POST /v5/position/trading-stop
                params_dict = {
                    'category': 'linear',
                    'symbol': bybit_symbol,
                    'stopLoss': str(stop_loss_price),
                    'takeProfit': str(tp_price),
                    'positionIdx': 0
                }
                
                # Попробуем через ccxt внутренний механизм
                try:
                    # Используем market().request() для прямого доступа к API
                    url = '/v5/position/trading-stop'
                    response = await exchange.request(url, 'private', 'post', params_dict)
                    
                    if response.get('retCode') == 0:
                        print(f'      ✅ SL и TP установлены!')
                    else:
                        print(f'      ⚠️ retCode={response.get(\"retCode\")}, msg={response.get(\"retMsg\")}')
                        
                        # Пробуем отдельно
                        if response.get('retCode') != 0:
                            # Только SL
                            try:
                                params_sl = {'category': 'linear', 'symbol': bybit_symbol, 'stopLoss': str(stop_loss_price), 'positionIdx': 0}
                                resp_sl = await exchange.request(url, 'private', 'post', params_sl)
                                if resp_sl.get('retCode') == 0:
                                    print(f'      ✅ SL установлен')
                                else:
                                    print(f'      ⚠️ SL: {resp_sl.get(\"retMsg\")}')
                            except Exception as e_sl:
                                print(f'      ⚠️ SL ошибка: {str(e_sl)[:80]}')
                            
                            # Только TP
                            try:
                                params_tp = {'category': 'linear', 'symbol': bybit_symbol, 'takeProfit': str(tp_price), 'positionIdx': 0}
                                resp_tp = await exchange.request(url, 'private', 'post', params_tp)
                                if resp_tp.get('retCode') == 0:
                                    print(f'      ✅ TP установлен')
                                else:
                                    print(f'      ⚠️ TP: {resp_tp.get(\"retMsg\")}')
                            except Exception as e_tp:
                                print(f'      ⚠️ TP ошибка: {str(e_tp)[:80]}')
                except Exception as e_req:
                    print(f'      ❌ Ошибка request: {str(e_req)[:80]}')
                    
            except Exception as e:
                print(f'      ❌ Ошибка: {str(e)[:80]}')
            
            print()
        
        await exchange.close()
        print('✅ Готово!')
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(add_sl_tp())
PYADD
"

echo ""
echo "=================================="
echo "✅ ГОТОВО!"
echo "=================================="





