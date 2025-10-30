#!/bin/bash
# Скрипт для проверки позиций и SL/TP на бирже

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

ssh -i "$SSH_KEY" root@"$SERVER_IP" "python3 << 'PYCHECK'
import asyncio
import sys
import os
sys.path.insert(0, '/opt/bot')
os.chdir('/opt/bot')
from dotenv import load_dotenv
load_dotenv('/opt/bot/api.env')

import ccxt.async_support as ccxt

async def check_positions():
    try:
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear', 'accountType': 'UNIFIED'}
        })
        
        print('🔍 ПРОВЕРКА ПОЗИЦИЙ И SL/TP НА БИРЖЕ:')
        print('=' * 70)
        
        # Получаем позиции
        positions = await exchange.fetch_positions(params={'category': 'linear'})
        open_pos = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
        
        print(f'📊 Открытых позиций: {len(open_pos)}\\n')
        
        if not open_pos:
            print('✅ Нет открытых позиций')
        else:
            for p in open_pos:
                sym = p.get('symbol', 'N/A')
                sz = p.get('contracts', 0) or p.get('size', 0)
                side = p.get('side', 'N/A')
                entry = p.get('entryPrice', 0) or p.get('entry', 0)
                sl = p.get('stopLossPrice', None) or p.get('stopLoss', None) or p.get('stopPrice', None)
                tp = p.get('takeProfitPrice', None) or p.get('takeProfit', None)
                
                print(f'📌 {sym} {side.upper()}:')
                print(f'   Размер: {sz}')
                print(f'   Вход: \${entry:.8f}' if entry else '   Вход: N/A')
                print(f'   🛑 SL: \${sl:.8f}' if sl else '   🛑 SL: ❌ НЕ УСТАНОВЛЕН')
                print(f'   🎯 TP: \${tp:.8f}' if tp else '   🎯 TP: ❌ НЕ УСТАНОВЛЕН')
                print()
        
        # Проверяем открытые ордера (SL/TP ордера)
        print('\\n🔍 ПРОВЕРКА ОТКРЫТЫХ ОРДЕРОВ (SL/TP):')
        print('=' * 70)
        
        try:
            orders = await exchange.fetch_open_orders(params={'category': 'linear'})
            sl_tp_orders = [o for o in orders if o.get('type', '').lower() in ['stop', 'stopmarket', 'takeprofit', 'takeprofitmarket']]
            
            print(f'📊 Найдено SL/TP ордеров: {len(sl_tp_orders)}\\n')
            
            if not sl_tp_orders:
                print('⚠️ Нет открытых SL/TP ордеров на бирже!')
            else:
                for o in sl_tp_orders:
                    sym = o.get('symbol', 'N/A')
                    otype = o.get('type', 'N/A')
                    price = o.get('stopPrice', 0) or o.get('triggerPrice', 0) or o.get('price', 0)
                    status = o.get('status', 'N/A')
                    print(f'   {sym} | {otype} | Цена: \${price:.8f} | Статус: {status}')
        except Exception as e:
            print(f'   ⚠️ Ошибка получения ордеров: {e}')
        
        await exchange.close()
        print('\\n' + '=' * 70)
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(check_positions())
PYCHECK
"

