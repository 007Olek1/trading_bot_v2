#!/bin/bash
# Скрипт для добавления SL/TP к существующим позициям на бирже

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
from datetime import datetime

# Параметры торговли
POSITION_SIZE = 5.0  # \$5 маржа
LEVERAGE = 5  # 5x плечо
MAX_STOP_LOSS_USD = 5.0  # Максимум -\$5
TP_PERCENT = 20.0  # +20% TP

async def add_sl_tp_to_positions():
    try:
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'accountType': 'UNIFIED'
            }
        })
        
        print('🔍 Получаем открытые позиции...')
        print('=' * 70)
        
        # Получаем позиции
        positions = await exchange.fetch_positions(params={'category': 'linear'})
        open_positions = []
        
        for p in positions:
            size = p.get('contracts', 0) or p.get('size', 0)
            if size > 0:
                open_positions.append(p)
        
        print(f'📊 Найдено открытых позиций: {len(open_positions)}\\n')
        
        if not open_positions:
            print('✅ Нет открытых позиций')
            await exchange.close()
            return
        
        # Обрабатываем каждую позицию
        for pos in open_positions:
            symbol = pos.get('symbol', 'N/A')
            side = pos.get('side', '').lower()
            entry_price = float(pos.get('entryPrice', 0) or pos.get('entry', 0))
            size = float(pos.get('contracts', 0) or pos.get('size', 0))
            current_price = float(pos.get('markPrice', 0) or pos.get('mark', 0))
            
            print(f'\\n📌 Обрабатываем {symbol} {side.upper()}:')
            print(f'   Вход: \${entry_price:.8f}')
            print(f'   Размер: {size}')
            print(f'   Текущая цена: \${current_price:.8f}')
            
            # Рассчитываем SL (максимум -\$5)
            # Формула: stop_loss_price = entry_price * (1 ± STOP_LOSS_PERCENT / 100)
            # STOP_LOSS_PERCENT = MAX_STOP_LOSS_USD / POSITION_NOTIONAL * 100
            position_notional = POSITION_SIZE * LEVERAGE  # \$25
            stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100  # 20%
            
            if side == 'long' or side == 'buy':
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + TP_PERCENT / 100.0)
                trigger_direction_sl = 'descending'
                trigger_direction_tp = 'ascending'
                sl_side = 'sell'
                tp_side = 'sell'
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - TP_PERCENT / 100.0)
                trigger_direction_sl = 'ascending'
                trigger_direction_tp = 'descending'
                sl_side = 'buy'
                tp_side = 'buy'
            
            print(f'   🛑 SL: \${stop_loss_price:.8f} (-\${MAX_STOP_LOSS_USD:.2f} максимум)')
            print(f'   🎯 TP: \${tp_price:.8f} (+{TP_PERCENT:.0f}%)')
            
            # Нормализуем символ для Bybit
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            # 1. Устанавливаем Stop Loss и Take Profit через прямой API позиции
            print(f'\\n   ⚙️ Устанавливаем Stop Loss и Take Profit...')
            sl_set = False
            tp_set = False
            
            try:
                # Используем прямой API вызов через request метод
                # Сначала пробуем установить оба
                try:
                    # Устанавливаем SL и TP одновременно через прямой API
                    response = await exchange.request('private', 'post', '/v5/position/trading-stop', {
                        'category': 'linear',
                        'symbol': bybit_symbol,
                        'stopLoss': str(stop_loss_price),
                        'takeProfit': str(tp_price),
                        'positionIdx': 0
                    })
                    if response.get('retCode') == 0:
                        print(f'      ✅ SL и TP установлены через API позиции')
                        sl_set = True
                        tp_set = True
                    else:
                        ret_code = response.get('retCode')
                        ret_msg = response.get('retMsg', 'N/A')
                        print(f'      ⚠️ Ошибка API: retCode={ret_code}, msg={ret_msg}')
                        # Пробуем установить отдельно
                        if ret_code != 0:
                            # Пробуем только SL
                            try:
                                response_sl = await exchange.request('private', 'post', '/v5/position/trading-stop', {
                                    'category': 'linear',
                                    'symbol': bybit_symbol,
                                    'stopLoss': str(stop_loss_price),
                                    'positionIdx': 0
                                })
                                if response_sl.get('retCode') == 0:
                                    print(f'      ✅ SL установлен отдельно')
                                    sl_set = True
                            except Exception as e_sl:
                                print(f'      ⚠️ Ошибка установки SL: {str(e_sl)[:100]}')
                            
                            # Пробуем только TP
                            try:
                                response_tp = await exchange.request('private', 'post', '/v5/position/trading-stop', {
                                    'category': 'linear',
                                    'symbol': bybit_symbol,
                                    'takeProfit': str(tp_price),
                                    'positionIdx': 0
                                })
                                if response_tp.get('retCode') == 0:
                                    print(f'      ✅ TP установлен отдельно')
                                    tp_set = True
                            except Exception as e_tp:
                                print(f'      ⚠️ Ошибка установки TP: {str(e_tp)[:100]}')
            except Exception as e:
                print(f'      ⚠️ Ошибка установки SL/TP: {str(e)[:100]}')
            
            # Итог для позиции
            if sl_set and tp_set:
                print(f'\\n   ✅ {symbol}: SL и TP успешно установлены!')
            elif sl_set:
                print(f'\\n   ⚠️ {symbol}: SL установлен, TP не установлен')
            elif tp_set:
                print(f'\\n   ⚠️ {symbol}: TP установлен, SL не установлен')
            else:
                print(f'\\n   ❌ {symbol}: Не удалось установить SL/TP')
        
        print('\\n' + '=' * 70)
        print('✅ Обработка завершена!')
        
        await exchange.close()
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(add_sl_tp_to_positions())
PYADD
"

echo ""
echo "=================================="
echo "✅ ГОТОВО!"
echo "=================================="

