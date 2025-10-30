#!/bin/bash
# Скрипт для добавления SL/TP через conditional orders (как в боте)

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

ssh -i "$SSH_KEY" root@"$SERVER_IP" "cd /opt/bot && python3 << 'PYADD'
import asyncio
import sys
import os
sys.path.insert(0, '/opt/bot')
from dotenv import load_dotenv
load_dotenv('/opt/bot/api.env')

import ccxt.async_support as ccxt

POSITION_SIZE = 5.0
LEVERAGE = 5
MAX_STOP_LOSS_USD = 1.0
TP_PERCENT = 1.0

async def add_sl_tp():
    try:
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear', 'accountType': 'UNIFIED'}
        })
        
        # Подготовка blacklist: из env и из файла на сервере
        innovation_blacklist = { 'TURTLEUSDT' }
        try:
            bl_env = os.getenv('INNOVATION_BLACKLIST', '')
            for token in [x.strip().upper().replace('/', '').replace('-', '') for x in bl_env.split(',') if x.strip()]:
                if token.endswith(':USDT'):
                    token = token[:-5] + 'USDT'
                token = token.replace(':', '')
                if not token.endswith('USDT') and token:
                    token = token + 'USDT'
                innovation_blacklist.add(token)
            bl_path = '/opt/bot/config/blacklist_symbols.txt'
            if os.path.exists(bl_path):
                with open(bl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        token = line.strip().upper().replace('/', '').replace('-', '')
                        if not token:
                            continue
                        if token.endswith(':USDT'):
                            token = token[:-5] + 'USDT'
                        token = token.replace(':', '')
                        if not token.endswith('USDT'):
                            token = token + 'USDT'
                        innovation_blacklist.add(token)
        except Exception:
            pass
        
        # Загрузим markets для автодетекции по маркерам биржи
        try:
            await exchange.load_markets()
        except Exception:
            pass
        
        print('🔍 Получаем открытые позиции...')
        positions = await exchange.fetch_positions(params={'category': 'linear'})
        open_pos = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
        
        print(f'📊 Найдено позиций: {len(open_pos)}\\n')
        
        for pos in open_pos:
            symbol = pos.get('symbol', '')
            side = pos.get('side', '').lower()
            entry_price = float(pos.get('entryPrice', 0))
            size = float(pos.get('contracts', 0) or pos.get('size', 0))
            
            # Пропускаем токены из «Зоны инноваций»/высокого риска
            norm_symbol = (symbol or '').replace('/', '').replace('-', '').upper()
            if norm_symbol.endswith(':USDT'):
                norm_symbol = norm_symbol[:-5] + 'USDT'
            elif ':USDT' in norm_symbol:
                norm_symbol = norm_symbol.replace(':USDT', '') + 'USDT'
            norm_symbol = norm_symbol.replace(':', '')
            if norm_symbol in innovation_blacklist:
                print(f"   🚫 Пропуск {norm_symbol}: в blacklist Innovation Zone")
                continue
            # Автодетекция по маркерам биржи
            try:
                market = exchange.markets.get(symbol) or exchange.markets.get(norm_symbol)
                info = (market or {}).get('info', {})
                blob = ' '.join(list(info.keys()) + [str(v) for v in info.values()]).lower()
                markers = {'innovation','newlisting','new_listing','hot','risk','specialtreatment','st','seed','launchpad','trial','isolated_only'}
                if any(m in blob for m in markers):
                    print(f"   🚫 Пропуск {norm_symbol}: помечен как innovation/risk в info")
                    continue
            except Exception:
                pass
            
            print(f'📌 {symbol} {side.upper()}: вход={entry_price:.8f}, размер={size}')
            
            # Рассчитываем SL и TP
            position_notional = POSITION_SIZE * LEVERAGE
            stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100
            
            if side in ['long', 'buy']:
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + TP_PERCENT / 100.0)
                sl_side = 'sell'
                tp_side = 'sell'
                trigger_direction_sl = 'descending'
                trigger_direction_tp = 'ascending'
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - TP_PERCENT / 100.0)
                sl_side = 'buy'
                tp_side = 'buy'
                trigger_direction_sl = 'ascending'
                trigger_direction_tp = 'descending'

            print(f'   🛑 SL: {stop_loss_price:.8f}')
            print(f'   🎯 TP: {tp_price:.8f}')
            
            # Устанавливаем SL через conditional order
            try:
                sl_order = await exchange.create_order(
                    symbol=symbol,
                    type='StopMarket',
                    side=sl_side,
                    amount=size,
                    params={
                        'category': 'linear',
                        'stopPrice': stop_loss_price,
                        'triggerPrice': stop_loss_price,
                        'triggerDirection': trigger_direction_sl,
                        'reduceOnly': True,
                        'closeOnTrigger': True,
                        'positionIdx': 0
                    }
                )
                print(f'   ✅ SL установлен: {sl_order.get(\"id\", \"N/A\")}')
            except Exception as e:
                print(f'   ❌ SL ошибка: {str(e)[:100]}')
            
            # Устанавливаем TP через conditional order
            try:
                tp_order = await exchange.create_order(
                    symbol=symbol,
                    type='TakeProfitMarket',
                    side=tp_side,
                    amount=size,
                    params={
                        'category': 'linear',
                        'stopPrice': tp_price,
                        'triggerPrice': tp_price,
                        'triggerDirection': trigger_direction_tp,
                        'reduceOnly': True,
                        'closeOnTrigger': True,
                        'positionIdx': 0
                    }
                )
                print(f'   ✅ TP установлен: {tp_order.get(\"id\", \"N/A\")}')
            except Exception as e:
                print(f'   ❌ TP ошибка: {str(e)[:100]}')
            
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
