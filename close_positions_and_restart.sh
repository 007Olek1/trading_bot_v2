#!/bin/bash
# Скрипт для закрытия позиций и перезапуска бота

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"

echo "=================================="
echo "🔄 ЗАКРЫТИЕ ПОЗИЦИЙ И ПЕРЕЗАПУСК БОТА"
echo "=================================="
echo ""

# Функция для выполнения команд на сервере
execute_remote() {
    ssh -i "$SSH_KEY" root@"$SERVER_IP" "$1"
}

# 1. Закрываем открытые позиции на бирже
echo "1️⃣ ЗАКРЫТИЕ ОТКРЫТЫХ ПОЗИЦИЙ:"
echo "----------------------------------"
execute_remote "python3 << 'PYCLOSE'
import asyncio
import sys
import os
sys.path.insert(0, '$BOT_DIR')
os.chdir('$BOT_DIR')
from dotenv import load_dotenv
load_dotenv('$BOT_DIR/api.env')

import ccxt.async_support as ccxt

async def close_all_positions():
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
        
        # Получаем позиции
        try:
            positions = await exchange.fetch_positions(params={'category': 'linear'})
        except Exception as e1:
            try:
                positions = await exchange.fetch_positions()
            except Exception as e2:
                print(f'❌ Ошибка получения позиций: {e1} / {e2}')
                await exchange.close()
                return
        
        open_positions = []
        for pos in positions:
            size = pos.get('contracts', 0) or pos.get('size', 0)
            if size > 0:
                open_positions.append(pos)
        
        print(f'📊 Найдено открытых позиций: {len(open_positions)}')
        
        if not open_positions:
            print('✅ Открытых позиций нет')
            await exchange.close()
            return
        
        # Закрываем каждую позицию
        for pos in open_positions:
            symbol = pos.get('symbol', 'N/A')
            side = pos.get('side', '').lower()
            size = pos.get('contracts', 0) or pos.get('size', 0)
            
            print(f'\\n📌 Закрываем {symbol} {side.upper()} (размер: {size})')
            
            try:
                # Закрываем противоположным ордером
                close_side = 'sell' if side == 'long' or side == 'buy' else 'buy'
                close_order = await exchange.create_market_order(
                    symbol=symbol,
                    side=close_side,
                    amount=size,
                    params={
                        'category': 'linear',
                        'reduceOnly': True
                    }
                )
                print(f'   ✅ Позиция закрыта! Order ID: {close_order.get(\"id\", \"N/A\")}')
            except Exception as e:
                print(f'   ❌ Ошибка закрытия: {e}')
        
        await exchange.close()
        print('\\n✅ Все позиции закрыты!')
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(close_all_positions())
PYCLOSE
"

echo ""
echo "2️⃣ ОСТАНОВКА БОТА:"
echo "----------------------------------"
execute_remote "systemctl stop trading-bot"

echo ""
echo "3️⃣ ОЖИДАНИЕ ПОЛНОЙ ОСТАНОВКИ (5 сек):"
echo "----------------------------------"
sleep 5

echo ""
echo "4️⃣ ПРОВЕРКА ПРОЦЕССОВ:"
echo "----------------------------------"
execute_remote "ps aux | grep -E 'super_bot_v4_mtf\.py|python.*bot' | grep -v grep || echo '   ✅ Процессы остановлены'"

echo ""
echo "5️⃣ ЗАГРУЗКА ОБНОВЛЕННОГО КОДА:"
echo "----------------------------------"

# Копируем основные файлы
echo "   📁 Копирование файлов..."
scp -i "$SSH_KEY" super_bot_v4_mtf.py root@"$SERVER_IP":"$BOT_DIR/"
scp -i "$SSH_KEY" smart_coin_selector.py root@"$SERVER_IP":"$BOT_DIR/"

echo ""
echo "6️⃣ ПРОВЕРКА СИНТАКСИСА:"
echo "----------------------------------"
execute_remote "cd $BOT_DIR && python3 -m py_compile super_bot_v4_mtf.py && echo '   ✅ Синтаксис правильный' || echo '   ❌ Ошибка синтаксиса'"

echo ""
echo "7️⃣ ПЕРЕЗАПУСК БОТА:"
echo "----------------------------------"
execute_remote "systemctl start trading-bot"

echo ""
echo "8️⃣ ПРОВЕРКА СТАТУСА:"
echo "----------------------------------"
sleep 3
execute_remote "systemctl status trading-bot --no-pager | head -15"

echo ""
echo "9️⃣ ПРОВЕРКА ЛОГОВ (последние 30 строк):"
echo "----------------------------------"
sleep 5
execute_remote "tail -30 $BOT_DIR/logs/system/bot.log 2>/dev/null || tail -30 $BOT_DIR/bot.log 2>/dev/null || echo '   Логи еще не созданы'"

echo ""
echo "=================================="
echo "✅ ГОТОВО!"
echo "=================================="
echo ""
echo "📋 Для мониторинга используйте:"
echo "   ssh -i $SSH_KEY root@$SERVER_IP 'tail -f $BOT_DIR/logs/system/bot.log'"
echo ""
