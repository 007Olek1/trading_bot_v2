#!/bin/bash
# Контрольное тестирование системы перед ночью

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

echo "========================================"
echo "🌙 КОНТРОЛЬНОЕ ТЕСТИРОВАНИЕ ПЕРЕД НОЧЬЮ"
echo "========================================"
echo ""
echo "Дата: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

ssh -i "$SSH_KEY" root@"$SERVER_IP" "python3 << 'PYTEST'
import asyncio
import sys
import os
sys.path.insert(0, '/opt/bot')
os.chdir('/opt/bot')
from dotenv import load_dotenv
load_dotenv('/opt/bot/api.env')

import ccxt.async_support as ccxt
from datetime import datetime
import pytz

# Переменные для проверки
POSITION_SIZE = 5.0
LEVERAGE = 5
MAX_STOP_LOSS_USD = 5.0
TP_PERCENT = 20.0
WARSAW_TZ = pytz.timezone('Europe/Warsaw')

def print_section(title):
    print('')
    print('=' * 70)
    print(f'📋 {title}')
    print('=' * 70)

def print_check(item, status, details=''):
    status_icon = '✅' if status else '❌'
    print(f'{status_icon} {item}', end='')
    if details:
        print(f': {details}')
    else:
        print()

async def test_system():
    try:
        # Инициализация биржи
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear', 'accountType': 'UNIFIED'}
        })
        
        # 1. ПРОВЕРКА БАЛАНСА
        print_section('1. ПРОВЕРКА БАЛАНСА')
        try:
            balance = await exchange.fetch_balance({'accountType': 'UNIFIED'})
            total = balance.get('total', {})
            free = balance.get('free', {})
            used = balance.get('used', {})
            
            usdt_total = total.get('USDT', 0)
            usdt_free = free.get('USDT', 0)
            usdt_used = used.get('USDT', 0)
            
            print_check('Общий баланс', usdt_total > 0, f'\${usdt_total:.2f} USDT')
            print_check('Свободный баланс', usdt_free > 0, f'\${usdt_free:.2f} USDT')
            print_check('Использовано', True, f'\${usdt_used:.2f} USDT')
            
            # Проверка достаточности баланса
            required_per_trade = POSITION_SIZE
            can_open_trades = int(usdt_free / required_per_trade)
            print_check('Достаточно для сделок', usdt_free >= required_per_trade, f'{can_open_trades} сделок можно открыть')
            
        except Exception as e:
            print_check('Получение баланса', False, str(e)[:80])
        
        # 2. ПРОВЕРКА ОТКРЫТЫХ ПОЗИЦИЙ
        print_section('2. ПРОВЕРКА ОТКРЫТЫХ ПОЗИЦИЙ')
        try:
            positions = await exchange.fetch_positions(params={'category': 'linear'})
            open_positions = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
            
            print_check('Найдено позиций', True, f'{len(open_positions)}')
            
            position_notional = POSITION_SIZE * LEVERAGE
            
            all_have_sl = True
            all_have_tp = True
            all_correct_sl = True
            all_correct_tp = True
            
            for pos in open_positions:
                symbol = pos.get('symbol', '')
                side = pos.get('side', '').lower()
                entry = float(pos.get('entryPrice', 0))
                size = float(pos.get('contracts', 0) or pos.get('size', 0))
                
                sl = pos.get('stopLossPrice') or pos.get('stopLoss')
                tp = pos.get('takeProfitPrice') or pos.get('takeProfit')
                
                if not sl:
                    all_have_sl = False
                if not tp:
                    all_have_tp = False
                
                # Проверяем расчет SL
                if entry > 0:
                    stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100
                    if side in ['long', 'buy']:
                        expected_sl = entry * (1 - stop_loss_percent / 100.0)
                    else:
                        expected_sl = entry * (1 + stop_loss_percent / 100.0)
                    
                    if sl:
                        sl_diff = abs(float(sl) - expected_sl) / entry * 100
                        if sl_diff > 1.0:  # Разница более 1%
                            all_correct_sl = False
                    
                    # Проверяем расчет TP
                    if side in ['long', 'buy']:
                        expected_tp = entry * (1 + TP_PERCENT / 100.0)
                    else:
                        expected_tp = entry * (1 - TP_PERCENT / 100.0)
                    
                    if tp:
                        tp_diff = abs(float(tp) - expected_tp) / entry * 100
                        if tp_diff > 1.0:  # Разница более 1%
                            all_correct_tp = False
                    
                    expected_profit = position_notional * (TP_PERCENT / 100.0)
                    
                    print(f'\\n  📌 {symbol} {side.upper()}:')
                    print(f'     Вход: \${entry:.8f}')
                    print(f'     Размер: {size}')
                    if sl:
                        print(f'     🛑 SL: \${float(sl):.8f}')
                    else:
                        print(f'     🛑 SL: ❌ НЕТ')
                    if tp:
                        print(f'     🎯 TP: \${float(tp):.8f} (+{TP_PERCENT}% = \${expected_profit:.2f})')
                    else:
                        print(f'     🎯 TP: ❌ НЕТ')
            
            print('')
            print_check('Все позиции имеют SL', all_have_sl)
            print_check('Все позиции имеют TP', all_have_tp)
            print_check('SL рассчитаны правильно', all_correct_sl)
            print_check('TP рассчитаны правильно', all_correct_tp)
            
        except Exception as e:
            print_check('Получение позиций', False, str(e)[:80])
        
        # 3. ПРОВЕРКА СИСТЕМНЫХ ПАРАМЕТРОВ
        print_section('3. ПРОВЕРКА ПАРАМЕТРОВ')
        print_check('POSITION_SIZE', True, f'\${POSITION_SIZE:.2f}')
        print_check('LEVERAGE', True, f'{LEVERAGE}x')
        print_check('POSITION_NOTIONAL', True, f'\${POSITION_SIZE * LEVERAGE:.2f}')
        print_check('MAX_STOP_LOSS_USD', True, f'\${MAX_STOP_LOSS_USD:.2f}')
        print_check('TP_PERCENT', True, f'+{TP_PERCENT}%')
        print_check('TP_PROFIT_USD', True, f'\${POSITION_SIZE * LEVERAGE * TP_PERCENT / 100.0:.2f}')
        
        # 4. ПРОВЕРКА СТАТУСА БОТА
        print_section('4. ПРОВЕРКА СТАТУСА БОТА')
        import subprocess
        result = subprocess.run(['systemctl', 'is-active', 'trading-bot'], 
                              capture_output=True, text=True)
        bot_active = result.stdout.strip() == 'active'
        print_check('Бот запущен', bot_active)
        
        if bot_active:
            result = subprocess.run(['systemctl', 'status', 'trading-bot', '--no-pager'], 
                                  capture_output=True, text=True)
            if 'Active: active (running)' in result.stdout:
                print_check('Бот работает', True)
            else:
                print_check('Бот работает', False)
        
        # 5. ПРОВЕРКА ЛОГОВ
        print_section('5. ПРОВЕРКА ЛОГОВ')
        log_file = '/opt/bot/logs/bot.log'
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    last_lines = lines[-10:] if len(lines) > 10 else lines
                    
                print_check('Файл логов существует', True)
                print_check('Последние записи', True, f'{len(last_lines)} строк')
                
                # Проверяем на ошибки в последних строках
                errors = [l for l in last_lines if 'ERROR' in l or '❌' in l or 'Exception' in l]
                if errors:
                    print_check('Ошибки в логах', False, f'{len(errors)} найдено')
                else:
                    print_check('Ошибки в логах', True, 'Нет ошибок')
                    
            except Exception as e:
                print_check('Чтение логов', False, str(e)[:80])
        else:
            print_check('Файл логов', False, 'Не найден')
        
        # 6. ПРОВЕРКА ВРЕМЕНИ
        print_section('6. ПРОВЕРКА ВРЕМЕНИ')
        now_warsaw = datetime.now(WARSAW_TZ)
        print_check('Текущее время (Варшава)', True, now_warsaw.strftime('%H:%M:%S %d.%m.%Y'))
        
        # 7. ИТОГОВАЯ ПРОВЕРКА
        print_section('7. ИТОГОВЫЙ СТАТУС')
        
        all_ok = (
            usdt_total > 0 and
            usdt_free >= required_per_trade and
            all_have_sl and
            all_have_tp and
            all_correct_sl and
            all_correct_tp and
            bot_active
        )
        
        if all_ok:
            print('✅ ВСЕ СИСТЕМЫ РАБОТАЮТ ПРАВИЛЬНО')
            print('🌙 СИСТЕМА ГОТОВА К РАБОТЕ НА НОЧЬ')
        else:
            print('⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ - ТРЕБУЕТСЯ ВНИМАНИЕ')
            if not all_have_sl:
                print('   - Не все позиции имеют SL')
            if not all_have_tp:
                print('   - Не все позиции имеют TP')
            if not bot_active:
                print('   - Бот не запущен')
            if usdt_free < required_per_trade:
                print(f'   - Недостаточно баланса (нужно \${required_per_trade}, есть \${usdt_free:.2f})')
        
        await exchange.close()
        
    except Exception as e:
        print(f'❌ КРИТИЧЕСКАЯ ОШИБКА: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_system())
PYTEST
"

echo ""
echo "========================================"
echo "✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО"
echo "========================================"





