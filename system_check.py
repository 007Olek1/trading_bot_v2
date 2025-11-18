#!/usr/bin/env python3
"""
ГЕНЕРАЛЬНАЯ ПРОВЕРКА СИСТЕМЫ
"""
import sys
import os
sys.path.insert(0, '/root/trading_bot_v4_mtf')

import json
from datetime import datetime
from pathlib import Path

print('='*80)
print('🔍 ГЕНЕРАЛЬНАЯ ПРОВЕРКА СИСТЕМЫ TRADING BOT V4.0')
print('='*80)
print()

# ═══════════════════════════════════════════════════════════════════
# 1. ПРОВЕРКА ФАЙЛОВ И МОДУЛЕЙ
# ═══════════════════════════════════════════════════════════════════
print('📁 1. ПРОВЕРКА ФАЙЛОВ И МОДУЛЕЙ')
print('-'*80)

required_files = {
    'main.py': 'Главный модуль',
    'config.py': 'Конфигурация',
    'strategies.py': 'Стратегии',
    'indicators.py': 'Индикаторы',
    'telegram_handler.py': 'Telegram',
    'trade_logger.py': 'Логирование сделок',
    'log_rotator.py': 'Ротация логов',
    'adaptive_tp_manager.py': 'Адаптивный TP',
    'market_scanner.py': 'Сканер рынка',
    'fee_calculator.py': 'Калькулятор комиссий',
    'daily_reporter.py': 'Ежедневные отчеты',
    'utils.py': 'Утилиты',
    'backtester.py': 'Бэктестинг',
    'check_positions.py': 'Проверка позиций',
    'apply_sl_tp.py': 'Применение SL/TP',
    'show_stats.py': 'Статистика'
}

for file, desc in required_files.items():
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024
        print(f'  ✅ {file:30} ({desc:25}) {size:>6.1f} KB')
    else:
        print(f'  ❌ {file:30} ({desc:25}) ОТСУТСТВУЕТ!')

print()

# ═══════════════════════════════════════════════════════════════════
# 2. ПРОВЕРКА ДИРЕКТОРИЙ И ЛОГОВ
# ═══════════════════════════════════════════════════════════════════
print('📂 2. ПРОВЕРКА ДИРЕКТОРИЙ И ЛОГОВ')
print('-'*80)

logs_dir = Path('logs')
if logs_dir.exists():
    print(f'  ✅ Директория logs существует')
    
    # Проверяем файлы логов
    log_files = list(logs_dir.glob('*'))
    total_size = sum(f.stat().st_size for f in log_files if f.is_file())
    
    print(f'  📊 Всего файлов: {len(log_files)}')
    print(f'  💾 Общий размер: {total_size / 1024 / 1024:.2f} MB')
    print()
    
    # Основные логи
    main_logs = {
        'trading_bot_v4.log': 'Основной лог',
        'trades.json': 'Лог сделок',
        'bot_output.log': 'Вывод бота'
    }
    
    for log_file, desc in main_logs.items():
        log_path = logs_dir / log_file
        if log_path.exists():
            size = log_path.stat().st_size / 1024
            mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
            print(f'  ✅ {log_file:25} ({desc:20}) {size:>8.1f} KB | {mtime.strftime("%Y-%m-%d %H:%M")}')
        else:
            print(f'  ⚠️  {log_file:25} ({desc:20}) НЕ НАЙДЕН')
    
    print()
    
    # Архивные логи
    archived = list(logs_dir.glob('*.gz'))
    if archived:
        print(f'  📦 Архивных логов: {len(archived)}')
        arch_size = sum(f.stat().st_size for f in archived)
        print(f'  💾 Размер архивов: {arch_size / 1024 / 1024:.2f} MB')
    else:
        print(f'  📦 Архивных логов: 0')
else:
    print(f'  ❌ Директория logs НЕ СУЩЕСТВУЕТ!')

print()

# ═══════════════════════════════════════════════════════════════════
# 3. ПРОВЕРКА ЛОГИКИ РАБОТЫ (ИМПОРТЫ)
# ═══════════════════════════════════════════════════════════════════
print('🔧 3. ПРОВЕРКА ЛОГИКИ РАБОТЫ')
print('-'*80)

try:
    import config
    print(f'  ✅ config.py загружен')
    print(f'     - Таймфреймы: {list(config.TIMEFRAMES.keys())}')
    print(f'     - Leverage: {config.LEVERAGE}x')
    print(f'     - Max positions: {config.MAX_CONCURRENT_POSITIONS}')
    print(f'     - Min confidence: {config.SIGNAL_THRESHOLDS["min_confidence"]:.0%}')
    print(f'     - Trailing SL: {config.TRAILING_SL_ACTIVATION_PERCENT}% / {config.TRAILING_SL_CALLBACK_PERCENT}%')
    print(f'     - Adaptive TP: {config.MIN_TP_ROI}% → {config.MAX_TP_ROI}%')
except Exception as e:
    print(f'  ❌ Ошибка загрузки config: {e}')

print()

try:
    from strategies import TrendVolumeStrategy
    print(f'  ✅ TrendVolumeStrategy загружена')
except Exception as e:
    print(f'  ❌ Ошибка загрузки стратегии: {e}')

try:
    from indicators import MarketIndicators
    print(f'  ✅ MarketIndicators загружены')
except Exception as e:
    print(f'  ❌ Ошибка загрузки индикаторов: {e}')

try:
    from trade_logger import TradeLogger
    print(f'  ✅ TradeLogger загружен')
except Exception as e:
    print(f'  ❌ Ошибка загрузки TradeLogger: {e}')

try:
    from log_rotator import LogRotator
    print(f'  ✅ LogRotator загружен')
except Exception as e:
    print(f'  ❌ Ошибка загрузки LogRotator: {e}')

try:
    from adaptive_tp_manager import AdaptiveTPManager
    print(f'  ✅ AdaptiveTPManager загружен')
except Exception as e:
    print(f'  ❌ Ошибка загрузки AdaptiveTPManager: {e}')

print()

# ═══════════════════════════════════════════════════════════════════
# 4. ПРОВЕРКА ЛОГОВ СДЕЛОК
# ═══════════════════════════════════════════════════════════════════
print('📊 4. ПРОВЕРКА ЛОГОВ СДЕЛОК')
print('-'*80)

trades_file = Path('logs/trades.json')
if trades_file.exists():
    try:
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        open_trades = [t for t in trades if t.get('status') == 'open']
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        
        print(f'  ✅ Файл trades.json читается')
        print(f'  📊 Всего сделок: {len(trades)}')
        print(f'  🟢 Открытых: {len(open_trades)}')
        print(f'  🔴 Закрытых: {len(closed_trades)}')
        
        if closed_trades:
            profitable = [t for t in closed_trades if t.get('pnl_usd', 0) > 0]
            print(f'  💰 Прибыльных: {len(profitable)} ({len(profitable)/len(closed_trades)*100:.1f}%)')
    except Exception as e:
        print(f'  ⚠️  Ошибка чтения trades.json: {e}')
else:
    print(f'  ⚠️  Файл trades.json не найден')

print()

# ═══════════════════════════════════════════════════════════════════
# 5. ПРОВЕРКА МЕСТА НА ДИСКЕ
# ═══════════════════════════════════════════════════════════════════
print('💾 5. ПРОВЕРКА МЕСТА НА ДИСКЕ')
print('-'*80)

import subprocess
try:
    df_output = subprocess.check_output(['df', '-h', '/']).decode()
    lines = df_output.strip().split('\n')
    if len(lines) > 1:
        header = lines[0]
        data = lines[1].split()
        
        print(f'  Filesystem: {data[0]}')
        print(f'  Size: {data[1]}')
        print(f'  Used: {data[2]}')
        print(f'  Available: {data[3]}')
        print(f'  Use%: {data[4]}')
        
        usage_percent = int(data[4].rstrip('%'))
        if usage_percent > 90:
            print(f'  ⚠️  ВНИМАНИЕ: Диск заполнен на {usage_percent}%!')
        elif usage_percent > 80:
            print(f'  ⚠️  Диск заполнен на {usage_percent}%')
        else:
            print(f'  ✅ Место на диске в норме ({usage_percent}%)')
except Exception as e:
    print(f'  ⚠️  Ошибка проверки диска: {e}')

print()

# Размер директории бота
try:
    bot_size = subprocess.check_output(['du', '-sh', '.']).decode().split()[0]
    print(f'  📁 Размер директории бота: {bot_size}')
except:
    pass

print()

# ═══════════════════════════════════════════════════════════════════
# 6. ПРОВЕРКА АКТИВНЫХ ПОЗИЦИЙ
# ═══════════════════════════════════════════════════════════════════
print('📈 6. ПРОВЕРКА АКТИВНЫХ ПОЗИЦИЙ')
print('-'*80)

try:
    from pybit.unified_trading import HTTP
    import config
    
    client = HTTP(
        testnet=config.USE_TESTNET,
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET
    )
    
    response = client.get_positions(category='linear', settleCoin='USDT')
    
    if response['retCode'] == 0:
        positions = [p for p in response['result']['list'] if float(p['size']) > 0]
        print(f'  ✅ Подключение к Bybit успешно')
        print(f'  📊 Открытых позиций: {len(positions)}')
        
        for pos in positions:
            symbol = pos['symbol']
            side = pos['side']
            size = float(pos['size'])
            entry = float(pos['avgPrice'])
            pnl = float(pos['unrealisedPnl'])
            tp = pos.get('takeProfit', 'N/A')
            sl = pos.get('stopLoss', 'N/A')
            
            direction = 'LONG' if side == 'Buy' else 'SHORT'
            print(f'  📍 {symbol} {direction}')
            print(f'     Entry: ${entry:.6f} | PnL: ${pnl:.2f}')
            print(f'     TP: {tp} | SL: {sl}')
    else:
        print(f'  ❌ Ошибка API: {response["retMsg"]}')
        
except Exception as e:
    print(f'  ⚠️  Ошибка проверки позиций: {e}')

print()

# ═══════════════════════════════════════════════════════════════════
# 7. ПРОВЕРКА РОТАЦИИ ЛОГОВ
# ═══════════════════════════════════════════════════════════════════
print('🔄 7. ПРОВЕРКА РОТАЦИИ ЛОГОВ')
print('-'*80)

try:
    from log_rotator import LogRotator
    
    rotator = LogRotator(logs_dir='logs')
    
    print(f'  ✅ LogRotator инициализирован')
    print(f'  📊 Макс размер лога: {rotator.max_size_mb} MB')
    print(f'  📅 Хранение: {rotator.keep_days} дней')
    print(f'  📁 Директория: {rotator.logs_dir}')
    
    # Проверяем старые файлы
    old_files = rotator.get_old_files()
    if old_files:
        print(f'  ⚠️  Старых файлов для удаления: {len(old_files)}')
    else:
        print(f'  ✅ Нет старых файлов для удаления')
        
except Exception as e:
    print(f'  ⚠️  Ошибка проверки ротации: {e}')

print()

# ═══════════════════════════════════════════════════════════════════
# ИТОГОВЫЙ ОТЧЕТ
# ═══════════════════════════════════════════════════════════════════
print('='*80)
print('✅ ГЕНЕРАЛЬНАЯ ПРОВЕРКА ЗАВЕРШЕНА')
print('='*80)
