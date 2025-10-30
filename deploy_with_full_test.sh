#!/bin/bash
# 🚀 ПОЛНЫЙ ДЕПЛОЙ С ТЕСТИРОВАНИЕМ

set -e

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"

echo "🚀 ПОЛНЫЙ ДЕПЛОЙ С ТЕСТИРОВАНИЕМ"
echo "=================================="
echo ""

# Шаг 1: Локальное тестирование
echo "📋 ШАГ 1: ЛОКАЛЬНОЕ ТЕСТИРОВАНИЕ"
echo "---------------------------------"
./test_before_deploy.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Локальные тесты провалились! Прерываем деплой."
    exit 1
fi

echo ""
echo "✅ Локальные тесты пройдены!"
echo ""

# Проверка SSH ключа
if [ ! -f "$SSH_KEY" ]; then
    SSH_KEY_OPTION=""
    echo "⚠️ Используется подключение без ключа"
else
    SSH_KEY_OPTION="-i $SSH_KEY"
    echo "✅ Используется SSH ключ: $SSH_KEY"
fi

echo ""
echo "📡 ШАГ 2: ПОДКЛЮЧЕНИЕ К СЕРВЕРУ"
echo "--------------------------------"

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    echo "🔍 Проверка текущего состояния..."
    
    # Проверка процессов бота
    BOT_PROCESSES=$(ps aux | grep -E "python.*super_bot" | grep -v grep || true)
    if [ -n "$BOT_PROCESSES" ]; then
        echo "⚠️ Найдены запущенные процессы бота:"
        echo "$BOT_PROCESSES"
    else
        echo "✅ Процессы бота не найдены"
    fi
    
    # Проверка systemd сервиса
    if systemctl is-active --quiet trading-bot 2>/dev/null; then
        echo "⚠️ Сервис trading-bot активен"
    else
        echo "✅ Сервис trading-bot не активен"
    fi
    
    echo ""
    echo "🛑 Остановка всех процессов бота..."
    
    # Остановка systemd сервиса
    systemctl stop trading-bot 2>/dev/null || true
    systemctl disable trading-bot 2>/dev/null || true
    
    # Принудительная остановка процессов
    pkill -f "super_bot_v4_mtf.py" 2>/dev/null || true
    pkill -f "python.*trading" 2>/dev/null || true
    
    sleep 3
    
    # Проверка что все остановлено
    REMAINING=$(ps aux | grep -E "python.*super_bot" | grep -v grep || true)
    if [ -n "$REMAINING" ]; then
        echo "⚠️ Некоторые процессы все еще работают, принудительно завершаем..."
        pkill -9 -f "super_bot_v4_mtf.py" 2>/dev/null || true
        sleep 2
    fi
    
    echo "✅ Все процессы остановлены"
SSHCOMMANDS

echo ""
echo "📤 ШАГ 3: ЗАГРУЗКА ФАЙЛОВ НА СЕРВЕР"
echo "-----------------------------------"

FILES_TO_COPY=(
    "super_bot_v4_mtf.py"
    "smart_coin_selector.py"
    "probability_calculator.py"
    "strategy_evaluator.py"
    "realism_validator.py"
    "telegram_commands_handler.py"
    "advanced_indicators.py"
    "llm_monitor.py"
    "ai_ml_system.py"
    "api_optimizer.py"
    "integrate_intelligent_agents.py"
    "intelligent_agents.py"
    "system_agents.py"
    "fed_event_manager.py"
    "adaptive_parameters.py"
    "adaptive_trading_system.py"
    "data_storage_system.py"
    "universal_learning_system.py"
    "advanced_ml_system.py"
    "advanced_manipulation_detector.py"
    "requirements_bot.txt"
)

for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        echo "  📤 Копирую $file..."
        scp $SSH_KEY_OPTION "$file" root@$SERVER_IP:$BOT_DIR/ 2>&1 | grep -v "Warning" || true
    else
        echo "  ⚠️ $file не найден, пропускаем"
    fi
done

echo ""
echo "⚙️ ШАГ 4: УСТАНОВКА И НАСТРОЙКА"
echo "-------------------------------"

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    set -e
    cd /opt/bot
    
    echo "📁 Создание структуры директорий..."
    mkdir -p data/models data/cache data/storage
    mkdir -p logs/trading logs/system logs/ml
    echo "✅ Структура создана"
    
    echo ""
    echo "🐍 Проверка Python..."
    python3 --version
    
    echo ""
    echo "📦 Установка зависимостей..."
    if [ -f "requirements_bot.txt" ]; then
        pip3 install -q --upgrade pip
        pip3 install -q -r requirements_bot.txt
        echo "✅ Зависимости установлены"
    else
        echo "⚠️ requirements_bot.txt не найден, устанавливаем базовые..."
        pip3 install -q ccxt python-telegram-bot pandas numpy scikit-learn openai apscheduler python-dotenv pytz
    fi
    
    echo ""
    echo "🔍 Проверка конфигурации..."
    if [ -f ".env" ]; then
        echo "✅ .env найден"
        if grep -q "BYBIT_API_KEY" .env && grep -q "TELEGRAM" .env; then
            echo "✅ API ключи присутствуют"
        else
            echo "⚠️ Некоторые ключи могут отсутствовать"
        fi
    else
        echo "⚠️ .env не найден! Убедитесь что он скопирован отдельно"
    fi
    
    echo ""
    echo "📋 Создание systemd сервиса..."
    cat > /etc/systemd/system/trading-bot.service << 'SERVICEFILE'
[Unit]
Description=Advanced Trading Bot V4.0 MTF - Real Trading
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bot
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 /opt/bot/super_bot_v4_mtf.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/bot/logs/system/bot.log
StandardError=append:/opt/bot/logs/system/bot_error.log

[Install]
WantedBy=multi-user.target
SERVICEFILE
    
    systemctl daemon-reload
    echo "✅ Сервис создан"
SSHCOMMANDS

echo ""
echo "🧪 ШАГ 5: ТЕСТИРОВАНИЕ НА СЕРВЕРЕ"
echo "---------------------------------"

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    cd /opt/bot
    
    echo "🔍 Проверка синтаксиса Python..."
    python3 -m py_compile super_bot_v4_mtf.py && echo "✅ Синтаксис корректен" || echo "❌ Ошибка синтаксиса"
    
    echo ""
    echo "📦 Проверка импортов..."
    python3 << 'PYTHON_CHECK'
import sys
errors = 0

try:
    from smart_coin_selector import SmartCoinSelector
    print("  ✅ SmartCoinSelector")
except Exception as e:
    print(f"  ❌ SmartCoinSelector: {e}")
    errors += 1

try:
    from probability_calculator import ProbabilityCalculator
    print("  ✅ ProbabilityCalculator")
except Exception as e:
    print(f"  ❌ ProbabilityCalculator: {e}")
    errors += 1

sys.exit(errors)
PYTHON_CHECK
    
    if [ $? -eq 0 ]; then
        echo "✅ Импорты работают"
    else
        echo "❌ Ошибки импортов!"
        exit 1
    fi
SSHCOMMANDS

echo ""
echo "🚀 ШАГ 6: ЗАПУСК БОТА"
echo "---------------------"

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    cd /opt/bot
    
    echo "🔧 Включение автозапуска..."
    systemctl enable trading-bot
    
    echo ""
    echo "▶️ Запуск сервиса..."
    systemctl start trading-bot
    
    sleep 5
    
    echo ""
    echo "📊 СТАТУС СЕРВИСА:"
    systemctl status trading-bot --no-pager -l | head -20
    
    echo ""
    echo "📋 ПОСЛЕДНИЕ СТРОКИ ЛОГОВ:"
    if [ -f "logs/system/bot.log" ]; then
        tail -30 logs/system/bot.log
    elif [ -f "super_bot_v4_mtf.log" ]; then
        tail -30 super_bot_v4_mtf.log
    else
        echo "  Логи пока пусты, подождите несколько секунд..."
    fi
    
    echo ""
    echo "🔍 Проверка процессов..."
    ps aux | grep -E "python.*super_bot" | grep -v grep || echo "  Процессы не найдены (может быть нормально на старте)"
SSHCOMMANDS

echo ""
echo "✅ ШАГ 7: ФИНАЛЬНАЯ ПРОВЕРКА"
echo "----------------------------"

sleep 10

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    echo "📊 Финальный статус:"
    
    if systemctl is-active --quiet trading-bot; then
        echo "  ✅ Сервис АКТИВЕН"
    else
        echo "  ❌ Сервис НЕ АКТИВЕН"
        echo ""
        echo "  📋 Последние ошибки:"
        tail -20 logs/system/bot_error.log 2>/dev/null || echo "    Логи ошибок пусты"
        exit 1
    fi
    
    echo ""
    echo "  📋 Последние логи (последние 20 строк):"
    tail -20 logs/system/bot.log 2>/dev/null || tail -20 super_bot_v4_mtf.log 2>/dev/null || echo "    Логи пусты"
    
    echo ""
    echo "  🔍 Запущенные процессы:"
    ps aux | grep -E "python.*super_bot" | grep -v grep || echo "    Процессы не найдены"
SSHCOMMANDS

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ ДЕПЛОЙ ЗАВЕРШЕН УСПЕШНО!"
    echo "=================================="
    echo ""
    echo "📋 Команды для мониторинга:"
    echo "   Логи в реальном времени:"
    echo "   ssh $SSH_KEY_OPTION root@$SERVER_IP 'tail -f /opt/bot/logs/system/bot.log'"
    echo ""
    echo "   Статус сервиса:"
    echo "   ssh $SSH_KEY_OPTION root@$SERVER_IP 'systemctl status trading-bot'"
    echo ""
    echo "   Перезапуск (если нужно):"
    echo "   ssh $SSH_KEY_OPTION root@$SERVER_IP 'systemctl restart trading-bot'"
    echo ""
else
    echo ""
    echo "❌ ДЕПЛОЙ ЗАВЕРШИЛСЯ С ОШИБКАМИ!"
    echo "   Проверьте логи на сервере для деталей."
    exit 1
fi

