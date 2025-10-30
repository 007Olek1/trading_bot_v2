#!/bin/bash
# Деплой и перезапуск бота на сервере

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"

echo "🚀 ДЕПЛОЙ И ПЕРЕЗАПУСК БОТА"
echo "=================================="
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
echo "📦 ПОДГОТОВКА ФАЙЛОВ..."

# Создаем список файлов для копирования
FILES_TO_COPY=(
    "super_bot_v4_mtf.py"
    "data_storage_system.py"
    "universal_learning_system.py"
    "smart_coin_selector.py"
    "advanced_indicators.py"
    "llm_monitor.py"
    "ai_ml_system.py"
    "telegram_commands_handler.py"
    "api_optimizer.py"
    "integrate_intelligent_agents.py"
    "intelligent_agents.py"
    "system_agents.py"
    "fed_event_manager.py"
    "requirements_bot.txt"
    ".env"
)

# Проверяем наличие файлов
for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ⚠️ $file - не найден (может быть не критично)"
    fi
done

echo ""
echo "📡 ПОДКЛЮЧЕНИЕ К СЕРВЕРУ $SERVER_IP..."

ssh $SSH_KEY_OPTION root@$SERVER_IP << SSHCOMMANDS
    set -e
    
    echo "📁 Создание структуры директорий..."
    mkdir -p $BOT_DIR
    mkdir -p $BOT_DIR/data
    mkdir -p $BOT_DIR/data/models
    mkdir -p $BOT_DIR/data/cache
    mkdir -p $BOT_DIR/data/storage
    mkdir -p $BOT_DIR/logs
    mkdir -p $BOT_DIR/logs/trading
    mkdir -p $BOT_DIR/logs/system
    mkdir -p $BOT_DIR/logs/ml
    
    echo "✅ Структура создана"
    echo ""
    
    echo "🔍 Проверка текущего состояния..."
    systemctl status trading-bot --no-pager 2>/dev/null || echo "   Сервис не запущен или не существует"
    ps aux | grep -E "python.*super_bot" | grep -v grep || echo "   Процессы бота не найдены"
    echo ""
    
    echo "🛑 Остановка старого бота..."
    systemctl stop trading-bot 2>/dev/null || true
    pkill -f "super_bot_v4_mtf.py" 2>/dev/null || true
    sleep 2
    echo "✅ Бот остановлен"
    echo ""
SSHCOMMANDS

echo ""
echo "📤 КОПИРОВАНИЕ ФАЙЛОВ НА СЕРВЕР..."

# Копируем основные файлы
for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        echo "  📤 Копирую $file..."
        scp $SSH_KEY_OPTION "$file" root@$SERVER_IP:$BOT_DIR/ 2>&1 | grep -v "Warning" || echo "    ✅ Скопировано"
    fi
done

echo ""
echo "⚙️ УСТАНОВКА И НАСТРОЙКА НА СЕРВЕРЕ..."

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    set -e
    cd /opt/bot
    
    echo "🐍 Проверка Python..."
    python3 --version
    
    echo ""
    echo "📦 Установка зависимостей..."
    if [ -f "requirements_bot.txt" ]; then
        pip3 install -q --upgrade pip
        pip3 install -q -r requirements_bot.txt 2>&1 | tail -5
        echo "✅ Зависимости установлены"
    else
        echo "⚠️ requirements_bot.txt не найден, используем базовые зависимости"
        pip3 install -q ccxt python-telegram-bot pandas numpy scikit-learn openai apscheduler python-dotenv
    fi
    
    echo ""
    echo "🔍 Проверка .env файла..."
    if [ -f ".env" ]; then
        echo "✅ .env найден"
        # Проверяем что есть необходимые ключи
        if grep -q "BYBIT_API_KEY" .env && grep -q "TELEGRAM" .env; then
            echo "✅ Ключи API присутствуют"
        else
            echo "⚠️ Некоторые ключи отсутствуют"
        fi
    else
        echo "⚠️ .env не найден! Бот может не запуститься"
    fi
    
    echo ""
    echo "📋 Создание systemd сервиса..."
    cat > /etc/systemd/system/trading-bot.service << 'SERVICEFILE'
[Unit]
Description=Advanced Trading Bot V4.0 MTF
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
    
    echo ""
    echo "✅ Все готово к запуску!"
SSHCOMMANDS

echo ""
echo "🚀 ЗАПУСК БОТА..."

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    systemctl enable trading-bot
    systemctl start trading-bot
    
    sleep 3
    
    echo ""
    echo "📊 СТАТУС БОТА:"
    systemctl status trading-bot --no-pager -l | head -15
    
    echo ""
    echo "📋 ПОСЛЕДНИЕ СТРОКИ ЛОГОВ:"
    tail -20 /opt/bot/logs/system/bot.log 2>/dev/null || tail -20 /opt/bot/super_bot_v4_mtf.log 2>/dev/null || echo "   Логи пока пусты"
SSHCOMMANDS

echo ""
echo "=" 
echo "✅ ДЕПЛОЙ ЗАВЕРШЕН!"
echo ""
echo "📋 Для проверки логов на сервере:"
echo "   ssh $SSH_KEY_OPTION root@$SERVER_IP 'tail -f /opt/bot/logs/system/bot.log'"
echo ""
echo "📊 Для проверки статуса:"
echo "   ssh $SSH_KEY_OPTION root@$SERVER_IP 'systemctl status trading-bot'"
echo ""


