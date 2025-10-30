#!/bin/bash
# 🚀 ДЕПЛОЙ ОБНОВЛЕНИЙ НА СЕРВЕР

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"

echo "🚀 ДЕПЛОЙ ОБНОВЛЕНИЙ НА СЕРВЕР"
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
echo "🛑 ШАГ 1: ОСТАНОВКА БОТА..."

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    echo "Останавливаем сервис..."
    systemctl stop trading-bot 2>/dev/null || true
    
    echo "Завершаем все процессы бота..."
    pkill -f "super_bot_v4_mtf.py" 2>/dev/null || true
    pkill -f "python.*trading" 2>/dev/null || true
    
    sleep 2
    
    echo "Проверка..."
    if pgrep -f "super_bot" > /dev/null; then
        echo "⚠️ Процессы все еще запущены, принудительное завершение..."
        pkill -9 -f "super_bot" 2>/dev/null || true
    else
        echo "✅ Все процессы остановлены"
    fi
SSHCOMMANDS

echo ""
echo "📦 ШАГ 2: ПОДГОТОВКА ФАЙЛОВ ДЛЯ ЗАГРУЗКИ..."

FILES_TO_COPY=(
    "super_bot_v4_mtf.py"
    "api_optimizer.py"
    "intelligent_agents.py"
    "integrate_intelligent_agents.py"
    "system_agents.py"
    "data_storage_system.py"
    "universal_learning_system.py"
    "smart_coin_selector.py"
    "advanced_indicators.py"
    "llm_monitor.py"
    "ai_ml_system.py"
    "telegram_commands_handler.py"
    "requirements_bot.txt"
)

echo "Файлы для копирования:"
for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ⚠️ $file - не найден"
    fi
done

echo ""
echo "📤 ШАГ 3: КОПИРОВАНИЕ НА СЕРВЕР..."

for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        echo "  📤 Копирую $file..."
        scp $SSH_KEY_OPTION "$file" root@$SERVER_IP:$BOT_DIR/ 2>&1 | grep -v "Warning" || echo "    ✅ Скопировано"
    fi
done

echo ""
echo "⚙️ ШАГ 4: УСТАНОВКА И НАСТРОЙКА..."

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    cd /opt/bot
    
    echo "Проверка Python..."
    python3 --version
    
    echo ""
    echo "Обновление зависимостей..."
    if [ -f "requirements_bot.txt" ]; then
        pip3 install -q -r requirements_bot.txt 2>&1 | tail -5 || echo "  ⚠️ Некоторые зависимости не установились"
    fi
    
    echo ""
    echo "Создание структуры директорий..."
    mkdir -p data/models data/cache data/storage knowledge
    mkdir -p logs/trading logs/system logs/ml
    
    echo "✅ Структура создана"
    
    echo ""
    echo "Проверка .env файла..."
    if [ -f ".env" ]; then
        echo "✅ .env найден"
    else
        echo "⚠️ .env не найден - создайте его вручную!"
    fi
SSHCOMMANDS

echo ""
echo "🔄 ШАГ 5: ПЕРЕЗАПУСК БОТА..."

ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    systemctl daemon-reload
    systemctl enable trading-bot
    systemctl start trading-bot
    
    sleep 3
    
    echo ""
    echo "📊 СТАТУС БОТА:"
    systemctl status trading-bot --no-pager -l | head -15
    
    echo ""
    echo "📋 ПОСЛЕДНИЕ СТРОКИ ЛОГОВ:"
    tail -30 /opt/bot/super_bot_v4_mtf.log 2>/dev/null | tail -10 || tail -20 /opt/bot/logs/system/bot.log 2>/dev/null | tail -10 || echo "   Логи пока пусты"
SSHCOMMANDS

echo ""
echo "=" 
echo "✅ ДЕПЛОЙ ЗАВЕРШЕН!"
echo ""
echo "📋 Команды для проверки:"
echo "  Статус: ssh $SSH_KEY_OPTION root@$SERVER_IP 'systemctl status trading-bot'"
echo "  Логи: ssh $SSH_KEY_OPTION root@$SERVER_IP 'tail -f /opt/bot/super_bot_v4_mtf.log'"
echo ""


