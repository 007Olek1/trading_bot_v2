#!/bin/bash
# Скрипт для остановки бота, загрузки обновлений, тестирования и перезапуска

set -e

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"
REMOTE_USER="root"

echo "=========================================="
echo "🚀 ОБНОВЛЕНИЕ БОТА С АДАПТИВНЫМ MIN_CONFIDENCE"
echo "=========================================="
echo ""

# 1. Остановка бота на сервере
echo "📌 Шаг 1: Остановка бота на сервере..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" << 'EOF'
    systemctl stop trading_bot || echo "⚠️ Сервис уже остановлен или не существует"
    pkill -f "super_bot_v4_mtf.py" || echo "⚠️ Процесс не найден"
    sleep 2
    echo "✅ Бот остановлен"
EOF

# 2. Загрузка обновленных файлов
echo ""
echo "📌 Шаг 2: Загрузка обновленных файлов на сервер..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    super_bot_v4_mtf.py \
    ADAPTIVE_CONFIDENCE_EXPLANATION.md \
    requirements_bot.txt \
    "$REMOTE_USER@$SERVER_IP:$BOT_DIR/"

echo "✅ Файлы загружены"

# 3. Установка зависимостей (если нужно)
echo ""
echo "📌 Шаг 3: Проверка зависимостей..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" << 'EOF'
    cd /opt/bot
    source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
    
    # Проверяем основные библиотеки
    python3 -c "import ccxt; import numpy; import pandas; import talib; print('✅ Основные библиотеки установлены')" || {
        echo "📦 Устанавливаем зависимости..."
        pip install --upgrade pip
        pip install -r requirements_bot.txt
    }
    
    echo "✅ Зависимости проверены"
EOF

# 4. Синтаксическая проверка
echo ""
echo "📌 Шаг 4: Синтаксическая проверка кода..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" << 'EOF'
    cd /opt/bot
    source venv/bin/activate
    python3 -m py_compile super_bot_v4_mtf.py && echo "✅ Синтаксис корректен" || {
        echo "❌ Ошибка синтаксиса!"
        exit 1
    }
EOF

# 5. Проверка .env файла
echo ""
echo "📌 Шаг 5: Проверка конфигурации..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" << 'EOF'
    if [ -f /opt/bot/.env ]; then
        echo "✅ .env файл найден"
        grep -q "BYBIT_API_KEY" /opt/bot/.env && echo "✅ BYBIT_API_KEY настроен" || echo "⚠️ BYBIT_API_KEY не найден"
        grep -q "TELEGRAM" /opt/bot/.env && echo "✅ TELEGRAM настроен" || echo "⚠️ TELEGRAM не найден"
    else
        echo "❌ .env файл не найден!"
        exit 1
    fi
EOF

# 6. Запуск бота
echo ""
echo "📌 Шаг 6: Запуск бота..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" << 'EOF'
    cd /opt/bot
    
    # Проверяем/создаем systemd сервис
    if [ ! -f /etc/systemd/system/trading_bot.service ]; then
        cat > /etc/systemd/system/trading_bot.service << SERVICE
[Unit]
Description=Advanced Trading Bot V4.0
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bot
Environment="PATH=/opt/bot/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/bot/venv/bin/python3 /opt/bot/super_bot_v4_mtf.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/bot/super_bot_v4_mtf.log
StandardError=append:/opt/bot/super_bot_v4_mtf.log

[Install]
WantedBy=multi-user.target
SERVICE
        systemctl daemon-reload
    fi
    
    # Запускаем бота
    systemctl start trading_bot
    systemctl enable trading_bot
    
    sleep 3
    
    # Проверяем статус
    if systemctl is-active --quiet trading_bot; then
        echo "✅ Бот успешно запущен"
        systemctl status trading_bot --no-pager | head -10
    else
        echo "❌ Ошибка запуска бота!"
        systemctl status trading_bot --no-pager
        exit 1
    fi
EOF

# 7. Проверка логов (первые 30 строк)
echo ""
echo "📌 Шаг 7: Проверка логов (последние 30 строк)..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" << 'EOF'
    echo "--- ПОСЛЕДНИЕ СТРОКИ ЛОГОВ ---"
    tail -30 /opt/bot/super_bot_v4_mtf.log 2>/dev/null || echo "⚠️ Логи пока пусты"
    echo ""
    echo "--- ПОИСК ОШИБОК ---"
    tail -100 /opt/bot/super_bot_v4_mtf.log 2>/dev/null | grep -i "error\|exception\|traceback" | tail -5 || echo "✅ Ошибок не найдено"
EOF

echo ""
echo "=========================================="
echo "✅ ОБНОВЛЕНИЕ ЗАВЕРШЕНО"
echo "=========================================="
echo ""
echo "📊 Проверить логи в реальном времени:"
echo "   ssh -i $SSH_KEY $REMOTE_USER@$SERVER_IP 'tail -f /opt/bot/super_bot_v4_mtf.log'"
echo ""
echo "🛑 Остановить бота:"
echo "   ssh -i $SSH_KEY $REMOTE_USER@$SERVER_IP 'systemctl stop trading_bot'"
echo ""
echo "🔄 Перезапустить бота:"
echo "   ssh -i $SSH_KEY $REMOTE_USER@$SERVER_IP 'systemctl restart trading_bot'"


