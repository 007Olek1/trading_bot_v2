#!/bin/bash
# Скрипт для деплоя TradeGPT Scalper на сервер

set -e

echo "========================================="
echo "TradeGPT Scalper - Деплой на сервер"
echo "========================================="

# Параметры сервера
SERVER_USER="root"
SERVER_HOST="185.70.199.244"
SERVER_PATH="/opt/tradegpt_scalper"
SSH_KEY="~/.ssh/upcloud_trading_bot"

echo ""
echo "Сервер: $SERVER_USER@$SERVER_HOST"
echo "Путь: $SERVER_PATH"
echo ""

# Проверка SSH ключа
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH ключ не найден: $SSH_KEY"
    exit 1
fi

echo "✅ SSH ключ найден"

# Создание директории на сервере
echo ""
echo "📁 Создание директории на сервере..."
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_HOST" "mkdir -p $SERVER_PATH"

# Копирование файлов
echo ""
echo "📤 Копирование файлов..."
scp -i "$SSH_KEY" tradegpt_scalper.py "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"
scp -i "$SSH_KEY" bybit_api.py "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"
scp -i "$SSH_KEY" disco57_simple.py "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"
scp -i "$SSH_KEY" telegram_notifier.py "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"
scp -i "$SSH_KEY" requirements.txt "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"
scp -i "$SSH_KEY" .env.example "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"
scp -i "$SSH_KEY" README.md "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"

echo "✅ Файлы скопированы"

# Установка зависимостей
echo ""
echo "📦 Установка зависимостей на сервере..."
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_HOST" << 'EOF'
cd /opt/tradegpt_scalper

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "Установка Python3..."
    apt-get update
    apt-get install -y python3 python3-pip
fi

# Установка зависимостей
pip3 install -r requirements.txt

echo "✅ Зависимости установлены"
EOF

# Проверка .env файла
echo ""
echo "⚙️ Проверка конфигурации..."
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_HOST" << 'EOF'
cd /opt/tradegpt_scalper

if [ ! -f ".env" ]; then
    echo "⚠️  Файл .env не найден!"
    echo "Создайте файл .env на основе .env.example:"
    echo "  cd /opt/tradegpt_scalper"
    echo "  cp .env.example .env"
    echo "  nano .env"
    exit 1
fi

echo "✅ Файл .env найден"
EOF

# Создание systemd сервиса
echo ""
echo "🔧 Создание systemd сервиса..."
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_HOST" << 'EOF'
cat > /etc/systemd/system/tradegpt-scalper.service << 'SERVICE'
[Unit]
Description=TradeGPT Scalper Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/tradegpt_scalper
ExecStart=/usr/bin/python3 /opt/tradegpt_scalper/tradegpt_scalper.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/tradegpt_scalper/log.txt
StandardError=append:/opt/tradegpt_scalper/log.txt

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
echo "✅ Systemd сервис создан"
EOF

echo ""
echo "========================================="
echo "✅ Деплой завершен!"
echo "========================================="
echo ""
echo "Следующие шаги:"
echo ""
echo "1. Настройте .env файл на сервере:"
echo "   ssh -i $SSH_KEY $SERVER_USER@$SERVER_HOST"
echo "   cd $SERVER_PATH"
echo "   nano .env"
echo ""
echo "2. Запустите бота:"
echo "   systemctl start tradegpt-scalper"
echo ""
echo "3. Проверьте статус:"
echo "   systemctl status tradegpt-scalper"
echo ""
echo "4. Просмотр логов:"
echo "   tail -f $SERVER_PATH/log.txt"
echo ""
echo "5. Включить автозапуск:"
echo "   systemctl enable tradegpt-scalper"
echo ""
echo "========================================="
