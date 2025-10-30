#!/bin/bash

# Развертывание бота на новый сервер 213.163.199.116

SERVER_IP="213.163.199.116"
SSH_KEY="~/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     🚀 РАЗВЕРТЫВАНИЕ БОТА НА НОВЫЙ СЕРВЕР                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 Шаг 1: Создание структуры директорий..."
ssh -i $SSH_KEY root@$SERVER_IP << 'ENDSSH'
mkdir -p /opt/bot
cd /opt/bot
mkdir -p logs data
echo "✅ Директории созданы"
ENDSSH

echo ""
echo "📋 Шаг 2: Установка зависимостей..."
ssh -i $SSH_KEY root@$SERVER_IP << 'ENDSSH'
apt-get update -qq
apt-get install -y python3 python3-pip python3-venv git
python3 -m pip install --upgrade pip
echo "✅ Зависимости установлены"
ENDSSH

echo ""
echo "📋 Шаг 3: Копирование файлов бота..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='*.md' \
    --exclude='*.txt' \
    --exclude='*.sh' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    /Users/aleksandrfilippov/Downloads/trading_bot/*.py \
    root@$SERVER_IP:/opt/bot/

echo ""
echo "📋 Шаг 4: Создание .env файла..."
echo "⚠️ Нужно будет создать .env файл с API ключами!"
echo "   На сервере: nano /opt/bot/.env"

echo ""
echo "✅ РАЗВЕРТЫВАНИЕ ЗАВЕРШЕНО!"
echo ""
echo "📝 Следующие шаги:"
echo "   1. Создайте /opt/bot/.env с API ключами"
echo "   2. Установите зависимости: cd /opt/bot && pip3 install -r requirements.txt"
echo "   3. Запустите бота: python3 super_bot_v4_mtf.py"


