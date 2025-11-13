#!/bin/bash
# Скрипт развертывания нового бота на сервере

echo "🚀 Развертывание Bybit Futures Bot на сервере"
echo "=============================================="

# 1. Остановка старого бота
echo "1️⃣ Остановка старого бота..."
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "ps aux | grep python | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null"
sleep 2

# 2. Создание директории
echo "2️⃣ Создание директории на сервере..."
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "mkdir -p /root/bybit_futures_bot"

# 3. Загрузка файлов
echo "3️⃣ Загрузка файлов..."
scp -i ~/.ssh/upcloud_trading_bot bybit_futures_bot/config.py root@185.70.199.244:/root/bybit_futures_bot/
scp -i ~/.ssh/upcloud_trading_bot bybit_futures_bot/utils.py root@185.70.199.244:/root/bybit_futures_bot/
scp -i ~/.ssh/upcloud_trading_bot bybit_futures_bot/indicators.py root@185.70.199.244:/root/bybit_futures_bot/
scp -i ~/.ssh/upcloud_trading_bot bybit_futures_bot/main.py root@185.70.199.244:/root/bybit_futures_bot/
scp -i ~/.ssh/upcloud_trading_bot bybit_futures_bot/requirements.txt root@185.70.199.244:/root/bybit_futures_bot/

# 4. Копирование .env
echo "4️⃣ Копирование .env..."
scp -i ~/.ssh/upcloud_trading_bot keys/.env root@185.70.199.244:/root/bybit_futures_bot/.env 2>/dev/null || echo "⚠️ Файл .env не найден локально, нужно создать на сервере"

# 5. Установка зависимостей
echo "5️⃣ Установка зависимостей на сервере..."
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 << 'COMMANDS'
cd /root/bybit_futures_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "✅ Зависимости установлены"
COMMANDS

# 6. Запуск бота
echo "6️⃣ Запуск бота..."
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 << 'COMMANDS'
cd /root/bybit_futures_bot
source venv/bin/activate
nohup python main.py > /tmp/bot.log 2>&1 &
sleep 3
echo "✅ Бот запущен"
ps aux | grep "python main.py" | grep -v grep
COMMANDS

echo ""
echo "=============================================="
echo "✅ Развертывание завершено!"
echo "=============================================="
echo ""
echo "Проверка логов:"
echo "  ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 'tail -f /root/bybit_futures_bot/logs/bot.log'"
echo ""
echo "Проверка процесса:"
echo "  ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 'ps aux | grep python'"
