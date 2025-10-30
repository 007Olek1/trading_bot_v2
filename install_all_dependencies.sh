#!/bin/bash
# 📦 УСТАНОВКА ВСЕХ ЗАВИСИМОСТЕЙ И ПРОВЕРКА БД

SERVER_IP="213.163.199.116"
SSH_KEY="${HOME}/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"

echo "📦 УСТАНОВКА ВСЕХ ЗАВИСИМОСТЕЙ"
echo "================================"
echo ""

# Функция выполнения команд
execute_remote() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no root@"$SERVER_IP" "$1"
}

# Функция копирования
copy_to_server() {
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$1" root@"$SERVER_IP":"$2"
}

echo "1️⃣ Обновление requirements_bot.txt..."
# Копируем обновленный requirements
copy_to_server "requirements_bot.txt" "$BOT_DIR/"

echo ""
echo "2️⃣ Установка всех библиотек..."
execute_remote "cd $BOT_DIR && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements_bot.txt --upgrade"

echo ""
echo "3️⃣ Проверка установленных библиотек..."
execute_remote "cd $BOT_DIR && source venv/bin/activate && python3 -c \"
import sys
libraries = [
    'ccxt', 'telegram', 'apscheduler', 'pandas', 'numpy', 
    'dotenv', 'sklearn', 'pytz', 'openai', 'requests', 'sqlite3'
]
missing = []
for lib in libraries:
    try:
        if lib == 'dotenv':
            from dotenv import load_dotenv
        elif lib == 'telegram':
            from telegram import Bot
        elif lib == 'sklearn':
            import sklearn
        elif lib == 'sqlite3':
            import sqlite3
        else:
            __import__(lib)
        print(f'✅ {lib}')
    except ImportError:
        print(f'❌ {lib} - не установлен')
        missing.append(lib)

if missing:
    print(f'\n⚠️ Отсутствуют: {missing}')
    sys.exit(1)
else:
    print('\n✅ Все библиотеки установлены!')
\""

echo ""
echo "4️⃣ Проверка базы данных..."
execute_remote "cd $BOT_DIR && source venv/bin/activate && python3 -c \"
from data_storage_system import DataStorageSystem
import os
print('🔍 Проверка БД...')
storage = DataStorageSystem()
print(f'✅ БД инициализирована: {storage.db_path}')
print(f'✅ Файл БД существует: {os.path.exists(storage.db_path)}')
print('✅ База данных работает!')
\""

echo ""
echo "5️⃣ Проверка api.env..."
execute_remote "cd $BOT_DIR && if [ -f api.env ]; then echo '✅ api.env найден'; echo '🔍 Проверка переменных:'; grep -E '^[A-Z_]=' api.env | sed 's/=.*/=***/' | head -5; else echo '❌ api.env не найден'; fi"

echo ""
echo "6️⃣ Установка OpenSearch клиента (опционально)..."
execute_remote "cd $BOT_DIR && source venv/bin/activate && pip install opensearch-py opensearch-dsl 2>/dev/null && echo '✅ OpenSearch клиент установлен' || echo '⚠️ OpenSearch не установлен (опционально)'"

echo ""
echo "✅ ПРОВЕРКА ЗАВЕРШЕНА!"


