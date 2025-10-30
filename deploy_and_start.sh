#!/bin/bash
# 🚀 ДЕПЛОЙ И ЗАПУСК БОТА НА СЕРВЕР

SERVER_IP="213.163.199.116"
SSH_KEY_PATH="${HOME}/.ssh/upcloud_trading_bot"
BOT_DIR="/opt/bot"
SYMBOLIC_LINK="/root/trading_bot"

echo "🚀 ДЕПЛОЙ И ЗАПУСК БОТА НА СЕРВЕР"
echo "==================================="
echo ""

# Проверка SSH ключа
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "❌ SSH ключ не найден: $SSH_KEY_PATH"
    echo "📝 Используем стандартный ключ"
    SSH_KEY_PATH="${HOME}/.ssh/id_rsa"
fi

# Функция выполнения команд на сервере
execute_remote() {
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no root@"$SERVER_IP" "$1"
}

# Функция копирования файлов
copy_to_server() {
    scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$1" root@"$SERVER_IP":"$2"
}

echo "1️⃣ Остановка старого бота..."
execute_remote "systemctl stop trading-bot 2>/dev/null || pkill -f super_bot_v4_mtf.py || true"
sleep 2

echo ""
echo "2️⃣ Создание директорий на сервере..."
execute_remote "mkdir -p $BOT_DIR/logs $BOT_DIR/data"

echo ""
echo "3️⃣ Копирование файлов..."

# Основные модули
for file in *.py; do
    if [ -f "$file" ]; then
        echo "   📄 $file"
        copy_to_server "$file" "$BOT_DIR/"
    fi
done

# Важные файлы
if [ -f "requirements_bot.txt" ]; then
    copy_to_server "requirements_bot.txt" "$BOT_DIR/"
fi

if [ -f "api.env" ]; then
    copy_to_server "api.env" "$BOT_DIR/"
else
    echo "⚠️ api.env не найден - создайте его на сервере вручную!"
fi

echo ""
echo "4️⃣ Установка зависимостей..."
execute_remote "cd $BOT_DIR && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements_bot.txt"

echo ""
echo "5️⃣ Создание systemd сервиса..."
execute_remote "cat > /etc/systemd/system/trading-bot.service << 'EOF'
[Unit]
Description=Super Bot V4.0 PRO Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$BOT_DIR
Environment=\"PATH=$BOT_DIR/venv/bin:\$PATH\"
ExecStart=$BOT_DIR/venv/bin/python3 $BOT_DIR/super_bot_v4_mtf.py
Restart=always
RestartSec=10
StandardOutput=append:$BOT_DIR/logs/bot.log
StandardError=append:$BOT_DIR/logs/bot_error.log
Environment=\"PYTHONUNBUFFERED=1\"

[Install]
WantedBy=multi-user.target
EOF"

echo ""
echo "6️⃣ Перезагрузка systemd и запуск..."
execute_remote "systemctl daemon-reload && systemctl enable trading-bot && systemctl start trading-bot"

echo ""
echo "7️⃣ Проверка статуса..."
sleep 3
execute_remote "systemctl status trading-bot --no-pager | head -20"

echo ""
echo "8️⃣ Последние логи..."
execute_remote "tail -30 $BOT_DIR/logs/bot.log 2>/dev/null || echo 'Логи пока пустые'"

echo ""
echo "✅ ДЕПЛОЙ ЗАВЕРШЕН!"
echo ""
echo "📋 Команды для мониторинга:"
echo "   ssh -i $SSH_KEY_PATH root@$SERVER_IP"
echo "   systemctl status trading-bot"
echo "   tail -f $BOT_DIR/logs/bot.log"


