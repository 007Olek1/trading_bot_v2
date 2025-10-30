#!/bin/bash
# Остановка и проверка бота на сервере

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

echo "🛑 ОСТАНОВКА БОТА НА СЕРВЕРЕ"
echo "=================================="
echo ""

# Проверяем наличие SSH ключа
if [ ! -f "$SSH_KEY" ]; then
    echo "⚠️ SSH ключ не найден: $SSH_KEY"
    echo "💡 Попробуем подключиться без ключа (с паролем) или найдем другой ключ..."
    SSH_KEY_OPTION=""
else
    SSH_KEY_OPTION="-i $SSH_KEY"
    echo "✅ SSH ключ найден: $SSH_KEY"
fi

echo ""
echo "📡 Подключение к серверу $SERVER_IP..."
echo ""

# Останавливаем бота
ssh $SSH_KEY_OPTION root@$SERVER_IP << 'SSHCOMMANDS'
    echo "🛑 Остановка торгового бота..."
    systemctl stop trading-bot 2>/dev/null || true
    pkill -f "super_bot_v4_mtf.py" 2>/dev/null || true
    pkill -f "python.*trading" 2>/dev/null || true
    
    echo ""
    echo "✅ Бот остановлен"
    echo ""
    echo "📊 Проверка процессов:"
    ps aux | grep -E "python|trading|bot" | grep -v grep || echo "   Процессы бота не найдены"
    
    echo ""
    echo "📁 Проверка структуры:"
    ls -la /opt/bot/ 2>/dev/null || echo "   /opt/bot/ не найден"
    ls -la /root/trading_bot/ 2>/dev/null || echo "   /root/trading_bot/ не найден"
    
    echo ""
    echo "💾 Проверка БД:"
    ls -lh /opt/bot/trading_data.db 2>/dev/null || ls -lh /root/trading_bot/data/trading_data.db 2>/dev/null || echo "   БД не найдена"
SSHCOMMANDS

echo ""
echo "✅ Проверка завершена"
