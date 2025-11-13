#!/bin/bash
# 🚀 Скрипт для применения оптимизаций на сервере

SERVER_IP="185.70.199.244"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
BOT_DIR="/root/bybit_bot"

echo "🚀 ПРИМЕНЕНИЕ ОПТИМИЗАЦИЙ ТОРГОВЛИ НА СЕРВЕРЕ"
echo "=============================================="
echo ""

# 1. Остановить бота
echo "1️⃣ Останавливаем бота..."
ssh -i "$SSH_KEY" root@"$SERVER_IP" "cd $BOT_DIR && pkill -f 'python.*run_bot.py' && sleep 2 && echo '✅ Бот остановлен'"

# 2. Создать бэкап
echo ""
echo "2️⃣ Создаем бэкап..."
ssh -i "$SSH_KEY" root@"$SERVER_IP" "cd $BOT_DIR && mkdir -p backups/$(date +%Y%m%d_%H%M%S) && cp -r scripts src backups/\$(date +%Y%m%d_%H%M%S)/ && echo '✅ Бэкап создан'"

# 3. Копируем измененные файлы
echo ""
echo "3️⃣ Копируем оптимизированные файлы..."

scp -i "$SSH_KEY" scripts/run_bot.py root@"$SERVER_IP":"$BOT_DIR/scripts/" && echo "   ✅ run_bot.py"
scp -i "$SSH_KEY" src/bybit_bot/core/signals.py root@"$SERVER_IP":"$BOT_DIR/src/bybit_bot/core/" && echo "   ✅ signals.py"
scp -i "$SSH_KEY" src/bybit_bot/core/scanner.py root@"$SERVER_IP":"$BOT_DIR/src/bybit_bot/core/" && echo "   ✅ scanner.py"
scp -i "$SSH_KEY" src/bybit_bot/core/executor.py root@"$SERVER_IP":"$BOT_DIR/src/bybit_bot/core/" && echo "   ✅ executor.py"
scp -i "$SSH_KEY" src/bybit_bot/api/client.py root@"$SERVER_IP":"$BOT_DIR/src/bybit_bot/api/" && echo "   ✅ client.py"

# 4. Копируем smart_coin_selector если нужно
if [ -f "smart_coin_selector.py" ]; then
    scp -i "$SSH_KEY" smart_coin_selector.py root@"$SERVER_IP":"$BOT_DIR/" && echo "   ✅ smart_coin_selector.py"
fi

# 5. Запускаем бота
echo ""
echo "4️⃣ Запускаем бота с оптимизациями..."
ssh -i "$SSH_KEY" root@"$SERVER_IP" << 'EOF'
cd /root/bybit_bot
source venv/bin/activate
PYTHONPATH=src:. nohup python scripts/run_bot.py > logs/run.log 2>&1 &
sleep 3
if ps aux | grep -q '[p]ython.*run_bot.py'; then
    echo "✅ Бот запущен успешно"
else
    echo "❌ Ошибка запуска бота, проверьте логи: tail -50 logs/run.log"
fi
EOF

echo ""
echo "=============================================="
echo "✅ ОПТИМИЗАЦИИ ПРИМЕНЕНЫ!"
echo ""
echo "📊 Проверьте логи:"
echo "   ssh -i $SSH_KEY root@$SERVER_IP 'tail -f $BOT_DIR/logs/bot.log'"
echo ""

