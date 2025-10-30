#!/bin/bash
# 📊 Мониторинг бота в реальном времени

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

echo "📊 МОНИТОРИНГ БОТА V4.0"
echo "=================================="
echo ""

# Функция для выполнения команд на сервере
execute_remote() {
    ssh -i "$SSH_KEY" -o ConnectTimeout=5 root@"$SERVER_IP" "$1"
}

# Проверка статуса бота
echo "1️⃣ СТАТУС БОТА:"
execute_remote "systemctl status trading-bot --no-pager | head -12"

echo ""
echo "2️⃣ ПРОЦЕССЫ:"
execute_remote "ps aux | grep -E 'super_bot_v4_mtf\.py' | grep -v grep"

echo ""
echo "3️⃣ ПОСЛЕДНИЕ СОБЫТИЯ В ЛОГАХ:"
echo "----------------------------------"
execute_remote "tail -50 /opt/bot/logs/system/bot.log 2>/dev/null | tail -20 || tail -50 /opt/bot/super_bot_v4_mtf.log 2>/dev/null | tail -20 || echo '   Логи пока пусты'"

echo ""
echo "4️⃣ ПОИСК ОШИБОК:"
execute_remote "tail -200 /opt/bot/logs/system/bot.log 2>/dev/null | grep -iE 'ERROR|❌|WARNING|⚠️' | tail -5 || tail -200 /opt/bot/super_bot_v4_mtf.log 2>/dev/null | grep -iE 'ERROR|❌|WARNING|⚠️' | tail -5 || echo '   ✅ Ошибок не найдено'"

echo ""
echo "5️⃣ ТОРГОВЫЕ ОПЕРАЦИИ:"
execute_remote "tail -200 /opt/bot/logs/system/bot.log 2>/dev/null | grep -iE 'СДЕЛКА|BUY|SELL|ОТКРЫТ|ЗАКРЫТ|📊.*анализ' | tail -10 || tail -200 /opt/bot/super_bot_v4_mtf.log 2>/dev/null | grep -iE 'СДЕЛКА|BUY|SELL|ОТКРЫТ|ЗАКРЫТ|📊.*анализ' | tail -10 || echo '   Нет торговых операций'"

echo ""
echo "6️⃣ РЕСУРСЫ СЕРВЕРА:"
execute_remote "echo '   Память:' && free -h | grep Mem && echo '   Диск:' && df -h / | tail -1"

echo ""
echo "=================================="
echo "✅ МОНИТОРИНГ ЗАВЕРШЕН"
echo ""
echo "📋 Для просмотра логов в реальном времени:"
echo "   ssh -i $SSH_KEY root@$SERVER_IP 'tail -f /opt/bot/logs/system/bot.log'"
echo ""
echo "📋 Или через super_bot_v4_mtf.log:"
echo "   ssh -i $SSH_KEY root@$SERVER_IP 'tail -f /opt/bot/super_bot_v4_mtf.log'"

