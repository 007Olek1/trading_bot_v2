#!/bin/bash
# 📊 Мониторинг бота в реальном времени - детальный анализ рынка

SERVER_IP="185.70.199.244"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
LOG_FILE="/root/bybit_bot/logs/bot.log"

echo "🚀 МОНИТОРИНГ БОТА В РЕАЛЬНОМ ВРЕМЕНИ"
echo "=========================================="
echo "📡 Сервер: $SERVER_IP"
echo "📁 Лог: $LOG_FILE"
echo "⏰ Время: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "🔍 Фильтруем: циклы анализа, сигналы, монеты, ошибки"
echo "=========================================="
echo ""

# Функция для форматирования вывода
format_log() {
    while IFS= read -r line; do
        # Цвета для разных типов сообщений
        if echo "$line" | grep -qE "ERROR|❌"; then
            echo -e "\033[31m$line\033[0m"  # Красный для ошибок
        elif echo "$line" | grep -qE "WARNING|⚠️"; then
            echo -e "\033[33m$line\033[0m"  # Желтый для предупреждений
        elif echo "$line" | grep -qE "signal|СИГНАЛ|BUY|SELL|🎯"; then
            echo -e "\033[32m$line\033[0m"  # Зеленый для сигналов
        elif echo "$line" | grep -qE "cycle|цикл|анализ|ANALYSIS"; then
            echo -e "\033[36m$line\033[0m"  # Голубой для циклов
        elif echo "$line" | grep -qE "BTC|ETH|SOL|XRP|BNB"; then
            echo -e "\033[35m$line\033[0m"  # Фиолетовый для монет
        else
            echo "$line"
        fi
    done
}

# Мониторинг в реальном времени
ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -f $LOG_FILE 2>/dev/null" | \
    grep --line-buffered -E "cycle|Trading cycle|run_cycle|анализ|ANALYSIS|market|Рынок|символ|symbol|signal|СИГНАЛ|уверенность|confidence|BTC|ETH|SOL|XRP|BNB|ERROR|WARNING|⚠️|❌|✅.*символ|probabilities|ensemble|BUY|SELL|открыт|закрыт|position" | \
    format_log

