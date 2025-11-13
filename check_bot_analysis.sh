#!/bin/bash
# 🔍 Быстрая проверка текущего анализа бота

SERVER_IP="185.70.199.244"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"
LOG_FILE="/root/bybit_bot/logs/bot.log"

echo "📊 ТЕКУЩИЙ СТАТУС АНАЛИЗА БОТА"
echo "=========================================="
echo ""

# Последние циклы
echo "🔄 ПОСЛЕДНИЕ ЦИКЛЫ АНАЛИЗА:"
ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -100 $LOG_FILE 2>/dev/null | grep -E 'TradingOrchestrator.*executed|run_cycle|cycle completed' | tail -5"
echo ""

# Анализируемые монеты
echo "💰 АНАЛИЗИРУЕМЫЕ МОНЕТЫ:"
ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -200 $LOG_FILE 2>/dev/null | grep -E 'BTC|ETH|SOL|XRP|BNB' | grep -E 'process|rank|signal|СИГНАЛ' | tail -10"
echo ""

# Сигналы
echo "🎯 СИГНАЛЫ И РЕШЕНИЯ:"
ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -200 $LOG_FILE 2>/dev/null | grep -E 'signal|СИГНАЛ|BUY|SELL|confidence|уверенность' | tail -10"
echo ""

# Ошибки
echo "⚠️ ОШИБКИ И ПРЕДУПРЕЖДЕНИЯ:"
ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -200 $LOG_FILE 2>/dev/null | grep -E 'ERROR|WARNING|⚠️|❌' | tail -5"
echo ""

# Последние 20 строк
echo "📝 ПОСЛЕДНИЕ СОБЫТИЯ:"
ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -20 $LOG_FILE 2>/dev/null"
echo ""

