#!/bin/bash
# 📊 Просмотр логов бота в реальном времени на сервере

SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

echo "📊 ПРОСМОТР ЛОГОВ БОТА В РЕАЛЬНОМ ВРЕМЕНИ"
echo "=================================="
echo ""

echo "🔍 Режимы просмотра:"
echo "  1. Все логи (без фильтра)"
echo "  2. Только важные (торговля, ошибки, анализ)"
echo "  3. Только ошибки"
echo ""

read -p "Выберите режим (1-3, по умолчанию 2): " mode
mode=${mode:-2}

LOG_FILE="/opt/bot/logs/system/bot.log"
FALLBACK_LOG="/opt/bot/super_bot_v4_mtf.log"

echo ""
echo "📋 Проверка логов на сервере..."
echo ""

# Определяем какой лог использовать
LOG_TO_USE=""
if ssh -i "$SSH_KEY" -o ConnectTimeout=5 root@"$SERVER_IP" "test -f $LOG_FILE && -s $LOG_FILE"; then
    LOG_TO_USE="$LOG_FILE"
    echo "✅ Используется: $LOG_FILE"
elif ssh -i "$SSH_KEY" -o ConnectTimeout=5 root@"$SERVER_IP" "test -f $FALLBACK_LOG && -s $FALLBACK_LOG"; then
    LOG_TO_USE="$FALLBACK_LOG"
    echo "✅ Используется: $FALLBACK_LOG"
else
    echo "⚠️ Логи пока пусты, ждем первые записи..."
    LOG_TO_USE="$LOG_FILE"
fi

echo ""
echo "📏 Показываю последние 10 строк перед началом:"
ssh -i "$SSH_KEY" -o ConnectTimeout=5 root@"$SERVER_IP" "tail -10 $LOG_TO_USE 2>/dev/null || tail -10 $FALLBACK_LOG 2>/dev/null || echo '   Логи пока пусты'"

echo ""
echo "🔄 Начинаю следить за обновлениями в реальном времени..."
echo "⏹ Для остановки: Ctrl+C"
echo ""
echo "=================================="
echo ""

# Выбираем режим фильтрации
case $mode in
    1)
        # Все логи
        ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -f $LOG_TO_USE 2>/dev/null || tail -f $FALLBACK_LOG 2>/dev/null"
        ;;
    2)
        # Только важные события
        ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -f $LOG_TO_USE 2>/dev/null || tail -f $FALLBACK_LOG 2>/dev/null" | grep --line-buffered -E "СДЕЛКА|BUY|SELL|ОТКРЫТ|ЗАКРЫТ|🚀|📊.*анализ|Анализ рынка|trading_loop|ERROR|❌|✅.*инициализирован|V4\.0|символов|уверенность|MIN_CONFIDENCE|СИГНАЛ|BEARISH|BULLISH|INFO.*символ"
        ;;
    3)
        # Только ошибки
        ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -f $LOG_TO_USE 2>/dev/null || tail -f $FALLBACK_LOG 2>/dev/null" | grep --line-buffered -iE "ERROR|❌|WARNING|⚠️|Exception|Traceback"
        ;;
    *)
        echo "❌ Неверный режим. Используется режим 2 (важные события)"
        ssh -i "$SSH_KEY" root@"$SERVER_IP" "tail -f $LOG_TO_USE 2>/dev/null || tail -f $FALLBACK_LOG 2>/dev/null" | grep --line-buffered -E "СДЕЛКА|BUY|SELL|ОТКРЫТ|ЗАКРЫТ|🚀|📊.*анализ|ERROR|❌|✅"
        ;;
esac

