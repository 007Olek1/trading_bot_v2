#!/bin/bash
# 📊 Просмотр логов бота в реальном времени с фильтрацией

echo "📊 ПРОСМОТР ЛОГОВ БОТА В РЕАЛЬНОМ ВРЕМЕНИ"
echo "=" | head -c 70 && echo ""
echo ""

LOG_FILE="super_bot_v4_mtf.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Файл лога не найден: $LOG_FILE"
    exit 1
fi

echo "✅ Файл: $LOG_FILE"
echo "📏 Размер: $(ls -lh $LOG_FILE | awk '{print $5}')"
echo "📝 Строк: $(wc -l < $LOG_FILE)"
echo ""
echo "🔍 Режимы просмотра:"
echo "  1. Все логи (без фильтра)"
echo "  2. Только важные (торговля, ошибки, анализ)"
echo "  3. Только торговые операции"
echo ""
echo "💡 Текущий режим: Все логи"
echo ""
echo "⏹ Для остановки: Ctrl+C"
echo ""
echo "=" | head -c 70 && echo ""
echo ""

# Показываем последние 10 строк перед началом
echo "📋 Последние важные события:"
tail -100 "$LOG_FILE" | grep -E "СДЕЛКА|BUY|SELL|ОТКРЫТ|ЗАКРЫТ|🚀|📊.*анализ|Анализ рынка|trading_loop|ERROR|❌|✅.*инициализирован|V4\.0|символов|уверенность|СИГНАЛ" | tail -10

echo ""
echo "🔄 Начинаю следить за обновлениями..."
echo "=" | head -c 70 && echo ""
echo ""

# Запускаем tail -f с фильтрацией важных событий
tail -f "$LOG_FILE" | grep --line-buffered --color=always -E "СДЕЛКА|BUY|SELL|ОТКРЫТ|ЗАКРЫТ|🚀|📊.*анализ|Анализ рынка|trading_loop|ERROR|❌|✅.*инициализирован|V4\.0|символов|уверенность|MIN_CONFIDENCE|СИГНАЛ|BEARISH|BULLISH" || tail -f "$LOG_FILE"


