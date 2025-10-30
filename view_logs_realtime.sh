#!/bin/bash
# 📊 Скрипт просмотра логов бота в реальном времени

echo "🔍 Поиск логов бота..."
echo ""

# Проверяем различные источники логов
LOG_SOURCES=(
    "nohup.out"
    "bot.log"
    "telegram_trading_bot.log"
    "trading_bot.log"
    ".bot.log"
)

FOUND_LOG=""

# Ищем существующий лог файл
for log_file in "${LOG_SOURCES[@]}"; do
    if [ -f "$log_file" ]; then
        FOUND_LOG="$log_file"
        echo "✅ Найден лог файл: $log_file"
        break
    fi
done

# Если лог файл найден
if [ -n "$FOUND_LOG" ]; then
    echo ""
    echo "📊 Просмотр логов в реальном времени:"
    echo "   Файл: $FOUND_LOG"
    echo "   Для остановки: Ctrl+C"
    echo ""
    echo "=" | head -c 70 && echo ""
    tail -f "$FOUND_LOG"
else
    # Если нет файла лога, смотрим процесс бота
    BOT_PID=$(ps aux | grep "super_bot_v4_mtf.py" | grep -v grep | awk '{print $2}' | head -1)
    
    if [ -n "$BOT_PID" ]; then
        echo "✅ Бот запущен (PID: $BOT_PID)"
        echo ""
        echo "📊 Логи выводятся в консоль. Создайте файл лога:"
        echo ""
        echo "   # Остановите текущий бот"
        echo "   kill $BOT_PID"
        echo ""
        echo "   # Запустите с логированием в файл"
        echo "   nohup python3 super_bot_v4_mtf.py > bot.log 2>&1 &"
        echo ""
        echo "   # Затем запустите этот скрипт снова"
        echo ""
        echo "🔍 Или посмотрите последние логи процесса:"
        echo ""
        # Пробуем посмотреть через lsof
        if command -v lsof >/dev/null 2>&1; then
            echo "📋 Файлы открытые процессом:"
            lsof -p "$BOT_PID" 2>/dev/null | grep -E "log|txt|out" | head -10
        fi
    else
        echo "❌ Бот не запущен"
        echo ""
        echo "💡 Запустите бота с логированием:"
        echo "   nohup python3 super_bot_v4_mtf.py > bot.log 2>&1 &"
        echo ""
        echo "   Затем запустите этот скрипт снова для просмотра логов"
    fi
fi


