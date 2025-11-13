#!/bin/bash
# Скрипт для перезапуска бота на сервере

set -e

echo "🔍 Проверка процессов..."

# Остановить все экземпляры бота и монитора
pkill -f super_bot_v4_mtf.py || true
pkill -f monitor_trailing_tp_generic.py || true
pkill -f monitor_trailing_tp_all.py || true

sleep 2

echo "✅ Процессы остановлены"

# Проверка: убедиться, что процессы остановлены
RUNNING=$(ps aux | grep -E 'super_bot_v4_mtf.py|monitor_trailing' | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "⚠️  Предупреждение: еще остались процессы, принудительно завершаю..."
    pkill -9 -f super_bot_v4_mtf.py || true
    pkill -9 -f monitor_trailing_tp_generic.py || true
    sleep 1
fi

# Загрузить переменные окружения
if [ -f /opt/bot/.env ]; then
    set -a
    . /opt/bot/.env
    set +a
fi

if [ -f /opt/bot/api.env ]; then
    set -a
    . /opt/bot/api.env
    set +a
fi

# Создать директорию для логов, если не существует
mkdir -p /opt/bot/logs/system

# Запустить монитор трейлинг TP/SL
echo "🚀 Запуск монитора трейлинг TP/SL..."
nohup python3 /opt/bot/monitor_trailing_tp_generic.py >> /opt/bot/logs/system/trailing_generic.log 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > /opt/bot/logs/system/trailing_generic.pid
echo "✅ Монитор запущен (PID: $MONITOR_PID)"

sleep 2

# Запустить основной бот
echo "🚀 Запуск основного бота..."
nohup python3 /opt/bot/super_bot_v4_mtf.py >> /opt/bot/logs/system/bot.log 2>&1 &
BOT_PID=$!
echo $BOT_PID > /opt/bot/logs/system/bot.pid
echo "✅ Бот запущен (PID: $BOT_PID)"

sleep 3

# Проверка запущенных процессов
echo ""
echo "📊 Текущие процессы:"
ps aux | grep -E 'super_bot_v4_mtf.py|monitor_trailing_tp_generic.py' | grep -v grep || echo "❌ Процессы не найдены!"

echo ""
echo "📋 Последние строки лога бота:"
tail -n 20 /opt/bot/logs/system/bot.log 2>/dev/null || echo "Лог пуст или не найден"

echo ""
echo "📋 Последние строки лога монитора:"
tail -n 10 /opt/bot/logs/system/trailing_generic.log 2>/dev/null || echo "Лог пуст или не найден"

echo ""
echo "✅ Готово! Бот и монитор должны работать."








