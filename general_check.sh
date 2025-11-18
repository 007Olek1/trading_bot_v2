#!/bin/bash

echo '================================================================================'
echo '🔍 ГЕНЕРАЛЬНАЯ ПРОВЕРКА TRADING BOT V4.0 MTF'
echo '================================================================================'
echo ''

# 1. СТАТУС БОТА
echo '1️⃣  СТАТУС БОТА'
echo '--------------------------------------------------------------------------------'
BOT_PID=$(pgrep -f 'python.*main.py' | head -1)
if [ -n "$BOT_PID" ]; then
    UPTIME=$(ps -p $BOT_PID -o etime= | tr -d ' ')
    CPU=$(ps -p $BOT_PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $BOT_PID -o %mem= | tr -d ' ')
    echo "   ✅ Бот работает"
    echo "   PID: $BOT_PID"
    echo "   Uptime: $UPTIME"
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
else
    echo "   ❌ Бот НЕ РАБОТАЕТ!"
fi
echo ''

# 2. ПОЗИЦИИ НА БИРЖЕ
echo '2️⃣  ПОЗИЦИИ НА БИРЖЕ'
echo '--------------------------------------------------------------------------------'
/root/trading_bot_v4_mtf/venv/bin/python3 check_positions.py 2>/dev/null | grep -A 10 'Найдено открытых'
echo ''

# 3. ПОСЛЕДНИЕ ЛОГИ
echo '3️⃣  ПОСЛЕДНИЕ ЛОГИ (10 строк)'
echo '--------------------------------------------------------------------------------'
tail -10 logs/trading_bot_v4.log | sed 's/^/   /'
echo ''

# 4. ОШИБКИ
echo '4️⃣  ОШИБКИ ЗА ПОСЛЕДНИЙ ЧАС'
echo '--------------------------------------------------------------------------------'
ERROR_COUNT=$(grep -c 'ERROR\|CRITICAL' logs/trading_bot_v4.log 2>/dev/null || echo '0')
echo "   Всего ошибок: $ERROR_COUNT"
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "   Последние 3 ошибки:"
    grep 'ERROR\|CRITICAL' logs/trading_bot_v4.log | tail -3 | sed 's/^/      /'
fi
echo ''

# 5. ФАЙЛЫ
echo '5️⃣  ВАЖНЫЕ ФАЙЛЫ'
echo '--------------------------------------------------------------------------------'
FILES=("main.py" "config.py" "telegram_handler.py" ".env" "adaptive_params.py" "ml_predictor.py" "circuit_breaker.py")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "   ✅ $file ($SIZE)"
    else
        echo "   ❌ $file - НЕ НАЙДЕН!"
    fi
done
echo ''

# 6. CRON ЗАДАЧИ
echo '6️⃣  CRON АВТОМАТИЗАЦИЯ'
echo '--------------------------------------------------------------------------------'
CRON_COUNT=$(crontab -l 2>/dev/null | grep -c 'trading_bot_v4_mtf' || echo '0')
echo "   Активных задач: $CRON_COUNT"
if [ "$CRON_COUNT" -gt 0 ]; then
    crontab -l | grep 'trading_bot_v4_mtf' | sed 's/^/   /'
fi
echo ''

# 7. ДИСКОВОЕ ПРОСТРАНСТВО
echo '7️⃣  ДИСКОВОЕ ПРОСТРАНСТВО'
echo '--------------------------------------------------------------------------------'
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')
echo "   Использовано: ${DISK_USAGE}%"
echo "   Свободно: ${DISK_FREE}"

PROJECT_SIZE=$(du -sh . | cut -f1)
LOGS_SIZE=$(du -sh logs/ | cut -f1)
echo "   Размер проекта: $PROJECT_SIZE"
echo "   Размер логов: $LOGS_SIZE"
echo ''

# 8. БЭКАПЫ
echo '8️⃣  БЭКАПЫ'
echo '--------------------------------------------------------------------------------'
if [ -d "/root/trading_bot_backups" ]; then
    BACKUP_COUNT=$(ls -1 /root/trading_bot_backups/*.tar.gz 2>/dev/null | wc -l)
    echo "   Всего бэкапов: $BACKUP_COUNT"
    if [ "$BACKUP_COUNT" -gt 0 ]; then
        LATEST=$(ls -t /root/trading_bot_backups/*.tar.gz | head -1)
        LATEST_DATE=$(stat -c %y "$LATEST" 2>/dev/null | cut -d' ' -f1)
        LATEST_SIZE=$(du -h "$LATEST" | cut -f1)
        echo "   Последний: $(basename $LATEST)"
        echo "   Дата: $LATEST_DATE"
        echo "   Размер: $LATEST_SIZE"
    fi
else
    echo "   ⚠️  Директория бэкапов не найдена"
fi
echo ''

# 9. TELEGRAM
echo '9️⃣  TELEGRAM ИНТЕГРАЦИЯ'
echo '--------------------------------------------------------------------------------'
if grep -q 'TELEGRAM_BOT_TOKEN' .env 2>/dev/null; then
    echo "   ✅ Токен настроен"
else
    echo "   ❌ Токен не найден"
fi
if grep -q 'TELEGRAM_CHAT_ID' .env 2>/dev/null; then
    echo "   ✅ Chat ID настроен"
else
    echo "   ❌ Chat ID не найден"
fi
echo ''

# 10. МОНИТОРИНГ
echo '🔟 МОНИТОРИНГ И АЛЕРТЫ'
echo '--------------------------------------------------------------------------------'
if [ -f "uptime_monitor.py" ]; then
    echo "   ✅ Uptime монитор установлен"
else
    echo "   ❌ Uptime монитор не найден"
fi
if [ -f "critical_alerts.py" ]; then
    echo "   ✅ Критические алерты установлены"
else
    echo "   ❌ Критические алерты не найдены"
fi
echo ''

# ИТОГОВАЯ ОЦЕНКА
echo '================================================================================'
echo '✅ ИТОГОВАЯ ОЦЕНКА'
echo '================================================================================'

SCORE=0
MAX_SCORE=10

# Проверки
[ -n "$BOT_PID" ] && SCORE=$((SCORE + 2))
[ "$ERROR_COUNT" -lt 10 ] && SCORE=$((SCORE + 1))
[ -f "main.py" ] && SCORE=$((SCORE + 1))
[ -f ".env" ] && SCORE=$((SCORE + 1))
[ "$CRON_COUNT" -gt 0 ] && SCORE=$((SCORE + 1))
[ "$DISK_USAGE" -lt 80 ] && SCORE=$((SCORE + 1))
[ "$BACKUP_COUNT" -gt 0 ] && SCORE=$((SCORE + 1))
[ -f "uptime_monitor.py" ] && SCORE=$((SCORE + 1))
[ -f "critical_alerts.py" ] && SCORE=$((SCORE + 1))

echo "   Оценка: $SCORE/$MAX_SCORE"
echo ''

if [ "$SCORE" -ge 9 ]; then
    echo "   🎉 ОТЛИЧНО! Все системы работают идеально!"
elif [ "$SCORE" -ge 7 ]; then
    echo "   ✅ ХОРОШО! Основные системы работают."
elif [ "$SCORE" -ge 5 ]; then
    echo "   ⚠️  УДОВЛЕТВОРИТЕЛЬНО. Есть проблемы."
else
    echo "   ❌ КРИТИЧНО! Требуется немедленное вмешательство!"
fi

echo '================================================================================'
