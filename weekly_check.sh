#!/bin/bash

# 📅 ЕЖЕНЕДЕЛЬНАЯ ПРОВЕРКА СИСТЕМЫ
# Запускается каждое воскресенье в 5:00

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📅 ЕЖЕНЕДЕЛЬНАЯ ПРОВЕРКА СИСТЕМЫ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Дата: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Переходим в директорию бота
cd /root/trading_bot_v4_mtf || exit 1

# Файл отчёта
REPORT_FILE="logs/weekly_report_$(date +%Y%m%d).txt"

# Функция для записи в отчёт и вывода
log_report() {
    echo -e "$1" | tee -a "$REPORT_FILE"
}

# Начало отчёта
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_report "📅 ЕЖЕНЕДЕЛЬНЫЙ ОТЧЁТ - $(date '+%Y-%m-%d')"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_report ""

# 1. СТАТУС БОТА
log_report "${BLUE}1. СТАТУС БОТА${NC}"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BOT_PID=$(pgrep -f "python.*main.py")
if [ -n "$BOT_PID" ]; then
    UPTIME=$(ps -p $BOT_PID -o etime= | tr -d ' ')
    CPU=$(ps -p $BOT_PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $BOT_PID -o %mem= | tr -d ' ')
    
    log_report "${GREEN}✅ Бот работает${NC}"
    log_report "   PID: $BOT_PID"
    log_report "   Uptime: $UPTIME"
    log_report "   CPU: ${CPU}%"
    log_report "   Memory: ${MEM}%"
else
    log_report "${RED}❌ Бот НЕ РАБОТАЕТ!${NC}"
fi
log_report ""

# 2. ИСПОЛЬЗОВАНИЕ РЕСУРСОВ
log_report "${BLUE}2. ИСПОЛЬЗОВАНИЕ РЕСУРСОВ${NC}"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Диск
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')

log_report "💾 Диск:"
log_report "   Использовано: ${DISK_USAGE}%"
log_report "   Свободно: ${DISK_FREE}"

if [ "$DISK_USAGE" -gt 80 ]; then
    log_report "${RED}   ⚠️ ВНИМАНИЕ: Диск заполнен более чем на 80%!${NC}"
fi
log_report ""

# Размер проекта
PROJECT_SIZE=$(du -sh . | cut -f1)
LOGS_SIZE=$(du -sh logs/ | cut -f1)

log_report "📊 Размеры:"
log_report "   Проект: $PROJECT_SIZE"
log_report "   Логи: $LOGS_SIZE"
log_report ""

# 3. СТАТИСТИКА СДЕЛОК ЗА НЕДЕЛЮ
log_report "${BLUE}3. СТАТИСТИКА СДЕЛОК ЗА НЕДЕЛЮ${NC}"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "logs/trades.json" ]; then
    # Подсчитываем сделки за последние 7 дней
    WEEK_AGO=$(date -d '7 days ago' +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d)
    TOTAL_TRADES=$(wc -l < logs/trades.json)
    
    log_report "📊 Всего сделок в базе: $TOTAL_TRADES"
    
    # Анализ через Python если доступен
    if command -v python3 &> /dev/null; then
        python3 << 'EOF' >> "$REPORT_FILE" 2>&1
import json
from datetime import datetime, timedelta

try:
    with open('logs/trades.json', 'r') as f:
        trades = [json.loads(line) for line in f]
    
    # Фильтруем за последнюю неделю
    week_ago = datetime.now() - timedelta(days=7)
    weekly_trades = [t for t in trades if datetime.fromisoformat(t.get('timestamp', '2020-01-01')) > week_ago]
    
    if weekly_trades:
        wins = sum(1 for t in weekly_trades if t.get('pnl_usd', 0) > 0)
        losses = sum(1 for t in weekly_trades if t.get('pnl_usd', 0) <= 0)
        total_pnl = sum(t.get('pnl_usd', 0) for t in weekly_trades)
        win_rate = (wins / len(weekly_trades) * 100) if weekly_trades else 0
        
        print(f"\n📈 За последние 7 дней:")
        print(f"   Сделок: {len(weekly_trades)}")
        print(f"   Прибыльных: {wins}")
        print(f"   Убыточных: {losses}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Общий PnL: ${total_pnl:.2f}")
    else:
        print("\n   Нет сделок за последнюю неделю")
except Exception as e:
    print(f"\n   Ошибка анализа: {e}")
EOF
    fi
else
    log_report "   Файл сделок не найден"
fi
log_report ""

# 4. ОШИБКИ В ЛОГАХ
log_report "${BLUE}4. КРИТИЧЕСКИЕ ОШИБКИ${NC}"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ERROR_COUNT=$(grep -c "ERROR\|CRITICAL" logs/trading_bot_v4.log 2>/dev/null || echo "0")
log_report "⚠️ Ошибок за неделю: $ERROR_COUNT"

if [ "$ERROR_COUNT" -gt 0 ]; then
    log_report ""
    log_report "Последние 5 ошибок:"
    grep "ERROR\|CRITICAL" logs/trading_bot_v4.log | tail -5 | while read line; do
        log_report "   $line"
    done
fi
log_report ""

# 5. РЕКОМЕНДАЦИИ
log_report "${BLUE}5. РЕКОМЕНДАЦИИ${NC}"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RECOMMENDATIONS=0

# Проверка размера логов
LOGS_SIZE_MB=$(du -sm logs/ | cut -f1)
if [ "$LOGS_SIZE_MB" -gt 50 ]; then
    log_report "${YELLOW}⚠️ Логи занимают более 50MB. Запустите cleanup.sh${NC}"
    RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

# Проверка свободного места
FREE_GB=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$FREE_GB" -lt 5 ]; then
    log_report "${RED}❌ Мало свободного места (< 5GB). Срочно нужна очистка!${NC}"
    RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

# Проверка количества ошибок
if [ "$ERROR_COUNT" -gt 50 ]; then
    log_report "${YELLOW}⚠️ Много ошибок в логах (> 50). Проверьте работу бота${NC}"
    RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

# Проверка uptime
if [ -n "$BOT_PID" ]; then
    UPTIME_HOURS=$(ps -p $BOT_PID -o etimes= | awk '{print int($1/3600)}')
    if [ "$UPTIME_HOURS" -lt 24 ]; then
        log_report "${YELLOW}⚠️ Бот работает менее 24ч. Возможны частые перезапуски${NC}"
        RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
    fi
fi

if [ "$RECOMMENDATIONS" -eq 0 ]; then
    log_report "${GREEN}✅ Всё в порядке! Рекомендаций нет${NC}"
fi
log_report ""

# 6. ДЕЙСТВИЯ
log_report "${BLUE}6. ВЫПОЛНЕННЫЕ ДЕЙСТВИЯ${NC}"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Создаём бэкап конфигурации
log_report "📦 Создание бэкапа конфигурации..."
./backup_config.sh >> "$REPORT_FILE" 2>&1
log_report "${GREEN}✅ Бэкап создан${NC}"
log_report ""

# Запускаем очистку если нужно
if [ "$LOGS_SIZE_MB" -gt 50 ]; then
    log_report "🧹 Запуск очистки..."
    ./cleanup.sh >> "$REPORT_FILE" 2>&1
    log_report "${GREEN}✅ Очистка выполнена${NC}"
fi
log_report ""

# Завершение
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_report "${GREEN}✅ ЕЖЕНЕДЕЛЬНАЯ ПРОВЕРКА ЗАВЕРШЕНА${NC}"
log_report "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_report ""
log_report "📄 Отчёт сохранён: $REPORT_FILE"

# Отправляем отчёт в Telegram
if [ -f "send_telegram_report.py" ]; then
    log_report ""
    log_report "📱 Отправка отчёта в Telegram..."
    python3 send_telegram_report.py "$REPORT_FILE"
fi

echo ""
echo "✅ Готово! Отчёт: $REPORT_FILE"
