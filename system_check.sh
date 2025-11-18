#!/bin/bash

# 🔍 Скрипт проверки системы Trading Bot V4.0 MTF
# Показывает полную информацию о состоянии бота

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 ПРОВЕРКА СИСТЕМЫ TRADING BOT V4.0 MTF"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. СТАТУС СЕРВИСА
echo -e "${BLUE}📊 1. СТАТУС СЕРВИСА${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
systemctl status trading-bot.service --no-pager | head -15
echo ""

# 2. ИСПОЛЬЗОВАНИЕ РЕСУРСОВ
echo -e "${BLUE}💻 2. ИСПОЛЬЗОВАНИЕ РЕСУРСОВ${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# CPU и Memory
BOT_PID=$(systemctl show -p MainPID trading-bot.service | cut -d= -f2)
if [ "$BOT_PID" != "0" ]; then
    echo -e "${GREEN}🔹 PID бота: $BOT_PID${NC}"
    ps -p $BOT_PID -o %cpu,%mem,vsz,rss,cmd --no-headers | awk '{printf "   CPU: %s%%\n   Memory: %s%% (%s KB)\n", $1, $2, $4}'
else
    echo -e "${RED}❌ Бот не запущен${NC}"
fi
echo ""

# Диск
echo -e "${GREEN}💾 Использование диска:${NC}"
df -h / | grep -E "Filesystem|/$"
echo ""

# 3. РАЗМЕРЫ ЛОГОВ
echo -e "${BLUE}📁 3. РАЗМЕРЫ ФАЙЛОВ И ЛОГОВ${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd /root/trading_bot_v4_mtf

echo -e "${GREEN}📊 Общий размер проекта:${NC}"
du -sh .

echo ""
echo -e "${GREEN}📝 Размеры логов:${NC}"
du -sh logs/
ls -lh logs/ | tail -n +2

if [ -d "testing/logs" ]; then
    echo ""
    echo -e "${GREEN}🧪 Логи тестирования:${NC}"
    du -sh testing/logs/ 2>/dev/null || echo "   Нет логов тестирования"
fi

echo ""

# 4. ПОСЛЕДНИЕ ЛОГИ
echo -e "${BLUE}📋 4. ПОСЛЕДНИЕ ЗАПИСИ В ЛОГЕ${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -n 10 logs/trading_bot_v4.log 2>/dev/null || echo "Нет логов"
echo ""

# 5. СДЕЛКИ
echo -e "${BLUE}💰 5. СТАТИСТИКА СДЕЛОК${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "logs/trades.json" ]; then
    TRADES_COUNT=$(wc -l < logs/trades.json)
    TRADES_SIZE=$(du -h logs/trades.json | cut -f1)
    echo -e "${GREEN}📊 Всего записей сделок: $TRADES_COUNT${NC}"
    echo -e "${GREEN}💾 Размер файла: $TRADES_SIZE${NC}"
else
    echo -e "${YELLOW}⚠️  Файл сделок не найден${NC}"
fi
echo ""

# 6. PYTHON ПРОЦЕССЫ
echo -e "${BLUE}🐍 6. PYTHON ПРОЦЕССЫ${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ps aux | grep "[p]ython.*main.py" | head -5
echo ""

# 7. СЕТЕВЫЕ ПОДКЛЮЧЕНИЯ
echo -e "${BLUE}🌐 7. АКТИВНЫЕ ПОДКЛЮЧЕНИЯ${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$BOT_PID" != "0" ]; then
    CONNECTIONS=$(lsof -p $BOT_PID -a -i 2>/dev/null | wc -l)
    echo -e "${GREEN}🔹 Активных подключений: $CONNECTIONS${NC}"
    lsof -p $BOT_PID -a -i 2>/dev/null | grep ESTABLISHED | head -5
else
    echo -e "${YELLOW}⚠️  Бот не запущен${NC}"
fi
echo ""

# 8. РЕКОМЕНДАЦИИ
echo -e "${BLUE}💡 8. РЕКОМЕНДАЦИИ${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Проверка размера логов
LOG_SIZE=$(du -sm logs/ | cut -f1)
if [ "$LOG_SIZE" -gt 50 ]; then
    echo -e "${YELLOW}⚠️  Логи занимают больше 50MB. Рекомендуется запустить cleanup.sh${NC}"
fi

# Проверка свободного места
FREE_SPACE=$(df / | tail -1 | awk '{print $4}')
if [ "$FREE_SPACE" -lt 5000000 ]; then
    echo -e "${RED}❌ Мало свободного места на диске! Срочно нужна очистка${NC}"
fi

# Проверка uptime бота
if [ "$BOT_PID" != "0" ]; then
    UPTIME=$(ps -p $BOT_PID -o etime= | tr -d ' ')
    echo -e "${GREEN}✅ Бот работает уже: $UPTIME${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Проверка завершена!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
