#!/bin/bash

# 🧹 Скрипт очистки старых данных Trading Bot V4.0 MTF
# Автоматически удаляет старые логи и кэш

echo "🧹 Начинаю очистку старых данных..."

# Переходим в директорию проекта
cd /root/trading_bot_v4_mtf || exit 1

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Показываем размер ДО очистки
echo -e "${YELLOW}📊 Размер проекта ДО очистки:${NC}"
du -sh .

echo ""
echo "🗑️  Удаляю старые данные..."

# 1. Удаляем Python cache
echo -e "${GREEN}1. Очистка Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# 2. Удаляем старые сжатые логи (старше 30 дней)
echo -e "${GREEN}2. Удаление старых сжатых логов (>30 дней)...${NC}"
find logs/ -name "*.gz" -mtime +30 -delete 2>/dev/null
find testing/logs/ -name "*.gz" -mtime +30 -delete 2>/dev/null

# 3. Удаляем временные файлы
echo -e "${GREEN}3. Удаление временных файлов...${NC}"
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.bak" -delete 2>/dev/null
find . -type f -name "*~" -delete 2>/dev/null

# 4. Очистка логов тестирования старше 60 дней
echo -e "${GREEN}4. Очистка старых тестовых логов (>60 дней)...${NC}"
find testing/logs/ -name "*.log" -mtime +60 -delete 2>/dev/null
find testing/logs/ -name "*.json" -mtime +60 -delete 2>/dev/null

# 5. Ограничиваем размер основного лог-файла (если больше 10MB - обрезаем)
LOG_FILE="logs/trading_bot_v4.log"
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null)
    MAX_SIZE=$((10 * 1024 * 1024))  # 10 MB
    
    if [ "$LOG_SIZE" -gt "$MAX_SIZE" ]; then
        echo -e "${YELLOW}⚠️  Основной лог больше 10MB, обрезаю...${NC}"
        # Оставляем последние 5000 строк
        tail -n 5000 "$LOG_FILE" > "$LOG_FILE.tmp"
        mv "$LOG_FILE.tmp" "$LOG_FILE"
    fi
fi

echo ""
echo -e "${GREEN}✅ Очистка завершена!${NC}"
echo ""
echo -e "${YELLOW}📊 Размер проекта ПОСЛЕ очистки:${NC}"
du -sh .

echo ""
echo -e "${GREEN}💾 Свободное место на диске:${NC}"
df -h / | grep -E "Filesystem|/$"
