#!/bin/bash

# 📦 АВТОМАТИЧЕСКИЙ БЭКАП КОНФИГУРАЦИИ
# Создаёт резервную копию всех важных файлов

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 АВТОМАТИЧЕСКИЙ БЭКАП КОНФИГУРАЦИИ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Переходим в директорию бота
cd /root/trading_bot_v4_mtf || exit 1

# Директория для бэкапов
BACKUP_DIR="/root/trading_bot_backups"
mkdir -p "$BACKUP_DIR"

# Имя бэкапа с датой
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="bot_config_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

echo -e "${YELLOW}📁 Создание директории бэкапа...${NC}"
mkdir -p "$BACKUP_PATH"

# Список файлов для бэкапа
FILES_TO_BACKUP=(
    "config.py"
    ".env"
    "requirements.txt"
    "trading-bot.service"
    "logrotate.conf"
)

# Список директорий для бэкапа
DIRS_TO_BACKUP=(
    "logs/trades.json"
)

echo -e "${YELLOW}📄 Копирование файлов конфигурации...${NC}"

# Копируем файлы
for file in "${FILES_TO_BACKUP[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_PATH/"
        echo -e "${GREEN}   ✅ $file${NC}"
    else
        echo -e "${RED}   ⚠️  $file не найден${NC}"
    fi
done

# Копируем важные файлы из логов
echo -e "${YELLOW}📊 Копирование данных сделок...${NC}"
for file in "${DIRS_TO_BACKUP[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_PATH/"
        echo -e "${GREEN}   ✅ $file${NC}"
    fi
done

# Создаём архив
echo -e "${YELLOW}🗜️  Создание архива...${NC}"
cd "$BACKUP_DIR" || exit 1
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"

if [ $? -eq 0 ]; then
    # Удаляем временную директорию
    rm -rf "$BACKUP_NAME"
    
    BACKUP_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
    echo -e "${GREEN}✅ Архив создан: ${BACKUP_NAME}.tar.gz (${BACKUP_SIZE})${NC}"
else
    echo -e "${RED}❌ Ошибка создания архива${NC}"
    exit 1
fi

# Создаём метаданные
echo -e "${YELLOW}📝 Создание метаданных...${NC}"
cat > "${BACKUP_NAME}.meta" << EOF
Backup Date: $(date '+%Y-%m-%d %H:%M:%S')
Bot Version: V4.0 MTF
Files Backed Up:
$(for file in "${FILES_TO_BACKUP[@]}"; do echo "  - $file"; done)
$(for file in "${DIRS_TO_BACKUP[@]}"; do echo "  - $file"; done)
Backup Size: $BACKUP_SIZE
EOF

echo -e "${GREEN}✅ Метаданные созданы${NC}"

# Удаляем старые бэкапы (храним последние 30)
echo -e "${YELLOW}🗑️  Удаление старых бэкапов...${NC}"
BACKUP_COUNT=$(ls -1 bot_config_*.tar.gz 2>/dev/null | wc -l)

if [ "$BACKUP_COUNT" -gt 30 ]; then
    OLD_BACKUPS=$((BACKUP_COUNT - 30))
    ls -1t bot_config_*.tar.gz | tail -n "$OLD_BACKUPS" | xargs rm -f
    ls -1t bot_config_*.meta | tail -n "$OLD_BACKUPS" | xargs rm -f
    echo -e "${GREEN}   Удалено старых бэкапов: $OLD_BACKUPS${NC}"
else
    echo -e "${GREEN}   Всего бэкапов: $BACKUP_COUNT (< 30, удаление не требуется)${NC}"
fi

# Итоговая информация
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ БЭКАП ЗАВЕРШЁН${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📦 Файл: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "📊 Размер: $BACKUP_SIZE"
echo "📁 Всего бэкапов: $BACKUP_COUNT"
echo ""
echo "📋 Для восстановления:"
echo "   cd /root/trading_bot_v4_mtf"
echo "   tar -xzf ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo ""

# Опционально: загрузка в облако
if [ -f "/root/upload_to_cloud.sh" ]; then
    echo -e "${YELLOW}☁️  Загрузка в облако...${NC}"
    /root/upload_to_cloud.sh "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
fi
