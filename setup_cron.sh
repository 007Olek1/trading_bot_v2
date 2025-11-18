#!/bin/bash

# 📅 НАСТРОЙКА CRON ЗАДАЧ
# Автоматизация обслуживания бота

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📅 НАСТРОЙКА CRON ЗАДАЧ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Директория бота
BOT_DIR="/root/trading_bot_v4_mtf"

# Делаем скрипты исполняемыми
echo -e "${YELLOW}🔧 Делаем скрипты исполняемыми...${NC}"
chmod +x "$BOT_DIR"/*.sh
chmod +x "$BOT_DIR"/*.py

# Создаём временный файл с cron задачами
CRON_FILE=$(mktemp)

# Добавляем существующие задачи (если есть)
crontab -l > "$CRON_FILE" 2>/dev/null || true

# Удаляем старые задачи бота (если есть)
sed -i '/trading_bot_v4_mtf/d' "$CRON_FILE"

# Добавляем новые задачи
cat >> "$CRON_FILE" << 'EOF'

# ═══════════════════════════════════════════════════════════════════
# TRADING BOT V4.0 MTF - АВТОМАТИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════

# Ежедневная очистка в 4:00
0 4 * * * /root/trading_bot_v4_mtf/cleanup.sh >> /root/trading_bot_v4_mtf/logs/cleanup.log 2>&1

# Еженедельная проверка в воскресенье 5:00
0 5 * * 0 /root/trading_bot_v4_mtf/weekly_check.sh >> /root/trading_bot_v4_mtf/logs/weekly_check.log 2>&1

# Ежедневный бэкап в 3:00
0 3 * * * /root/trading_bot_v4_mtf/backup_config.sh >> /root/trading_bot_v4_mtf/logs/backup.log 2>&1

# Мониторинг uptime каждые 5 минут
*/5 * * * * /root/trading_bot_v4_mtf/venv/bin/python3 /root/trading_bot_v4_mtf/uptime_monitor.py >> /root/trading_bot_v4_mtf/logs/uptime_monitor.log 2>&1

# Мониторинг критических ошибок каждые 2 минуты
*/2 * * * * /root/trading_bot_v4_mtf/venv/bin/python3 /root/trading_bot_v4_mtf/critical_alerts.py >> /root/trading_bot_v4_mtf/logs/critical_alerts.log 2>&1

EOF

# Устанавливаем новый crontab
crontab "$CRON_FILE"

# Удаляем временный файл
rm "$CRON_FILE"

echo -e "${GREEN}✅ Cron задачи настроены!${NC}"
echo ""
echo "📋 Расписание:"
echo "   • Ежедневно 03:00 - Бэкап конфигурации"
echo "   • Ежедневно 04:00 - Очистка старых данных"
echo "   • Воскресенье 05:00 - Еженедельная проверка"
echo "   • Каждые 5 минут - Мониторинг uptime"
echo "   • Каждые 2 минуты - Мониторинг критических ошибок"
echo ""
echo "📊 Проверка установленных задач:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
crontab -l | grep trading_bot_v4_mtf
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✅ Готово!${NC}"
