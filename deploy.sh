#!/bin/bash

echo "🚀 Развёртывание Trading Bot V4.0 MTF на сервере"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Проверка, что скрипт запущен на сервере
if [ ! -d "/root" ]; then
    echo -e "${RED}❌ Этот скрипт должен быть запущен на сервере${NC}"
    exit 1
fi

# 1. Остановка старых ботов
echo -e "\n${YELLOW}⏹️  Остановка старых ботов...${NC}"
sudo systemctl stop trading-bot.service 2>/dev/null || true
sudo systemctl stop trading_bot.service 2>/dev/null || true

# 2. Создание директории
echo -e "\n${YELLOW}📁 Создание директории...${NC}"
mkdir -p /root/trading_bot_v4_mtf
cd /root/trading_bot_v4_mtf

# 3. Создание виртуального окружения
echo -e "\n${YELLOW}🐍 Создание виртуального окружения...${NC}"
python3 -m venv venv
source venv/bin/activate

# 4. Установка зависимостей
echo -e "\n${YELLOW}📦 Установка зависимостей...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 5. Копирование .env файла
echo -e "\n${YELLOW}🔑 Настройка переменных окружения...${NC}"
if [ -f "/root/.env" ]; then
    cp /root/.env /root/trading_bot_v4_mtf/.env
    echo -e "${GREEN}✅ .env файл скопирован${NC}"
else
    echo -e "${RED}⚠️  Файл /root/.env не найден. Создайте его вручную!${NC}"
fi

# 6. Создание директории для логов
echo -e "\n${YELLOW}📝 Создание директории для логов...${NC}"
mkdir -p logs

# 7. Установка systemd сервиса
echo -e "\n${YELLOW}⚙️  Установка systemd сервиса...${NC}"
sudo cp trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service

# 8. Запуск сервиса
echo -e "\n${YELLOW}🚀 Запуск сервиса...${NC}"
sudo systemctl start trading-bot.service

# 9. Проверка статуса
echo -e "\n${YELLOW}📊 Проверка статуса...${NC}"
sleep 3
sudo systemctl status trading-bot.service --no-pager

# 10. Удаление старых ботов (опционально)
echo -e "\n${YELLOW}🗑️  Удаление старых ботов...${NC}"
read -p "Удалить старые боты (bybit_bot и bybit_futures_bot)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo systemctl disable trading_bot.service 2>/dev/null || true
    sudo rm /etc/systemd/system/trading_bot.service 2>/dev/null || true
    
    # Создаём бэкап перед удалением
    echo -e "${YELLOW}📦 Создание бэкапа...${NC}"
    tar -czf /root/old_bots_backup_$(date +%Y%m%d_%H%M%S).tar.gz /root/bybit_bot /root/bybit_futures_bot 2>/dev/null || true
    
    # Удаляем старые директории
    rm -rf /root/bybit_bot
    rm -rf /root/bybit_futures_bot
    rm -rf /root/trading_bot_env
    
    echo -e "${GREEN}✅ Старые боты удалены (бэкап создан)${NC}"
fi

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Развёртывание завершено!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n📊 Полезные команды:"
echo -e "  ${YELLOW}sudo systemctl status trading-bot.service${NC} - статус"
echo -e "  ${YELLOW}sudo journalctl -u trading-bot.service -f${NC} - логи в реальном времени"
echo -e "  ${YELLOW}tail -f /root/trading_bot_v4_mtf/logs/trading_bot_v4.log${NC} - логи бота"
echo -e "  ${YELLOW}sudo systemctl restart trading-bot.service${NC} - перезапуск"
echo -e "  ${YELLOW}sudo systemctl stop trading-bot.service${NC} - остановка"
echo ""
