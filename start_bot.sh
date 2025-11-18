#!/bin/bash

echo "🚀 Trading Bot V4.0 MTF - Запуск"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Меню
echo ""
echo "Выберите режим запуска:"
echo ""
echo "1️⃣  Тестирование компонентов (валидация индикаторов и стратегий)"
echo "2️⃣  Live Market Testing (тестирование на реальных данных БЕЗ сделок)"
echo "3️⃣  Полное тестирование (все 4 фазы)"
echo "4️⃣  Запуск бота в РЕАЛЬНОМ режиме"
echo "5️⃣  Просмотр логов"
echo "6️⃣  Статус сервиса"
echo "0️⃣  Выход"
echo ""
read -p "Введите номер: " choice

case $choice in
    1)
        echo -e "\n${BLUE}1️⃣ Запуск тестирования компонентов...${NC}\n"
        cd /root/trading_bot_v4_mtf
        source venv/bin/activate
        python testing/test_indicators.py
        ;;
    2)
        echo -e "\n${BLUE}2️⃣ Запуск Live Market Testing...${NC}\n"
        read -p "Длительность теста в минутах (по умолчанию 30): " duration
        duration=${duration:-30}
        cd /root/trading_bot_v4_mtf
        source venv/bin/activate
        echo "$duration" | python testing/live_market_test.py
        ;;
    3)
        echo -e "\n${BLUE}3️⃣ Запуск полного тестирования...${NC}\n"
        read -p "Длительность Live Testing в минутах (по умолчанию 60): " duration
        duration=${duration:-60}
        cd /root/trading_bot_v4_mtf
        source venv/bin/activate
        echo "$duration" | python testing/run_full_test.py
        ;;
    4)
        echo -e "\n${RED}⚠️  ВНИМАНИЕ: Запуск бота в РЕАЛЬНОМ режиме!${NC}"
        echo -e "${YELLOW}Бот будет открывать реальные позиции на бирже.${NC}\n"
        read -p "Вы уверены? (yes/no): " confirm
        
        if [ "$confirm" = "yes" ]; then
            echo -e "\n${GREEN}🚀 Запуск бота через systemd...${NC}\n"
            sudo systemctl start trading-bot.service
            sleep 2
            sudo systemctl status trading-bot.service --no-pager
            echo -e "\n${GREEN}✅ Бот запущен!${NC}"
            echo -e "${YELLOW}Для просмотра логов: sudo journalctl -u trading-bot.service -f${NC}\n"
        else
            echo -e "\n${YELLOW}Запуск отменён${NC}\n"
        fi
        ;;
    5)
        echo -e "\n${BLUE}📋 Просмотр логов...${NC}\n"
        echo "Выберите лог:"
        echo "1. Логи бота (main)"
        echo "2. Логи systemd"
        echo "3. Логи тестирования"
        read -p "Введите номер: " log_choice
        
        case $log_choice in
            1)
                tail -f /root/trading_bot_v4_mtf/logs/trading_bot_v4.log
                ;;
            2)
                sudo journalctl -u trading-bot.service -f
                ;;
            3)
                tail -f /root/trading_bot_v4_mtf/testing/logs/live_market_test.log
                ;;
        esac
        ;;
    6)
        echo -e "\n${BLUE}📊 Статус сервиса...${NC}\n"
        sudo systemctl status trading-bot.service --no-pager
        echo ""
        echo "Команды управления:"
        echo "  sudo systemctl start trading-bot.service   - запустить"
        echo "  sudo systemctl stop trading-bot.service    - остановить"
        echo "  sudo systemctl restart trading-bot.service - перезапустить"
        echo ""
        ;;
    0)
        echo -e "\n${GREEN}👋 До свидания!${NC}\n"
        exit 0
        ;;
    *)
        echo -e "\n${RED}❌ Неверный выбор${NC}\n"
        exit 1
        ;;
esac
