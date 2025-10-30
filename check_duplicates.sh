#!/bin/bash
# Скрипт для проверки дубликатов бота

echo "🔍 ПРОВЕРКА ДУБЛИКАТОВ БОТА"
echo "=================================="
echo ""

echo "1️⃣ ЛОКАЛЬНЫЕ ПРОЦЕССЫ (Mac):"
LOCAL_PROCESSES=$(ps aux | grep -E "super_bot_v4_mtf\.py" | grep -v grep | grep -v "tail\|grep")
if [ -z "$LOCAL_PROCESSES" ]; then
    echo "   ✅ Локальных процессов бота не найдено"
else
    echo "   ⚠️ Найдены локальные процессы:"
    echo "$LOCAL_PROCESSES"
    echo ""
    echo "   Для остановки выполните:"
    echo "   pkill -f 'super_bot_v4_mtf.py'"
fi

echo ""
echo "2️⃣ ПРОЦЕССЫ НА СЕРВЕРЕ:"
SERVER_IP="213.163.199.116"
SSH_KEY="$HOME/.ssh/upcloud_trading_bot"

if [ -f "$SSH_KEY" ]; then
    SERVER_PROCESSES=$(ssh -i "$SSH_KEY" -o ConnectTimeout=5 root@"$SERVER_IP" 'ps aux | grep -E "super_bot_v4_mtf\.py" | grep -v grep' 2>/dev/null)
    
    if [ -z "$SERVER_PROCESSES" ]; then
        echo "   ⚠️ Процессы бота на сервере не найдены"
    else
        COUNT=$(echo "$SERVER_PROCESSES" | wc -l)
        if [ "$COUNT" -eq 1 ]; then
            echo "   ✅ Только 1 процесс (правильно):"
            echo "$SERVER_PROCESSES" | sed 's/^/      /'
        else
            echo "   ⚠️ Найдено $COUNT процессов (должен быть 1):"
            echo "$SERVER_PROCESSES" | sed 's/^/      /'
        fi
    fi
    
    echo ""
    echo "3️⃣ СТАТУС СЕРВИСА:"
    STATUS=$(ssh -i "$SSH_KEY" -o ConnectTimeout=5 root@"$SERVER_IP" 'systemctl is-active trading-bot' 2>/dev/null)
    if [ "$STATUS" = "active" ]; then
        echo "   ✅ Сервис активен"
    else
        echo "   ⚠️ Сервис не активен: $STATUS"
    fi
    
    echo ""
    echo "4️⃣ ПОСЛЕДНИЕ ОШИБКИ В ЛОГАХ:"
    CONFLICTS=$(ssh -i "$SSH_KEY" -o ConnectTimeout=5 root@"$SERVER_IP" 'tail -100 /opt/bot/super_bot_v4_mtf.log 2>/dev/null | grep -i "conflict\|error" | tail -3' 2>/dev/null)
    if [ -z "$CONFLICTS" ]; then
        echo "   ✅ Конфликтов не найдено"
    else
        echo "   ⚠️ Найдены конфликты/ошибки:"
        echo "$CONFLICTS" | sed 's/^/      /'
    fi
else
    echo "   ⚠️ SSH ключ не найден: $SSH_KEY"
fi

echo ""
echo "=================================="
echo "✅ ПРОВЕРКА ЗАВЕРШЕНА"


