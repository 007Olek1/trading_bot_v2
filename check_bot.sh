#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        🔍 ПРОВЕРКА СТАТУСА БОТА НА СЕРВЕРЕ                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo "1️⃣ ПРОВЕРКА ПРОЦЕССОВ БОТА:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ps aux | grep -E "(python|super_bot|trading)" | grep -v grep || echo "❌ Процессы бота не найдены"
echo ""

echo "2️⃣ ПРОВЕРКА ДИРЕКТОРИЙ:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Проверка /opt/bot:"
if [ -d /opt/bot ]; then
    echo "✅ Директория /opt/bot существует"
    ls -lh /opt/bot/*.py 2>/dev/null | head -5 || echo "   Python файлы не найдены"
    echo "   Размер: $(du -sh /opt/bot 2>/dev/null | cut -f1)"
else
    echo "❌ Директория /opt/bot не найдена"
fi

echo ""
echo "📁 Проверка /root/trading_bot:"
if [ -d /root/trading_bot ]; then
    echo "✅ Директория /root/trading_bot существует"
    ls -lh /root/trading_bot/*.py 2>/dev/null | head -5 || echo "   Python файлы не найдены"
    echo "   Размер: $(du -sh /root/trading_bot 2>/dev/null | cut -f1)"
else
    echo "❌ Директория /root/trading_bot не найдена"
fi
echo ""

echo "3️⃣ СТАТУС SYSTEMD СЕРВИСА:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
systemctl status trading-bot --no-pager | head -15 2>/dev/null || echo "❌ Сервис trading-bot не найден или не настроен"
echo ""

echo "4️⃣ ПОСЛЕДНИЕ ЛОГИ:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f /opt/bot/bot.log ]; then
    echo "📝 Логи из /opt/bot/bot.log:"
    tail -20 /opt/bot/bot.log
elif [ -f /root/trading_bot/bot.log ]; then
    echo "📝 Логи из /root/trading_bot/bot.log:"
    tail -20 /root/trading_bot/bot.log
else
    echo "❌ Логи не найдены"
    echo "   Ищем логи в других местах:"
    find /opt /root -name "*.log" -type f 2>/dev/null | head -5
fi
echo ""

echo "5️⃣ РЕСУРСЫ СЕРВЕРА:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💾 Память:"
free -h | grep Mem
echo ""
echo "💿 Диск:"
df -h / | tail -1
echo ""

echo "6️⃣ СЕТЕВЫЕ СОЕДИН sentenceния:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
netstat -tuln 2>/dev/null | grep -E "(3000|8000|443|80|22)" | head -5 || ss -tuln 2>/dev/null | grep -E "(3000|8000|443|80|22)" | head -5 || echo "Нет активных соединений"
echo ""

echo "✅ ПРОВЕРКА ЗАВЕРШЕНА!"
echo ""
echo "📋 СЛЕДУЮЩИЕ ШАГИ:"
echo "   - Если бот не запущен: cd /opt/bot && python3 super_bot_v4_mtf.py &"
echo "   - Если нужно перезапустить: systemctl restart trading-bot"
echo "   - Для просмотра логов в реальном времени: tail -f /opt/bot/bot.log"

