#!/bin/bash

echo "================================================================================"
echo "🔄 ПЕРЕЗАПУСК БОТА С ОБНОВЛЕНИЯМИ"
echo "================================================================================"
echo ""

# Переходим в директорию бота
cd "$(dirname "$0")"

# Проверяем наличие .env файла
if [ ! -f .env ]; then
    echo "❌ Файл .env не найден!"
    echo "📝 Создайте файл .env на основе .env.example"
    echo ""
    echo "Пример содержимого .env:"
    echo "BYBIT_API_KEY=your_api_key_here"
    echo "BYBIT_API_SECRET=your_api_secret_here"
    echo "TELEGRAM_BOT_TOKEN=your_telegram_token_here"
    echo "TELEGRAM_CHAT_ID=your_chat_id_here"
    echo "USE_TESTNET=False"
    echo ""
    exit 1
fi

# Останавливаем старый процесс бота
echo "🛑 Останавливаем старый процесс бота..."
pkill -f "python.*main.py" 2>/dev/null
sleep 2

# Проверяем, что процесс остановлен
if pgrep -f "python.*main.py" > /dev/null; then
    echo "⚠️  Принудительная остановка..."
    pkill -9 -f "python.*main.py"
    sleep 1
fi

echo "✅ Старый процесс остановлен"
echo ""

# Проверяем зависимости
echo "📦 Проверка зависимостей..."
if ! python3 -c "import pybit, pandas, numpy, telegram" 2>/dev/null; then
    echo "⚠️  Устанавливаем недостающие зависимости..."
    pip3 install -r requirements.txt
fi
echo "✅ Зависимости установлены"
echo ""

# Создаём директорию для логов
mkdir -p logs

# Проверяем позиции на бирже (если API ключи валидны)
echo "🔍 Проверка текущих позиций на бирже..."
python3 check_positions.py
echo ""

# Запускаем бота
echo "🚀 Запуск бота с обновлениями..."
echo "   ✅ Реализовано отслеживание TP уровней"
echo "   ✅ Реализован Trailing Stop Loss"
echo "   ✅ Улучшены Telegram уведомления"
echo ""

# Запускаем в фоне с перенаправлением вывода
nohup python3 main.py > logs/bot_output.log 2>&1 &
BOT_PID=$!

echo "✅ Бот запущен с PID: $BOT_PID"
echo ""

# Ждём 3 секунды и проверяем, что бот работает
sleep 3

if ps -p $BOT_PID > /dev/null; then
    echo "✅ Бот успешно запущен и работает!"
    echo ""
    echo "📊 Для просмотра логов используйте:"
    echo "   tail -f logs/bot_output.log"
    echo "   tail -f logs/trading_bot_v4.log"
    echo ""
    echo "🛑 Для остановки бота:"
    echo "   kill $BOT_PID"
    echo "   или: pkill -f 'python.*main.py'"
    echo ""
    echo "📱 Проверьте Telegram для уведомлений о запуске"
else
    echo "❌ Ошибка запуска бота!"
    echo "📋 Проверьте логи:"
    echo "   cat logs/bot_output.log"
    exit 1
fi

echo "================================================================================"
echo "✅ ПЕРЕЗАПУСК ЗАВЕРШЁН"
echo "================================================================================"
