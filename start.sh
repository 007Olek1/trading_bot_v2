#!/bin/bash
# Быстрый старт TradeGPT Scalper

echo "========================================="
echo "TradeGPT Scalper - Запуск"
echo "========================================="

# Проверка .env
if [ ! -f ".env" ]; then
    echo "❌ Файл .env не найден!"
    echo ""
    echo "Создайте файл .env:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    echo ""
    echo "Заполните API ключи Bybit и Telegram"
    exit 1
fi

echo "✅ Файл .env найден"

# Проверка зависимостей
echo ""
echo "Проверка зависимостей..."
if ! python3 -c "import ccxt" 2>/dev/null; then
    echo "⚠️  Зависимости не установлены"
    echo "Установка..."
    pip3 install -r requirements.txt
fi

echo "✅ Зависимости установлены"

# Запуск
echo ""
echo "========================================="
echo "🚀 Запуск TradeGPT Scalper..."
echo "========================================="
echo ""

python3 tradegpt_scalper.py
