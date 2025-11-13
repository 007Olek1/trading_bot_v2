#!/bin/bash

# 📦 Скрипт деплоя и тестирования адаптивной системы

echo "🚀 Начинаем процесс деплоя..."

# Проверка соединения с сервером
echo "🔍 Проверка соединения с сервером..."
ssh -i ~/.ssh/upcloud_trading_bot root@213.163.199.116 "echo '✅ Соединение установлено'"

# Создание бэкапа
echo "💾 Создание бэкапа текущей версии..."
ssh -i ~/.ssh/upcloud_trading_bot root@213.163.199.116 "cd /root/trading_bot && \
    tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz *.py"

# Копирование новых файлов
echo "📤 Копирование новых файлов..."
scp -i ~/.ssh/upcloud_trading_bot \
    adaptive_parameters.py \
    adaptive_trading_system.py \
    test_adaptive_system.py \
    root@213.163.199.116:/root/trading_bot/

# Установка зависимостей
echo "📦 Установка зависимостей..."
ssh -i ~/.ssh/upcloud_trading_bot root@213.163.199.116 "cd /root/trading_bot && \
    pip3 install -r requirements.txt"

# Запуск тестирования
echo "🧪 Запуск тестирования..."
ssh -i ~/.ssh/upcloud_trading_bot root@213.163.199.116 "cd /root/trading_bot && \
    python3 test_adaptive_system.py"
#!/usr/bin/env python3
# Проверка результатов
echo "📊 Проверка результатов..."
ssh -i ~/.ssh/upcloud_trading_bot root@213.163.199.116 "cd /root/trading_bot && \
    cat test_results.json"

echo "✅ Деплой завершен!"
"""
🧪 Тестирование адаптивной торговой системы
==========================================
"""

import logging
from datetime import datetime, timedelta
import json
from adaptive_trading_system import AdaptiveTradingSystem
from adaptive_parameters import AdaptiveParameterSystem
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_market_data(days_back: int = 7) -> list:
    """Загрузка исторических данных"""
    # Здесь должна быть ваша логика загрузки данных
    # Возвращает список словарей с данными рынка
    pass

def calculate_pnl(trades: list) -> dict:
    """Расчет прибыли/убытков"""
    total_profit = 0
    winning_trades = 0
    total_trades = len(trades)

    for trade in trades:
        if trade['profit'] > 0:
            winning_trades += 1
        total_profit += trade['profit']

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        'total_profit': total_profit,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate
    }

def main():
    logger.info("🚀 Начало тестирования адаптивной системы")

    # Инициализация системы
    trading_system = AdaptiveTradingSystem()

    # Загрузка исторических данных
    market_data = load_market_data(days_back=7)

    if not market_data:
        logger.error("❌ Не удалось загрузить рыночные данные")
        return

    trades = []

    # Проход по историческим данным
    for data in market_data:
        result = trading_system.process_market_update(data)

        if result['action'] == 'enter_trade':
            trades.append({
                'timestamp': data['timestamp'],
                'entry_price': result['setup']['entry_price'],
                'position_size': result['setup']['position_size'],
                'take_profit': result['setup']['take_profit'],
                'stop_loss': result['setup']['stop_loss'],
                'leverage': result['setup']['leverage']
            })

    # Анализ результатов
    performance = calculate_pnl(trades)

    logger.info("📊 Результаты тестирования:")
    logger.info(f"Всего сделок: {performance['total_trades']}")
    logger.info(f"Прибыльных сделок: {performance['winning_trades']}")
    logger.info(f"Винрейт: {performance['win_rate']:.2f}%")
    logger.info(f"Общая прибыль: ${performance['total_profit']:.2f}")

    # Сохранение результатов
    with open('test_results.json', 'w') as f:
        json.dump(performance, f, indent=2)

if __name__ == "__main__":
    main()
