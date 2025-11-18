# 🚀 Quick Start - Trading Bot V4.0 MTF

## 📋 Что создано

### ✅ Объединённый бот
- **Trading Bot V4.0 MTF** - объединение двух ботов (bybit_bot + bybit_futures_bot)
- Multi-Timeframe анализ: 15m → 30m → 1h → 4h → 1D
- 3 стратегии: Тренд+Объём+Bollinger, Детектор манипуляций, Глобальный тренд
- 6 уровней TP, Trailing SL
- Полная интеграция с Telegram

### ✅ Система тестирования
- **1️⃣ Comprehensive Bot Testing** - валидация индикаторов и стратегий
- **2️⃣ Live Market Testing** - тестирование на реальных данных БЕЗ сделок
- **3️⃣ Performance Optimization** - анализ и оптимизация параметров
- **4️⃣ Trade Analysis** - детальный анализ результатов

## 🖥️ На сервере

### Подключение к серверу
```bash
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244
```

### Структура проекта
```
/root/trading_bot_v4_mtf/
├── main.py                 # Главный модуль бота
├── config.py               # Конфигурация
├── indicators.py           # Индикаторы
├── strategies.py           # Стратегии
├── utils.py                # Утилиты
├── telegram_handler.py     # Telegram
├── requirements.txt        # Зависимости
├── .env                    # API ключи
├── deploy.sh               # Скрипт развёртывания
├── start_bot.sh            # Скрипт запуска
├── trading-bot.service     # Systemd сервис
└── testing/                # Система тестирования
    ├── test_indicators.py
    ├── live_market_test.py
    ├── performance_optimizer.py
    ├── trade_analyzer.py
    ├── run_full_test.py
    └── README.md
```

## 🚀 Первый запуск

### 1. Развёртывание на сервере

```bash
# Подключитесь к серверу
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244

# Перейдите в директорию
cd /root/trading_bot_v4_mtf

# Запустите развёртывание
bash deploy.sh
```

Скрипт автоматически:
- ✅ Создаст виртуальное окружение
- ✅ Установит зависимости
- ✅ Скопирует .env файл
- ✅ Настроит systemd сервис
- ✅ Предложит удалить старые боты

### 2. Тестирование (ОБЯЗАТЕЛЬНО!)

**⚠️ НЕ ЗАПУСКАЙТЕ БОТА БЕЗ ТЕСТИРОВАНИЯ!**

```bash
# Используйте интерактивное меню
bash start_bot.sh
```

Выберите:
- **1** - Тестирование компонентов (5 минут)
- **3** - Полное тестирование (60+ минут)

Или напрямую:

```bash
# Быстрая проверка компонентов
cd /root/trading_bot_v4_mtf
source venv/bin/activate
python testing/test_indicators.py

# Live Market Testing (30 минут)
python testing/live_market_test.py
# Введите: 30

# Полное тестирование (60 минут)
python testing/run_full_test.py
# Введите: 60
```

### 3. Анализ результатов

```bash
# Оптимизация параметров
python testing/performance_optimizer.py

# Детальный анализ
python testing/trade_analyzer.py
```

**Проверьте метрики:**
- ✅ Win Rate ≥ 60%
- ✅ Profit Factor ≥ 1.5
- ✅ Средний PnL ≥ 1.5%

### 4. Применение рекомендаций

Если тестирование показало, что нужно изменить параметры:

```bash
nano config.py
```

Измените:
- `SIGNAL_THRESHOLDS['min_confidence']` - порог уверенности
- `MIN_TIMEFRAME_ALIGNMENT` - минимум таймфреймов
- Другие параметры по рекомендациям

### 5. Запуск бота

**Только после успешного тестирования!**

```bash
# Через меню
bash start_bot.sh
# Выберите: 4

# Или напрямую
sudo systemctl start trading-bot.service
```

## 📊 Мониторинг

### Просмотр логов

```bash
# Логи бота
tail -f /root/trading_bot_v4_mtf/logs/trading_bot_v4.log

# Логи systemd
sudo journalctl -u trading-bot.service -f

# Через меню
bash start_bot.sh
# Выберите: 5
```

### Статус сервиса

```bash
sudo systemctl status trading-bot.service

# Через меню
bash start_bot.sh
# Выберите: 6
```

### Telegram команды

- `/start` - Стартовое сообщение
- `/status` - Статус бота
- `/balance` - Баланс
- `/positions` - Открытые позиции
- `/stop` - Остановить торговлю
- `/resume` - Возобновить

## 🔧 Управление

### Остановка бота
```bash
sudo systemctl stop trading-bot.service
```

### Перезапуск
```bash
sudo systemctl restart trading-bot.service
```

### Отключение автозапуска
```bash
sudo systemctl disable trading-bot.service
```

## 🗑️ Удаление старых ботов

Если вы ещё не удалили старые боты при развёртывании:

```bash
# Остановка старых сервисов
sudo systemctl stop trading_bot.service
sudo systemctl disable trading_bot.service
sudo rm /etc/systemd/system/trading_bot.service

# Создание бэкапа
tar -czf /root/old_bots_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    /root/bybit_bot /root/bybit_futures_bot

# Удаление
rm -rf /root/bybit_bot
rm -rf /root/bybit_futures_bot
rm -rf /root/trading_bot_env

# Перезагрузка systemd
sudo systemctl daemon-reload
```

## ⚙️ Конфигурация

### Основные параметры (config.py)

```python
# Торговля
POSITION_SIZE_USD = 1.0      # Размер позиции
LEVERAGE = 10                # Плечо
MAX_CONCURRENT_POSITIONS = 3 # Макс. позиций

# TP уровни (6 штук)
TAKE_PROFIT_LEVELS = [...]

# SL
STOP_LOSS_MAX_USD = 1.0     # Макс. убыток

# Таймфреймы
TIMEFRAMES = {
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}

# Минимум таймфреймов для сигнала
MIN_TIMEFRAME_ALIGNMENT = 3
```

## 📝 Рекомендации

### Перед запуском на реальный счёт:

1. ✅ Пройдите полное тестирование (60+ минут)
2. ✅ Убедитесь, что Win Rate ≥ 60%
3. ✅ Проверьте, что Profit Factor ≥ 1.5
4. ✅ Начните с минимальной суммы ($1-2)
5. ✅ Мониторьте первые 24 часа
6. ✅ Постепенно увеличивайте размер позиции

### Во время работы:

1. 📊 Проверяйте логи каждые 2-4 часа
2. 💬 Следите за Telegram уведомлениями
3. 📈 Анализируйте результаты раз в день
4. ⚙️ Корректируйте параметры при необходимости

## 🆘 Troubleshooting

### Бот не запускается

```bash
# Проверьте логи
sudo journalctl -u trading-bot.service -n 50

# Проверьте .env файл
cat /root/trading_bot_v4_mtf/.env

# Проверьте зависимости
cd /root/trading_bot_v4_mtf
source venv/bin/activate
pip install -r requirements.txt
```

### Ошибки API

```bash
# Проверьте API ключи
nano /root/trading_bot_v4_mtf/.env

# Проверьте подключение
cd /root/trading_bot_v4_mtf
source venv/bin/activate
python -c "from pybit.unified_trading import HTTP; import os; from dotenv import load_dotenv; load_dotenv(); client = HTTP(api_key=os.getenv('BYBIT_API_KEY'), api_secret=os.getenv('BYBIT_API_SECRET')); print(client.get_wallet_balance(accountType='UNIFIED', coin='USDT'))"
```

### Telegram не работает

```bash
# Проверьте токен и chat_id
nano /root/trading_bot_v4_mtf/.env

# Убедитесь, что бот добавлен в чат
```

## 📞 Поддержка

- 📁 Логи: `/root/trading_bot_v4_mtf/logs/`
- 📊 Результаты тестов: `/root/trading_bot_v4_mtf/testing/logs/`
- 📖 Документация: `/root/trading_bot_v4_mtf/README.md`
- 🧪 Тестирование: `/root/trading_bot_v4_mtf/testing/README.md`

---

**Удачной торговли! 🚀💰**
