# 🚀 Trading Bot V4.0 MTF (Super Bot)

Автоматизированный торговый бот для Bybit фьючерсов с Multi-Timeframe анализом и продвинутыми стратегиями.

## 📊 Особенности

### MTF Таймфреймы
- **15m ⏩ 30m ⏩ 1h ⏩ 4h ⏩ 1D**
- Анализ по всем таймфреймам одновременно
- Минимум 3 таймфрейма должны подтверждать сигнал

### 🎯 Стратегии
1. **💹 Тренд + Объём + Bollinger** (40% веса)
   - Определение тренда по EMA (20/50/200)
   - Подтверждение объёмом
   - Вход на отбое от Bollinger Bands

2. **🎭 Детектор манипуляций** (30% веса)
   - Обнаружение pump & dump
   - Определение ложных пробоев
   - Анализ аномальных объёмов

3. **🌍 Анализ глобального тренда** (30% веса)
   - Долгосрочное направление (4h + 1D)
   - Фильтрация сделок против тренда

### 🎯 Take Profit (6 уровней)
- **TP1**: +4% (закрыть 40% позиции)
- **TP2**: +6% (закрыть 20% позиции)
- **TP3**: +8% (закрыть 20% позиции)
- **TP4**: +10% (закрыть 10% позиции)
- **TP5**: +12% (закрыть 5% позиции)
- **TP6**: +15% (закрыть 5% позиции)

**Гарантия**: минимум +$1.00 прибыли

### 🛑 Stop Loss
- Максимальный убыток: **-$1.0**
- **Trailing Stop Loss** активируется при +2%
- Откат для trailing: 1.5%

### 💰 Торговля
- Размер позиции: **$1.0**
- Плечо: **x10**
- Эффективный размер: **$10.0**
- Максимум позиций: **3**

### ⏱️ Режим работы
- **Анализ рынка**: каждые 15 минут
- **Мониторинг позиций**: каждую минуту

## 📱 Telegram Команды

- `/start` - 🟢 Стартовое сообщение
- `/help` - 📝 Список команд
- `/status` - 📊 Статус бота
- `/balance` - 💰 Баланс
- `/positions` - 📈 Открытые позиции
- `/history` - 📜 История сделок
- `/stop` - ⛔ Остановить торговлю
- `/resume` - ▶️ Возобновить
- `/stats` - 📊 Статистика

## 🚀 Установка

### 1. Клонирование и настройка

```bash
cd /root
git clone <repository_url> trading_bot_v4_mtf
cd trading_bot_v4_mtf
```

### 2. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения

```bash
cp .env.example .env
nano .env
```

Заполните:
- `BYBIT_API_KEY` - API ключ Bybit
- `BYBIT_API_SECRET` - API секрет Bybit
- `TELEGRAM_TOKEN` - Токен Telegram бота
- `TELEGRAM_CHAT_ID` - ID чата Telegram

### 5. Создание systemd сервиса

```bash
sudo nano /etc/systemd/system/trading-bot.service
```

Содержимое:
```ini
[Unit]
Description=Trading Bot V4.0 MTF (Super Bot)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/trading_bot_v4_mtf
Environment="PATH=/root/trading_bot_v4_mtf/venv/bin"
ExecStart=/root/trading_bot_v4_mtf/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 6. Запуск сервиса

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service
sudo systemctl start trading-bot.service
```

### 7. Проверка статуса

```bash
sudo systemctl status trading-bot.service
sudo journalctl -u trading-bot.service -f
```

## 📊 Мониторинг

### Логи
```bash
tail -f logs/trading_bot_v4.log
```

### Статус сервиса
```bash
sudo systemctl status trading-bot.service
```

### Перезапуск
```bash
sudo systemctl restart trading-bot.service
```

### Остановка
```bash
sudo systemctl stop trading-bot.service
```

## 🔧 Конфигурация

Основные параметры находятся в `config.py`:
- Размер позиции и плечо
- Таймфреймы для анализа
- Параметры индикаторов
- Пороги сигналов
- Watchlist монет

## ⚠️ Важно

- Используйте только на свой страх и риск
- Начните с малых сумм для тестирования
- Регулярно проверяйте логи и статус
- Не забывайте про риск-менеджмент

## 📝 Лицензия

MIT License

## 👨‍💻 Автор

Trading Bot V4.0 MTF - 2025
