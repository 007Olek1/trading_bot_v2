# 🚀 Trading Bot V4.0 MTF - Universal Strategy v2.0

**Автоматизированный торговый бот для Bybit фьючерсов**  
Универсальная стратегия с адаптивным TP и Multi-Timeframe анализом

**Версия:** v2.0 (18.11.2025)  
**Статус:** ✅ Production Ready - Протестировано и работает на 100%

---

## 🎯 Универсальная Стратегия v2.0

### 📊 Индикаторы (5 компонентов):

1. **EMA (20/50/200)** — Направление тренда (25%)
2. **RSI + MACD** — Точки входа (35%)
3. **Volume Profile** — Подтверждение (25%)
4. **ATR** — Риск-менеджмент (фактор x0.9-1.1)
5. **Bollinger Bands** — Экстремальные зоны (15%)

### 🎯 Адаптивный Take Profit

```
Начало: +10% ROI (1% от цены при 10x)
   ↓
Достигли +10% → подтягиваем до +20% ROI
   ↓
Достигли +20% → подтягиваем до +30% ROI
   ↓
...
   ↓
Максимум: +100% ROI (10% от цены при 10x)
```

**Параметры:**
- MIN_TP_ROI: 10%
- MAX_TP_ROI: 100%
- STEP: 10%

### 🛑 Trailing Stop Loss

- **Активация:** +1.0% прибыли (быстрая защита)
- **Откат:** 2.5% от максимума (больше свободы)
- **Макс убыток:** -$1.0

### 📈 MTF Таймфреймы

- **5m** → Быстрые сигналы
- **15m** → Краткосрочный тренд
- **1h** → Среднесрочный тренд
- **4h** → Долгосрочный тренд

**Минимум:** 3 из 4 таймфреймов должны подтверждать сигнал

### 💰 Торговые параметры

- **Размер позиции:** $1.0
- **Плечо:** x10
- **Эффективный размер:** $10.0
- **Макс. позиций:** 3
- **Мин. уверенность:** 65%

### ⏱️ Режим работы

- **Анализ рынка:** каждые 15 минут
- **Мониторинг позиций:** каждую минуту
- **Ротация логов:** автоматическая
- **Контроль диска:** встроенный

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
