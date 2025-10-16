# 🤖 Trading Bot V2.0

Автоматизированный торговый бот для криптовалютной биржи Bybit с продвинутой аналитикой и управлением рисками.

## 📊 Основные возможности

- ✅ Автоматический анализ 100+ криптовалютных пар
- ✅ Продвинутая техническая индикаторная система (RSI, MACD, Bollinger Bands, EMA, ATR)
- ✅ Анализ трендов и волатильности
- ✅ Trailing Stop Loss для защиты прибыли
- ✅ Частичное закрытие позиций (до 5 уровней Take Profit)
- ✅ Cooldown механизм для предотвращения повторных входов
- ✅ Telegram бот для уведомлений и управления
- ✅ Auto-Healing система для автоматического восстановления
- ✅ Ротация логов

## 🛠️ Технологии

- Python 3.10+
- CCXT (Bybit API)
- Pandas (анализ данных)
- aiogram (Telegram bot)
- asyncio (асинхронная обработка)

## 📦 Установка

```bash
# Клонировать репозиторий
git clone https://github.com/007Olek1/trading_bot_v2.git
cd trading_bot_v2

# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scriptsctivate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

## ⚙️ Настройка

Создайте файл `.env` с вашими API ключами:

```env
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

## 🚀 Запуск

```bash
python3 trading_bot_v2_main.py
```

## 📱 Telegram команды

- `/start` - Запуск бота
- `/status` - Текущий статус и позиции
- `/positions` - Открытые позиции
- `/history` - История сделок
- `/pause` - Приостановить торговлю
- `/resume` - Возобновить торговлю
- `/close_all` - Закрыть все позиции

## ⚠️ Disclaimer

Этот бот предназначен только для образовательных целей. Торговля криптовалютами несет высокий риск. Используйте на свой страх и риск.

## 📄 Лицензия

MIT License

