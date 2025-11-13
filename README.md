# Bybit Futures AI Trading Bot

Инфраструктура проекта для разработки и тестирования бота на фьючерсах Bybit с ML/AI компонентами.

## Структура
- `src/bybit_bot/` — исходный код бота (API, логика, ML, Telegram).
- `config/` — конфигурация и загрузка переменных окружения.
- `tests/` — модульные и интеграционные тесты.
- `data/` — исторические данные и артефакты бэктестов.
- `data/trades/` — журнал сделок (самоочищается по ротации).
- `data/analysis/` — лог адаптации и точности.
- `notebooks/` — исследования, эксперименты, ML-прототипы.
- `scripts/` — служебные утилиты (деплой, миграции и т.п.).
- `keys/` — секреты (`.env` хранить локально, пример: `.env.example`).
- `logs/` — файлы логов с автоматической ротацией.

## Подготовка окружения
1. Создайте Python 3.12+ виртуальное окружение и активируйте его.
2. Установите зависимости: `pip install -r bybit_bot_requirements.txt`.
3. Скопируйте `keys/.env.example` в `keys/.env` и заполните реальными ключами.

## Проверка настроек
```bash
python -c "from config.settings import settings; print(settings.bybit_api_key[:4] + '***')"
```

## Дальнейшая работа
Проект развивается по этапам: API-клиент, ML пайплайн, торговая логика, Telegram-интерфейс, оркестрация и деплой. Для обучения ансамбля используется правило DiscoRL (`Disco57`) — оно задаёт адаптивный цикл переобучения моделей на новых данных.

## Запуск
1. Убедитесь, что обученный ансамбль сохранён в `models/ensemble/ensemble.joblib`.
2. Выполните:
   ```bash
   python scripts/run_bot.py
   ```
   Скрипт поднимет Telegram-бот, подключится к Bybit и запустит циклы торговли через оркестратор.

### Что делает рантайм
- Telegram-команды `/start`, `/help`, `/status`, `/balance`, `/positions`, `/history`, `/stop`, `/resume`, `/stats`, `/analysis`.
- Онлайн-адаптация (дисциплина Disco57): веса ансамбля, порог сигналов, размер позиции подстраиваются под прибыльность сделок.
- Market scanner автоматически сканирует watchlist и рекомендует монеты с высокой вероятностью (`/analysis`, блок в `/status`).
- Журнал сделок (`data/trades/*.csv`), логи (`logs/bot.log`) и аналитика (`data/analysis/adaptation.log`) поддерживаются автоматически с очисткой старых файлов (по умолчанию 14 дней).

## Обучение моделей
```bash
PYTHONPATH=src:. python scripts/train_ensemble.py
```
Скрипт загрузит исторические данные через ccxt, обучит ансамбль (с применением правила DiscoRL `Disco57`) и сохранит артефакт в `models/ensemble/ensemble.joblib`.

Для регулярного переобучения можно запланировать выполнение скрипта (cron/systemd timer). После завершения стоит архивировать предыдущий `ensemble.joblib` в `models/ensemble/backups/`.

## Исторические данные и бэктест
1. Собрать котировки:
   ```bash
   PYTHONPATH=src:. python scripts/fetch_data.py
   ```
   Файлы появятся в `data/historical/`.
2. Запустить бэктест:
   ```bash
   PYTHONPATH=src:. python scripts/backtest.py
   ```
   Отчёты сохраняются в `data/analysis/backtest_*.json` (метрики PnL, Sharpe, max drawdown). 

## Тестирование
```bash
PYTHONPATH=src:. pytest
```
Интеграционный тест `tests/integration/test_trading_cycle.py` проверяет цикл торговли с адаптацией весов и рисков без обращения к реальному API.
Для ускоренной проверки можно запускать подмножество:
```bash
PYTHONPATH=src:. pytest tests/core/test_risk.py tests/core/test_journal.py tests/core/test_storage.py \
    tests/ml/test_pipeline.py tests/integration/test_trading_cycle.py tests/telegram/test_messages.py
```

## CI/CD и деплой
- GitHub Actions workflow: `.github/workflows/ci.yml` (устанавливает зависимости, запускает pytest).
- Пример unit-файла systemd: `config/systemd/bybit-bot.service` — скопируйте на сервер, обновите пути и активируйте `systemctl enable --now bybit-bot`.
- Для автоматического переобучения можно запланировать `scripts/train_ensemble.py` (cron/systemd timer) — артефакт обновится в `models/ensemble/`.
