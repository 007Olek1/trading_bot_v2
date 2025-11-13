# Bybit Futures Bot Concept

## Overview
- Deployment target: server `185.70.199.244`
- Trading venue: Bybit USDT Perpetual futures
- Operating mode: `MULTI_OPTIMIZE`
- Leverage: `x10`
- Base position size: `$1` per entry
- Maximum concurrent positions: `3`
- Telegram control bot and notifications enabled

## Multi-Timeframe Stack
- 15m → 30m → 1h → 4h → 24h alignment
- Ensemble signal confirmation across rolling MTF windows

## Strategy Components
- 5 ML models ensemble: Random Forest, LightGBM, SVM, Neural Network (XGBoost исключён из-за зависимостей macOS)
- 20+ technical indicators и тысячи микропаттернов, собранных в единую логику 🧠
- Real-time Bybit API integration (execution + account data)
- AI-driven forecasting layer for predictive bias
- Risk-management module controlling leverage, exposure, and stop logic
- Применяется правило обучения DiscoRL (`Disco57`) для адаптивного итеративного обучения ансамбля
<<<<<<< Current (Your changes)
=======
- Автоматический журнал сделок, аналитика и ротация логов с самоочисткой директорий
- Бэктест на исторических данных (через `scripts/fetch_data.py` и `scripts/backtest.py`) для оценки PnL, Sharpe, drawdown
>>>>>>> Incoming (Background Agent changes)

## Bot Status Template
```
🚀 БОТ  — ЗАПУЩЕН!

📊 MTF Таймфреймы

15m ⏩ 30m ⏩ 1h ⏩ 4h ⏩ 24h

💰 Баланс: $

💎 Всего: $

🆓 Свободно: $

⚡ ПАРАМЕТРЫ

├ Режим: MULTI_OPTIMIZE

├ Плечо: x10

├ Позиция: $1

└ Макс. позиций: 3 одновременно
```

## Telegram Command Set
- `/start` – стартовое сообщение
- `/help` – список команд
- `/status` – статус бота
- `/balance` – текущий баланс
- `/positions` – открытые позиции
- `/history` – история сделок
- `/stop` – остановить торговлю
- `/resume` – возобновить
- `/stats` – статистика
- `/analysis` – анализ рынка

## Operational Metadata
- Local time reference: Warsaw timezone
- Server: `185.70.199.244`
- Telegram notifications for trades, alerts, and status
- Professional reporting pipeline (performance summaries, risk metrics)

## Next Steps
1. Formalize requirements for datasets and live data ingestion.
2. Design feature pipeline for 20+ indicators and ML inputs.
3. Implement ensemble trainer and real-time inference orchestrator.
4. Build Bybit execution module with robust error handling.
5. Integrate Telegram bot commands with stateful controller.
6. Backtest, forward-test, and deploy with monitoring and automated reports.
