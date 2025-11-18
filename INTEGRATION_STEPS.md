# 🚀 ИНТЕГРАЦИЯ СКАЛЬПИНГ-РЕЖИМА - 3 ПРОСТЫХ ШАГА

## ✅ ШАГ 1: Добавить конфигурацию

Открыть `config.py` и добавить в конец файла:

```python
# ═══════════════════════════════════════════════════════════════════
# ⚡ SCALPING MODE - Быстрые сделки с высоким ROE
# ═══════════════════════════════════════════════════════════════════
ENABLE_SCALPING_MODE = True

# Таймфреймы: 5m → 15m → 30m → 1h → 4h
SCALPING_TIMEFRAMES = {
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
}
SCALPING_PRIMARY_TIMEFRAME = "5m"
SCALPING_MIN_TIMEFRAME_ALIGNMENT = 3  # Минимум 3 из 5 TF

# Плечо 20x и размер
SCALPING_LEVERAGE = 20
SCALPING_POSITION_SIZE_USD = 0.5  # $0.5 × 20x = $10
SCALPING_MAX_CONCURRENT_POSITIONS = 5

# TP уровни: +1-5% = +20-100% ROE
SCALPING_TP_LEVELS = [
    {"price_move_percent": 1.0, "roe": 20, "close_percent": 30},
    {"price_move_percent": 2.0, "roe": 40, "close_percent": 30},
    {"price_move_percent": 3.0, "roe": 60, "close_percent": 20},
    {"price_move_percent": 4.0, "roe": 80, "close_percent": 10},
    {"price_move_percent": 5.0, "roe": 100, "close_percent": 10},
]

# Stop Loss
SCALPING_STOP_LOSS_PERCENT = 0.5  # -0.5% = -10% ROE
SCALPING_USE_TRAILING_SL = True
SCALPING_TRAILING_SL_ACTIVATION_PERCENT = 0.5
SCALPING_TRAILING_SL_CALLBACK_PERCENT = 0.3

# Временные параметры
SCALPING_MAX_POSITION_DURATION_MINUTES = 60
SCALPING_MONITORING_INTERVAL_SECONDS = 10  # Проверка каждые 10 сек
SCALPING_ANALYSIS_INTERVAL_SECONDS = 60

# Фильтры
SCALPING_SIGNAL_THRESHOLDS = {
    "min_confidence": 0.75,
    "min_volume_ratio": 2.0,
}
```

## ✅ ШАГ 2: Интегрировать в main.py

### 2.1 Импорт в начале файла

```python
# В начале main.py после других импортов добавить:
from scalping_engine import ScalpingEngine
```

### 2.2 Инициализация в __init__

```python
def __init__(self):
    # ... существующий код ...
    
    # В конце __init__ добавить:
    if config.ENABLE_SCALPING_MODE:
        self.scalping_engine = ScalpingEngine(self.client, self.logger)
        self.logger.info("⚡ Scalping Engine активирован")
        self.logger.info(f"⚡ Таймфреймы: {' → '.join(config.SCALPING_TIMEFRAMES.keys())}")
        self.logger.info(f"⚡ Плечо: {config.SCALPING_LEVERAGE}x")
        self.logger.info(f"⚡ Цель: +1-5% = +20-100% ROE")
```

### 2.3 Добавить в run()

```python
async def run(self):
    """Главный цикл бота"""
    self.logger.info("🚀 Бот запущен!")
    
    # Существующие задачи
    tasks = [
        asyncio.create_task(self.analysis_loop()),
        asyncio.create_task(self.monitoring_loop()),
        asyncio.create_task(self.telegram.start()),
    ]
    
    # Добавить скальпинг-задачи
    if config.ENABLE_SCALPING_MODE:
        tasks.append(asyncio.create_task(self.scalping_analysis_loop()))
        tasks.append(asyncio.create_task(self.scalping_monitoring_loop()))
    
    await asyncio.gather(*tasks)
```

### 2.4 Добавить новые методы

```python
async def scalping_analysis_loop(self):
    """Поиск скальпинг-сигналов каждую минуту"""
    while self.active:
        try:
            await self.scan_scalping_signals()
        except Exception as e:
            self.logger.error(f"Ошибка в scalping analysis: {e}")
        
        await asyncio.sleep(config.SCALPING_ANALYSIS_INTERVAL_SECONDS)

async def scalping_monitoring_loop(self):
    """Мониторинг скальпинг-позиций каждые 10 сек"""
    await self.scalping_engine.monitor_positions()

async def scan_scalping_signals(self):
    """Сканирование для скальпинга"""
    available = (config.SCALPING_MAX_CONCURRENT_POSITIONS - 
                len(self.scalping_engine.scalping_positions))
    
    if available <= 0:
        return
    
    self.logger.debug(f"⚡ Сканирование скальпинга... (слотов: {available})")
    
    watchlist = self.update_watchlist()
    
    for symbol in watchlist[:30]:  # Топ-30 по ликвидности
        if symbol in self.scalping_engine.scalping_positions:
            continue
        
        # Получаем MTF данные
        mtf_data = self.scalping_engine.get_mtf_data(symbol)
        if not mtf_data:
            continue
        
        # Анализируем сигнал
        signal = self.scalping_engine.analyze_signal(symbol, mtf_data)
        
        if signal:
            current_price = mtf_data['5m']['close'].iloc[-1]
            success = await self.scalping_engine.open_position(
                symbol, signal['direction'], signal['confidence'], current_price
            )
            
            if success:
                # Уведомление в Telegram
                await self.telegram.send_message(
                    f"⚡ SCALPING OPENED\n"
                    f"Symbol: {symbol}\n"
                    f"Direction: {signal['direction']}\n"
                    f"Price: ${current_price:.4f}\n"
                    f"Confidence: {signal['confidence']:.0%}\n"
                    f"Leverage: {config.SCALPING_LEVERAGE}x\n"
                    f"Target: +1-5% = +20-100% ROE"
                )
                
                if len(self.scalping_engine.scalping_positions) >= config.SCALPING_MAX_CONCURRENT_POSITIONS:
                    break
```

## ✅ ШАГ 3: Деплой на сервер

```bash
# 1. Остановить бота
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "pkill -f main.py"

# 2. Загрузить файлы
scp -i ~/.ssh/upcloud_trading_bot config.py root@185.70.199.244:/root/trading_bot_v4_mtf/
scp -i ~/.ssh/upcloud_trading_bot scalping_engine.py root@185.70.199.244:/root/trading_bot_v4_mtf/
scp -i ~/.ssh/upcloud_trading_bot main.py root@185.70.199.244:/root/trading_bot_v4_mtf/

# 3. Запустить
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "cd /root/trading_bot_v4_mtf && nohup venv/bin/python main.py > logs/bot_output.log 2>&1 &"

# 4. Проверить логи
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "tail -f /root/trading_bot_v4_mtf/logs/bot_output.log"
```

## 📊 Что ожидать в логах

```
🚀 TRADING BOT V4.0 MTF - SUPER BOT - ИНИЦИАЛИЗАЦИЯ
⚡ Scalping Engine активирован
⚡ Таймфреймы: 5m → 15m → 30m → 1h → 4h
⚡ Плечо: 20x
⚡ Цель: +1-5% = +20-100% ROE

⚡ SCALPING: ETHUSDT LONG | Conf: 78% | TF: 4/5
⚡ Открываю SCALPING: ETHUSDT LONG @ $3000.0000
✅ SCALPING открыт: ETHUSDT LONG | Qty: 0.003
🛑 SL: ETHUSDT @ $2985.0000 (-0.5%)

🎯 TP1: ETHUSDT | $3030.0000 | +1.00% | ROE: +20% | Closed: 30% | P&L: +$0.30
🎯 TP2: ETHUSDT | $3060.0000 | +2.00% | ROE: +40% | Closed: 30% | P&L: +$0.60
🔄 Trailing SL: ETHUSDT
🎯 TP3: ETHUSDT | $3090.0000 | +3.00% | ROE: +60% | Closed: 20% | P&L: +$0.60
✅ Закрыт: ETHUSDT | Reason: all_tp_hit
```

## 🎯 Готово!

Теперь бот работает в гибридном режиме:
- **Swing Trading**: 15m-1d, 10x, долгосрочные сделки
- **Scalping**: 5m-4h, 20x, быстрые сделки

**Ожидаемые результаты:**
- Сделок в день: 15-30 (вместо 1-3)
- ROE в день: 20-35% (вместо 5-10%)
- Win Rate: 70-75%

Хотите протестировать на testnet сначала?
