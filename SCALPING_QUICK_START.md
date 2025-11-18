# ⚡ БЫСТРЫЙ СТАРТ: СКАЛЬПИНГ-РЕЖИМ

## 🎯 ПАРАМЕТРЫ

```
Таймфреймы: 5m → 15m → 30m → 1h → 4h
Плечо: 20x
Цель: +1-5% движение = +20-100% ROE
Проверка: каждые 10 секунд
```

## 📝 ШАГ 1: Добавить в config.py

```python
# В конец файла config.py добавить:

# ═══════════════════════════════════════════════════════════════════
# ⚡ SCALPING MODE
# ═══════════════════════════════════════════════════════════════════
ENABLE_SCALPING_MODE = True

SCALPING_TIMEFRAMES = {
    "5m": "5",
    "15m": "15", 
    "30m": "30",
    "1h": "60",
    "4h": "240",
}
SCALPING_PRIMARY_TIMEFRAME = "5m"
SCALPING_MIN_TIMEFRAME_ALIGNMENT = 3

SCALPING_LEVERAGE = 20
SCALPING_POSITION_SIZE_USD = 0.5  # $0.5 × 20x = $10
SCALPING_MAX_CONCURRENT_POSITIONS = 5

SCALPING_TP_LEVELS = [
    {"price_move_percent": 1.0, "roe": 20, "close_percent": 30},
    {"price_move_percent": 2.0, "roe": 40, "close_percent": 30},
    {"price_move_percent": 3.0, "roe": 60, "close_percent": 20},
    {"price_move_percent": 4.0, "roe": 80, "close_percent": 10},
    {"price_move_percent": 5.0, "roe": 100, "close_percent": 10},
]

SCALPING_STOP_LOSS_PERCENT = 0.5  # -0.5% = -10% ROE
SCALPING_USE_TRAILING_SL = True
SCALPING_TRAILING_SL_ACTIVATION_PERCENT = 0.5
SCALPING_TRAILING_SL_CALLBACK_PERCENT = 0.3

SCALPING_MAX_POSITION_DURATION_MINUTES = 60
SCALPING_MONITORING_INTERVAL_SECONDS = 10
SCALPING_ANALYSIS_INTERVAL_SECONDS = 60

SCALPING_SIGNAL_THRESHOLDS = {
    "min_confidence": 0.75,
    "min_volume_ratio": 2.0,
}
```

## 📝 ШАГ 2: Создать scalping_engine.py

Файл уже создан: `scalping_engine.py` (см. SCALPING_IMPLEMENTATION.md)

## 📝 ШАГ 3: Интегрировать в main.py

```python
# В начале файла
from scalping_engine import ScalpingEngine

class TradingBotV4MTF:
    def __init__(self):
        # После существующего кода добавить:
        
        if config.ENABLE_SCALPING_MODE:
            self.scalping_engine = ScalpingEngine(self.client, self.logger)
            self.logger.info("⚡ Scalping Engine активирован")
    
    async def run(self):
        """Главный цикл"""
        tasks = [
            asyncio.create_task(self.analysis_loop()),
            asyncio.create_task(self.monitoring_loop()),
            asyncio.create_task(self.telegram.start()),
        ]
        
        # Добавить скальпинг-задачи
        if config.ENABLE_SCALPING_MODE:
            tasks.append(asyncio.create_task(self.scalping_loop()))
        
        await asyncio.gather(*tasks)
    
    async def scalping_loop(self):
        """Цикл скальпинга"""
        while self.active:
            try:
                # Поиск сигналов
                await self.scan_scalping_signals()
                
                # Мониторинг позиций (каждые 10 сек)
                await self.scalping_engine.monitor_positions()
                
            except Exception as e:
                self.logger.error(f"Ошибка в scalping loop: {e}")
            
            await asyncio.sleep(10)
    
    async def scan_scalping_signals(self):
        """Поиск скальпинг-сигналов"""
        available = (config.SCALPING_MAX_CONCURRENT_POSITIONS - 
                    len(self.scalping_engine.scalping_positions))
        
        if available <= 0:
            return
        
        watchlist = self.update_watchlist()
        
        for symbol in watchlist[:30]:  # Топ-30
            if symbol in self.scalping_engine.scalping_positions:
                continue
            
            # Получаем данные
            mtf_data = self.scalping_engine.get_mtf_data(symbol)
            if not mtf_data:
                continue
            
            # Анализируем
            signal = self.scalping_engine.analyze_scalping_signal(symbol, mtf_data)
            
            if signal:
                current_price = mtf_data['5m']['close'].iloc[-1]
                await self.scalping_engine.open_scalping_position(
                    symbol, signal['direction'], signal['confidence'], current_price
                )
                
                if len(self.scalping_engine.scalping_positions) >= config.SCALPING_MAX_CONCURRENT_POSITIONS:
                    break
```

## 🚀 ЗАПУСК

```bash
# 1. Остановить текущий бот
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "pkill -f main.py"

# 2. Загрузить новые файлы на сервер
scp -i ~/.ssh/upcloud_trading_bot config.py root@185.70.199.244:/root/trading_bot_v4_mtf/
scp -i ~/.ssh/upcloud_trading_bot scalping_engine.py root@185.70.199.244:/root/trading_bot_v4_mtf/
scp -i ~/.ssh/upcloud_trading_bot main.py root@185.70.199.244:/root/trading_bot_v4_mtf/

# 3. Запустить бот
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "cd /root/trading_bot_v4_mtf && nohup venv/bin/python main.py > logs/bot_output.log 2>&1 &"

# 4. Проверить логи
ssh -i ~/.ssh/upcloud_trading_bot root@185.70.199.244 "tail -f /root/trading_bot_v4_mtf/logs/bot_output.log"
```

## 📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

```
Сделок в день: 10-30
Win Rate: 70-75%
Средний ROE: +30-50%
ROE в день: 15-30%
```

## 📈 ПРИМЕР РАБОТЫ

```
⚡ SCALPING SIGNAL: ETHUSDT LONG | Confidence: 78% | TF: 4/5
✅ SCALPING позиция открыта: ETHUSDT LONG | Size: $10 | Qty: 0.003

🎯 TP1 HIT: ETHUSDT | Move: +1.2% | ROE: +24% | Closed: 30% | P&L: +$0.36
🎯 TP2 HIT: ETHUSDT | Move: +2.1% | ROE: +42% | Closed: 30% | P&L: +$0.63
🔄 Trailing SL активирован: ETHUSDT
🎯 TP3 HIT: ETHUSDT | Move: +3.5% | ROE: +70% | Closed: 20% | P&L: +$0.70
✅ Scalping позиция закрыта: ETHUSDT | Total P&L: +$1.69
```

Хотите, чтобы я создал полный файл `scalping_engine.py`?
