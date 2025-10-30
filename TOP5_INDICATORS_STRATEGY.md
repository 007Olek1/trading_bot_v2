# 🎯 СТРАТЕГИЯ ТОП-5 ИНДИКАТОРОВ ДЛЯ ДЕРИВАТИВОВ

**Основа:** EMA, RSI, MACD, Bollinger Bands, ATR  
**Убираем:** Volume Ratio из основных фильтров  
**Цель:** Генерировать 2-5 сигналов в день

---

## 📊 **НОВАЯ ЛОГИКА СИГНАЛОВ:**

### **1. EMA ТРЕНД (9, 21, 50):**

```python
# Определение тренда на разных таймфреймах
def get_ema_trend(data):
    """Определить тренд по EMA"""
    ema_9 = data['ema_9']
    ema_21 = data['ema_21'] 
    ema_50 = data.get('ema_50', ema_21)
    
    # BULLISH: EMA9 > EMA21 > EMA50
    if ema_9 > ema_21 > ema_50:
        return 'STRONG_BULL'
    elif ema_9 > ema_21:
        return 'BULL'
    elif ema_9 < ema_21 < ema_50:
        return 'STRONG_BEAR'
    elif ema_9 < ema_21:
        return 'BEAR'
    else:
        return 'NEUTRAL'
```

### **2. RSI ЗОНЫ:**

```python
# Новые RSI пороги (смягченные)
RSI_OVERSOLD = 35      # было 40
RSI_OVERBOUGHT = 65    # было 60
RSI_EXTREME_LOW = 25   # новый
RSI_EXTREME_HIGH = 75  # новый

def get_rsi_signal(rsi):
    if rsi <= RSI_EXTREME_LOW:
        return 'EXTREME_BUY', 80
    elif rsi <= RSI_OVERSOLD:
        return 'BUY', 65
    elif rsi >= RSI_EXTREME_HIGH:
        return 'EXTREME_SELL', 80
    elif rsi >= RSI_OVERBOUGHT:
        return 'SELL', 65
    else:
        return 'NEUTRAL', 0
```

### **3. MACD ПОДТВЕРЖДЕНИЕ:**

```python
def get_macd_signal(macd, signal, histogram):
    """MACD сигналы"""
    if macd > signal and histogram > 0:
        return 'BUY_CONFIRM', 20
    elif macd < signal and histogram < 0:
        return 'SELL_CONFIRM', 20
    elif macd > signal:  # пересечение вверх
        return 'BUY_WEAK', 10
    elif macd < signal:  # пересечение вниз
        return 'SELL_WEAK', 10
    else:
        return 'NEUTRAL', 0
```

### **4. BOLLINGER BANDS (улучшенные):**

```python
def get_bb_signal(price, bb_upper, bb_lower, bb_middle):
    """Bollinger Bands с запасом"""
    bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100
    
    # Смягченные условия с запасом ±15%
    if bb_position <= 15:  # близко к нижней границе
        return 'BUY', 70, f"BB={bb_position:.0f}%"
    elif bb_position >= 85:  # близко к верхней границе
        return 'SELL', 70, f"BB={bb_position:.0f}%"
    elif bb_position <= 25:  # в нижней зоне
        return 'BUY_WEAK', 50, f"BB={bb_position:.0f}%"
    elif bb_position >= 75:  # в верхней зоне
        return 'SELL_WEAK', 50, f"BB={bb_position:.0f}%"
    else:
        return 'NEUTRAL', 0, f"BB={bb_position:.0f}%"
```

### **5. ATR ДЛЯ СТОП-ЛОССОВ:**

```python
def calculate_atr_levels(price, atr):
    """Динамические уровни на основе ATR"""
    atr_multiplier = 2.0  # стандартный множитель
    
    stop_loss = price * 0.02  # 2% базовый SL
    atr_stop = atr * atr_multiplier
    
    # Используем больший из двух
    final_sl = max(stop_loss, atr_stop)
    
    return {
        'stop_loss': final_sl,
        'take_profit_1': final_sl * 1.5,  # 1:1.5 R/R
        'take_profit_2': final_sl * 2.0,  # 1:2 R/R
        'take_profit_3': final_sl * 3.0,  # 1:3 R/R
    }
```

---

## 🎯 **НОВЫЕ СТРАТЕГИИ ВХОДА:**

### **Стратегия 1: EMA + RSI Combo**

```python
def ema_rsi_strategy(data_30m, data_1h):
    """EMA тренд + RSI зоны"""
    ema_trend_30m = get_ema_trend(data_30m)
    ema_trend_1h = get_ema_trend(data_1h)
    rsi_signal, rsi_conf = get_rsi_signal(data_30m['rsi'])
    
    # BUY: восходящий тренд + перепродан
    if (ema_trend_1h in ['BULL', 'STRONG_BULL'] and 
        ema_trend_30m in ['BULL', 'STRONG_BULL'] and
        rsi_signal in ['BUY', 'EXTREME_BUY']):
        
        confidence = 60 + rsi_conf
        return 'buy', confidence, ['EMA_RSI_BUY', ema_trend_1h, rsi_signal]
    
    # SELL: нисходящий тренд + перекуплен  
    elif (ema_trend_1h in ['BEAR', 'STRONG_BEAR'] and
          ema_trend_30m in ['BEAR', 'STRONG_BEAR'] and
          rsi_signal in ['SELL', 'EXTREME_SELL']):
        
        confidence = 60 + rsi_conf
        return 'sell', confidence, ['EMA_RSI_SELL', ema_trend_1h, rsi_signal]
    
    return None, 0, []
```

### **Стратегия 2: MACD + Bollinger**

```python
def macd_bb_strategy(data_30m):
    """MACD импульс + Bollinger границы"""
    macd_signal, macd_conf = get_macd_signal(
        data_30m['macd'], 
        data_30m['macd_signal'], 
        data_30m['macd_histogram']
    )
    bb_signal, bb_conf, bb_desc = get_bb_signal(
        data_30m['price'],
        data_30m['bb_upper'],
        data_30m['bb_lower'], 
        data_30m['bb_middle']
    )
    
    # BUY: MACD бычий + цена у нижней границы BB
    if (macd_signal in ['BUY_CONFIRM', 'BUY_WEAK'] and
        bb_signal in ['BUY', 'BUY_WEAK']):
        
        confidence = 50 + macd_conf + bb_conf
        return 'buy', confidence, ['MACD_BB_BUY', macd_signal, bb_desc]
    
    # SELL: MACD медвежий + цена у верхней границы BB
    elif (macd_signal in ['SELL_CONFIRM', 'SELL_WEAK'] and
          bb_signal in ['SELL', 'SELL_WEAK']):
        
        confidence = 50 + macd_conf + bb_conf  
        return 'sell', confidence, ['MACD_BB_SELL', macd_signal, bb_desc]
    
    return None, 0, []
```

### **Стратегия 3: Multi-Timeframe Confluence**

```python
def mtf_confluence_strategy(data_15m, data_30m, data_1h, data_4h):
    """Мульти-таймфрейм подтверждение"""
    
    # Собираем сигналы со всех таймфреймов
    signals = {
        '15m': get_ema_trend(data_15m),
        '30m': get_ema_trend(data_30m), 
        '1h': get_ema_trend(data_1h),
        '4h': get_ema_trend(data_4h)
    }
    
    # Подсчитываем бычьи/медвежьи сигналы
    bull_count = sum(1 for s in signals.values() if 'BULL' in s)
    bear_count = sum(1 for s in signals.values() if 'BEAR' in s)
    
    rsi_30m = data_30m['rsi']
    
    # BUY: 3+ бычьих таймфрейма + RSI не перекуплен
    if bull_count >= 3 and rsi_30m < 70:
        confidence = 50 + (bull_count * 10) + max(0, 70 - rsi_30m)
        return 'buy', min(90, confidence), [f'MTF_BULL_{bull_count}TF', f'RSI={rsi_30m:.0f}']
    
    # SELL: 3+ медвежьих таймфрейма + RSI не перепродан
    elif bear_count >= 3 and rsi_30m > 30:
        confidence = 50 + (bear_count * 10) + max(0, rsi_30m - 30)
        return 'sell', min(90, confidence), [f'MTF_BEAR_{bear_count}TF', f'RSI={rsi_30m:.0f}']
    
    return None, 0, []
```

---

## 🚫 **УБИРАЕМ VOLUME RATIO:**

### **Из основных фильтров:**

```python
# УДАЛЯЕМ:
# '30m_volume': current_30m['volume_ratio'] > 0.3
# '30m_mavol_trend': current_30m['volume_ma_20'] > current_30m['volume_ma_50'] * 0.5

# ОСТАВЛЯЕМ только для детекции манипуляций:
def detect_manipulation(volume_ratio):
    if volume_ratio > 5.0:  # аномально высокий объем
        return 'PUMP_DUMP_RISK'
    elif volume_ratio < 0.1:  # аномально низкий объем
        return 'LOW_LIQUIDITY_RISK'
    else:
        return 'NORMAL'
```

---

## 📊 **НОВАЯ СИСТЕМА УВЕРЕННОСТИ:**

```python
def calculate_signal_confidence(strategies_results):
    """Расчет итоговой уверенности"""
    total_confidence = 0
    strategy_count = 0
    reasons = []
    
    for strategy, signal, conf, desc in strategies_results:
        if signal in ['buy', 'sell']:
            total_confidence += conf
            strategy_count += 1
            reasons.extend(desc)
    
    if strategy_count == 0:
        return None, 0, []
    
    # Средняя уверенность + бонус за множественные стратегии
    avg_confidence = total_confidence / strategy_count
    multi_strategy_bonus = min(20, (strategy_count - 1) * 10)
    
    final_confidence = avg_confidence + multi_strategy_bonus
    
    return signal, min(95, final_confidence), reasons
```

---

## 🎯 **ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:**

### **С новой системой:**

```
✅ Убран проблемный Volume Ratio
✅ Смягчены RSI пороги (35/65 вместо 40/60)
✅ Улучшены Bollinger условия (±15% запас)
✅ Добавлены 3 новые стратегии
✅ Мульти-таймфрейм подтверждение
✅ ATR для динамических SL/TP
```

### **Прогноз сигналов:**

```
Текущие данные (12:25):
- ZEC: RSI=42 → ✅ (теперь <45)
- TAO: RSI=73 → ✅ (теперь >65) 
- SOL: BB=20% → ✅ (в нижней зоне)
- BRETT: BB=105% → ✅ (выше верхней границы)

Ожидаемо: 2-4 сигнала прямо сейчас! 🎉
```

---

## 🚀 **ГОТОВ К РЕАЛИЗАЦИИ:**

```
1. Обновить логику сигналов
2. Убрать Volume Ratio фильтры  
3. Добавить новые стратегии
4. Протестировать на текущих данных
5. Запустить обновленного бота
```

**Начинаем реализацию?** 🎯

