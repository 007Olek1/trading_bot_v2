"""
🎯 SCALPING DCA STRATEGY
Стратегия как у конкурентов:
- Зона входа с усреднением (DCA)
- Множественные TP уровни
- Плечо 20x
- Tight Stop Loss
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class EntryZone:
    """Зона входа с несколькими уровнями"""
    upper: float  # Верхняя граница (первый вход)
    lower: float  # Нижняя граница (последний вход)
    levels: int = 3  # Количество уровней входа
    
    def get_entry_levels(self) -> List[Dict]:
        """
        Получить уровни входа
        Пример: upper=3030, lower=2969, levels=3
        Результат: [3030, 3000, 2969]
        """
        step = (self.upper - self.lower) / (self.levels - 1)
        
        entries = []
        for i in range(self.levels):
            price = self.upper - (step * i)
            # Распределение капитала: 30%, 40%, 30%
            if i == 0:
                percent = 30
            elif i == self.levels - 1:
                percent = 30
            else:
                percent = 40
            
            entries.append({
                'price': price,
                'percent': percent,
                'filled': False
            })
        
        return entries
    
    def get_average_entry(self) -> float:
        """Средняя точка входа"""
        return (self.upper + self.lower) / 2


@dataclass
class TPLevel:
    """Уровень Take Profit"""
    price: float
    percent_from_entry: float
    close_percent: int  # Процент позиции для закрытия
    roe: float  # Ожидаемый ROE


class ScalpingDCAStrategy:
    """
    Скальпинг стратегия с DCA входами как у конкурентов
    
    Особенности:
    - Зона входа (entry zone) вместо одной точки
    - Множественные TP уровни (6 уровней)
    - Плечо 20x
    - Частичное закрытие позиции на каждом TP
    - Tight Stop Loss (-2%)
    """
    
    def __init__(self, leverage: int = 20, logger: logging.Logger = None):
        self.leverage = leverage
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_signal(self, 
                      df: pd.DataFrame, 
                      indicators: Dict,
                      current_price: float) -> Optional[Dict]:
        """
        Анализ и генерация сигнала
        
        Returns:
            {
                'direction': 'LONG' or 'SHORT',
                'entry_zone': EntryZone,
                'tp_levels': List[TPLevel],
                'stop_loss': float,
                'confidence': float,
                'expected_roe': float
            }
        """
        if len(df) < 50:
            return None
        
        # Определяем направление
        direction = self._detect_direction(df, indicators)
        if direction is None:
            return None
        
        # Создаем зону входа
        entry_zone = self._calculate_entry_zone(current_price, direction)
        
        # Рассчитываем TP уровни
        tp_levels = self._calculate_tp_levels(
            entry_zone.get_average_entry(), 
            direction
        )
        
        # Рассчитываем Stop Loss
        stop_loss = self._calculate_stop_loss(
            entry_zone.get_average_entry(), 
            direction
        )
        
        # Оцениваем уверенность
        confidence = self._calculate_confidence(df, indicators, direction)
        
        # Ожидаемый ROE (средний по TP уровням)
        expected_roe = sum(tp.roe for tp in tp_levels) / len(tp_levels)
        
        return {
            'direction': direction,
            'entry_zone': entry_zone,
            'tp_levels': tp_levels,
            'stop_loss': stop_loss,
            'confidence': confidence,
            'expected_roe': expected_roe,
            'leverage': self.leverage
        }
    
    def _detect_direction(self, 
                         df: pd.DataFrame, 
                         indicators: Dict) -> Optional[str]:
        """
        Определить направление сделки
        
        Условия для LONG:
        - EMA20 > EMA50
        - RSI выходит из перепроданности (30-50)
        - Объем выше среднего
        - Цена отбилась от поддержки
        
        Условия для SHORT:
        - EMA20 < EMA50
        - RSI выходит из перекупленности (50-70)
        - Объем выше среднего
        - Цена отбилась от сопротивления
        """
        close = df['close'].iloc[-1]
        ema_short = indicators['ema_short'].iloc[-1]
        ema_medium = indicators['ema_medium'].iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        volume_ratio = indicators['volume_ratio'].iloc[-1]
        
        # LONG условия
        if (ema_short > ema_medium and
            30 < rsi < 50 and
            volume_ratio > 1.5 and
            close > ema_short):
            return 'LONG'
        
        # SHORT условия
        elif (ema_short < ema_medium and
              50 < rsi < 70 and
              volume_ratio > 1.5 and
              close < ema_short):
            return 'SHORT'
        
        return None
    
    def _calculate_entry_zone(self, 
                             current_price: float, 
                             direction: str,
                             zone_width_percent: float = 2.0) -> EntryZone:
        """
        Рассчитать зону входа
        
        Для LONG:
        - Верхняя граница: текущая цена
        - Нижняя граница: -2% от текущей цены
        
        Для SHORT:
        - Верхняя граница: +2% от текущей цены
        - Нижняя граница: текущая цена
        """
        zone_width = current_price * (zone_width_percent / 100)
        
        if direction == 'LONG':
            upper = current_price
            lower = current_price - zone_width
        else:  # SHORT
            upper = current_price + zone_width
            lower = current_price
        
        return EntryZone(upper=upper, lower=lower, levels=3)
    
    def _calculate_tp_levels(self, 
                            avg_entry: float, 
                            direction: str) -> List[TPLevel]:
        """
        Рассчитать уровни Take Profit
        
        6 уровней как у конкурентов:
        TP1: +2.0% (закрыть 30%)
        TP2: +3.0% (закрыть 25%)
        TP3: +4.0% (закрыть 20%)
        TP4: +5.0% (закрыть 15%)
        TP5: +6.0% (закрыть 10%)
        TP6: +7.0% (закрыть остаток)
        """
        tp_configs = [
            {'percent': 2.0, 'close': 30},
            {'percent': 3.0, 'close': 25},
            {'percent': 4.0, 'close': 20},
            {'percent': 5.0, 'close': 15},
            {'percent': 6.0, 'close': 10},
            {'percent': 7.0, 'close': 100},  # Остаток
        ]
        
        tp_levels = []
        
        for config in tp_configs:
            percent = config['percent']
            
            if direction == 'LONG':
                price = avg_entry * (1 + percent / 100)
            else:  # SHORT
                price = avg_entry * (1 - percent / 100)
            
            roe = percent * self.leverage
            
            tp_levels.append(TPLevel(
                price=price,
                percent_from_entry=percent,
                close_percent=config['close'],
                roe=roe
            ))
        
        return tp_levels
    
    def _calculate_stop_loss(self, 
                            avg_entry: float, 
                            direction: str,
                            sl_percent: float = 2.0) -> float:
        """
        Рассчитать Stop Loss
        
        Tight SL: -2% от средней точки входа
        """
        if direction == 'LONG':
            return avg_entry * (1 - sl_percent / 100)
        else:  # SHORT
            return avg_entry * (1 + sl_percent / 100)
    
    def _calculate_confidence(self, 
                             df: pd.DataFrame, 
                             indicators: Dict,
                             direction: str) -> float:
        """
        Рассчитать уверенность в сигнале (0.0 - 1.0)
        """
        confidence = 0.0
        
        # Проверяем различные факторы
        ema_short = indicators['ema_short'].iloc[-1]
        ema_medium = indicators['ema_medium'].iloc[-1]
        ema_long = indicators['ema_long'].iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        volume_ratio = indicators['volume_ratio'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        
        if direction == 'LONG':
            # Тренд
            if ema_short > ema_medium > ema_long:
                confidence += 0.3
            elif ema_short > ema_medium:
                confidence += 0.15
            
            # RSI
            if 30 < rsi < 45:
                confidence += 0.2
            elif 45 <= rsi < 50:
                confidence += 0.1
            
            # Объем
            if volume_ratio > 2.0:
                confidence += 0.3
            elif volume_ratio > 1.5:
                confidence += 0.2
            
            # Сила тренда (ADX)
            if adx > 25:
                confidence += 0.2
        
        else:  # SHORT
            # Аналогично для SHORT
            if ema_short < ema_medium < ema_long:
                confidence += 0.3
            elif ema_short < ema_medium:
                confidence += 0.15
            
            if 55 < rsi < 70:
                confidence += 0.2
            elif 50 < rsi <= 55:
                confidence += 0.1
            
            if volume_ratio > 2.0:
                confidence += 0.3
            elif volume_ratio > 1.5:
                confidence += 0.2
            
            if adx > 25:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def format_signal_message(self, signal: Dict, symbol: str) -> str:
        """
        Форматировать сигнал для Telegram (как у конкурентов)
        
        Пример:
        🌹 #ETH/USDT
        
        ⏩ Long 
        ⚜️ Leverage: 20x
        
        ✳️ Entry: 3030 - 2969
        
        🥂 Target: 3060 - 3091 - 3121 - 3151 - 3182 - 3212
        
        ❌ StopLoss: 2939
        """
        entry_zone = signal['entry_zone']
        tp_levels = signal['tp_levels']
        stop_loss = signal['stop_loss']
        direction = signal['direction']
        
        # Эмодзи для направления
        direction_emoji = "⏩" if direction == "LONG" else "⏬"
        
        # Форматируем entry zone
        entry_str = f"{entry_zone.upper:.2f} - {entry_zone.lower:.2f}"
        
        # Форматируем TP уровни
        tp_str = " - ".join([f"{tp.price:.2f}" for tp in tp_levels])
        
        message = f"""🌹 #{symbol}

{direction_emoji} {direction}
⚜️ Leverage: {signal['leverage']}x

✳️ Entry: {entry_str}

🥂 Target: {tp_str}

❌ StopLoss: {stop_loss:.2f}

📊 Confidence: {signal['confidence']:.0%}
💰 Expected ROE: +{signal['expected_roe']:.0f}%"""
        
        return message


# Пример использования
if __name__ == "__main__":
    # Создаем стратегию
    strategy = ScalpingDCAStrategy(leverage=20)
    
    # Пример: ETH/USDT
    current_price = 3000
    
    # Создаем entry zone
    entry_zone = EntryZone(upper=3030, lower=2969, levels=3)
    print("Entry Levels:")
    for level in entry_zone.get_entry_levels():
        print(f"  ${level['price']:.2f} - {level['percent']}%")
    
    print(f"\nAverage Entry: ${entry_zone.get_average_entry():.2f}")
    
    # Рассчитываем TP уровни
    tp_levels = strategy._calculate_tp_levels(3000, 'LONG')
    print("\nTP Levels:")
    for i, tp in enumerate(tp_levels, 1):
        print(f"  TP{i}: ${tp.price:.2f} (+{tp.percent_from_entry:.1f}%) "
              f"→ ROE: +{tp.roe:.0f}% | Close: {tp.close_percent}%")
    
    # Stop Loss
    sl = strategy._calculate_stop_loss(3000, 'LONG')
    print(f"\nStop Loss: ${sl:.2f} (-2.0%) → ROE: -40%")
    
    # Risk/Reward
    print("\nRisk/Reward:")
    for i, tp in enumerate(tp_levels, 1):
        rr = tp.percent_from_entry / 2.0  # SL = 2%
        print(f"  TP{i}: R/R = 1:{rr:.1f}")
