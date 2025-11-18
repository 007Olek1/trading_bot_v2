"""
🎯 СТРАТЕГИИ - Торговые стратегии
💹 Тренд + Объём + Bollinger
🎭 Детектор манипуляций
🌍 Анализ глобального тренда
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class TrendVolumeStrategy:
    """
    💹 УНИВЕРСАЛЬНАЯ СТРАТЕГИЯ v2.0
    
    📊 EMA — Направление тренда (20/50/200)
    🎯 RSI + MACD — Точки входа
    📈 Volume Profile + ATR — Подтверждение и риск-менеджмент
    🔔 Bollinger Bands — Зоны перекупленности/перепроданности
    """
    
    def __init__(self, params: Dict):
        self.params = params
    
    def analyze(self, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Анализ сигнала с использованием всех индикаторов
        
        Логика:
        1. EMA определяет направление тренда
        2. RSI + MACD дают точку входа
        3. Volume Profile подтверждает силу движения
        4. ATR определяет волатильность для риск-менеджмента
        5. Bollinger Bands показывают экстремальные зоны
        """
        if not indicators or len(df) < 50:
            return None
        
        # ═══════════════════════════════════════════════════════════════
        # 1️⃣ EMA — НАПРАВЛЕНИЕ ТРЕНДА
        # ═══════════════════════════════════════════════════════════════
        close = df['close'].iloc[-1]
        ema_short = indicators['ema_short'].iloc[-1]    # EMA 20
        ema_medium = indicators['ema_medium'].iloc[-1]  # EMA 50
        ema_long = indicators['ema_long'].iloc[-1]      # EMA 200
        
        # Определяем тренд по EMA
        trend = None
        ema_strength = 0.0
        
        if ema_short > ema_medium > ema_long:
            trend = 'LONG'
            # Сила тренда = расстояние между EMA
            ema_spread = ((ema_short - ema_long) / ema_long) * 100
            ema_strength = min(ema_spread / 2.0, 1.0)  # Нормализуем 0-1
        elif ema_short < ema_medium < ema_long:
            trend = 'SHORT'
            ema_spread = ((ema_long - ema_short) / ema_long) * 100
            ema_strength = min(ema_spread / 2.0, 1.0)
        
        if trend is None:
            return {'direction': None, 'confidence': 0.0}
        
        # ═══════════════════════════════════════════════════════════════
        # 2️⃣ RSI + MACD — ТОЧКИ ВХОДА
        # ═══════════════════════════════════════════════════════════════
        rsi = indicators['rsi'].iloc[-1]
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['macd_signal'].iloc[-1]
        macd_hist = macd - macd_signal
        
        entry_signal = 0.0
        
        if trend == 'LONG':
            # RSI: перепроданность (хороший вход)
            if rsi < self.params['rsi_oversold']:
                entry_signal += 0.3
            elif rsi < 50:  # RSI ниже середины
                entry_signal += 0.15
            
            # MACD: бычий кроссовер
            if macd > macd_signal and macd_hist > 0:
                entry_signal += 0.3
            elif macd > macd_signal:  # Просто выше сигнальной
                entry_signal += 0.15
        
        elif trend == 'SHORT':
            # RSI: перекупленность (хороший вход)
            if rsi > self.params['rsi_overbought']:
                entry_signal += 0.3
            elif rsi > 50:  # RSI выше середины
                entry_signal += 0.15
            
            # MACD: медвежий кроссовер
            if macd < macd_signal and macd_hist < 0:
                entry_signal += 0.3
            elif macd < macd_signal:  # Просто ниже сигнальной
                entry_signal += 0.15
        
        # ═══════════════════════════════════════════════════════════════
        # 3️⃣ VOLUME PROFILE — ПОДТВЕРЖДЕНИЕ
        # ═══════════════════════════════════════════════════════════════
        volume_ratio = indicators['volume_ratio'].iloc[-1]
        
        volume_confirmation = 0.0
        if volume_ratio > self.params['volume_spike_multiplier'] * 1.5:  # Сильный объем
            volume_confirmation = 0.25
        elif volume_ratio > self.params['volume_spike_multiplier']:  # Нормальный объем
            volume_confirmation = 0.15
        
        # ═══════════════════════════════════════════════════════════════
        # 4️⃣ ATR — ВОЛАТИЛЬНОСТЬ И РИСК-МЕНЕДЖМЕНТ
        # ═══════════════════════════════════════════════════════════════
        atr = indicators['atr'].iloc[-1]
        atr_percent = (atr / close) * 100
        
        # ATR влияет на уверенность:
        # - Низкая волатильность (< 1%) = меньше риска = выше уверенность
        # - Высокая волатильность (> 3%) = больше риска = ниже уверенность
        atr_factor = 1.0
        if atr_percent < 1.0:
            atr_factor = 1.1  # Бонус за низкую волатильность
        elif atr_percent > 3.0:
            atr_factor = 0.9  # Штраф за высокую волатильность
        
        # ═══════════════════════════════════════════════════════════════
        # 5️⃣ BOLLINGER BANDS — ЭКСТРЕМАЛЬНЫЕ ЗОНЫ
        # ═══════════════════════════════════════════════════════════════
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        
        bb_signal = 0.0
        bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        
        if trend == 'LONG':
            # Цена у нижней полосы = хороший вход для LONG
            if close <= bb_lower * 1.01:
                bb_signal = 0.2
            elif bb_position < 0.3:  # В нижней трети
                bb_signal = 0.1
        
        elif trend == 'SHORT':
            # Цена у верхней полосы = хороший вход для SHORT
            if close >= bb_upper * 0.99:
                bb_signal = 0.2
            elif bb_position > 0.7:  # В верхней трети
                bb_signal = 0.1
        
        # ═══════════════════════════════════════════════════════════════
        # 📊 ИТОГОВАЯ УВЕРЕННОСТЬ
        # ═══════════════════════════════════════════════════════════════
        base_confidence = (
            ema_strength * 0.25 +      # 25% - направление тренда
            entry_signal * 0.35 +       # 35% - точка входа (RSI + MACD)
            volume_confirmation * 0.25 + # 25% - подтверждение объемом
            bb_signal * 0.15            # 15% - зона BB
        )
        
        # Применяем ATR фактор
        final_confidence = base_confidence * atr_factor
        final_confidence = min(final_confidence, 1.0)
        
        # Минимальный порог для сигнала
        if final_confidence < 0.4:
            return {'direction': None, 'confidence': 0.0}
        
        return {
            'direction': trend,
            'confidence': final_confidence,
            'details': {
                'trend': trend,
                'ema_strength': ema_strength,
                'rsi': rsi,
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish',
                'volume_ratio': volume_ratio,
                'atr_percent': atr_percent,
                'bb_position': bb_position,
                'entry_score': entry_signal,
                'volume_score': volume_confirmation,
                'bb_score': bb_signal
            }
        }


class ManipulationDetector:
    """
    🎭 Детектор манипуляций и ложных пробоев
    - Обнаружение pump & dump
    - Определение ложных пробоев
    - Анализ аномальных объёмов
    """
    
    def analyze(self, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """Анализ на манипуляции"""
        if not indicators or len(df) < 20:
            return None
        
        # Последние свечи
        last_candles = df.tail(10)
        
        # Проверяем на pump (резкий рост с большим объёмом)
        price_change = (last_candles['close'].iloc[-1] - last_candles['close'].iloc[0]) / last_candles['close'].iloc[0]
        avg_volume = indicators['volume_sma'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        # Признаки манипуляции
        is_pump = price_change > 0.05 and current_volume > avg_volume * 3  # +5% и объём x3
        is_dump = price_change < -0.05 and current_volume > avg_volume * 3  # -5% и объём x3
        
        # Ложный пробой (цена вернулась после пробоя)
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        false_breakout_up = prev_close > bb_upper and close < bb_upper
        false_breakout_down = prev_close < bb_lower and close > bb_lower
        
        # Если обнаружена манипуляция, возвращаем противоположный сигнал
        if is_pump or false_breakout_up:
            return {
                'direction': 'SHORT',
                'confidence': 0.7,
                'details': {
                    'manipulation_type': 'pump' if is_pump else 'false_breakout_up',
                    'price_change': price_change,
                    'volume_ratio': current_volume / avg_volume
                }
            }
        
        if is_dump or false_breakout_down:
            return {
                'direction': 'LONG',
                'confidence': 0.7,
                'details': {
                    'manipulation_type': 'dump' if is_dump else 'false_breakout_down',
                    'price_change': price_change,
                    'volume_ratio': current_volume / avg_volume
                }
            }
        
        return {'direction': None, 'confidence': 0.0}


class GlobalTrendAnalyzer:
    """
    🌍 Анализ глобального тренда (4h + 1D)
    - Определение долгосрочного направления
    - Фильтрация сделок против тренда
    """
    
    def analyze(self, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """Анализ глобального тренда"""
        if not indicators or len(df) < 100:
            return None
        
        # EMA для определения тренда
        ema_short = indicators['ema_short'].iloc[-1]
        ema_medium = indicators['ema_medium'].iloc[-1]
        ema_long = indicators['ema_long'].iloc[-1]
        
        # ADX для силы тренда
        adx = indicators['adx'].iloc[-1]
        di_plus = indicators['di_plus'].iloc[-1]
        di_minus = indicators['di_minus'].iloc[-1]
        
        # MACD для подтверждения
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['macd_signal'].iloc[-1]
        
        # Определяем тренд
        trend = None
        confidence = 0.0
        
        # Восходящий тренд
        if ema_short > ema_medium > ema_long:
            trend = 'LONG'
            confidence += 0.3
            
            # ADX подтверждает силу тренда
            if adx > 25 and di_plus > di_minus:
                confidence += 0.3
            
            # MACD бычий
            if macd > macd_signal:
                confidence += 0.2
            
            # Расстояние между EMA
            ema_spread = (ema_short - ema_long) / ema_long
            if ema_spread > 0.02:  # 2% разница
                confidence += 0.2
        
        # Нисходящий тренд
        elif ema_short < ema_medium < ema_long:
            trend = 'SHORT'
            confidence += 0.3
            
            # ADX подтверждает силу тренда
            if adx > 25 and di_minus > di_plus:
                confidence += 0.3
            
            # MACD медвежий
            if macd < macd_signal:
                confidence += 0.2
            
            # Расстояние между EMA
            ema_spread = (ema_long - ema_short) / ema_long
            if ema_spread > 0.02:  # 2% разница
                confidence += 0.2
        
        return {
            'direction': trend,
            'confidence': min(confidence, 1.0),
            'details': {
                'adx': adx,
                'trend_strength': 'strong' if adx > 25 else 'weak',
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish'
            }
        }
