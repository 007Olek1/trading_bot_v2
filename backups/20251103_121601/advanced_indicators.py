#!/usr/bin/env python3
"""
🎯 ADVANCED TECHNICAL INDICATORS
Расширенные технические индикаторы:
- Ichimoku Cloud
- Fibonacci Retracement
- Support/Resistance Levels
- Pattern Recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class IchimokuCloud:
    """Ишимоку облако"""
    tenkan_sen: float  # Conversion Line (9 periods)
    kijun_sen: float   # Base Line (26 periods)
    senkou_span_a: float  # Leading Span A
    senkou_span_b: float  # Leading Span B (52 periods)
    chikou_span: float    # Lagging Span
    cloud_top: float
    cloud_bottom: float
    trend: str  # 'bullish', 'bearish', 'neutral'
    signal: str  # 'buy', 'sell', 'hold'


@dataclass
class FibonacciLevels:
    """Фибоначчи уровни"""
    level_0: float  # 0% (высокая)
    level_236: float  # 23.6%
    level_382: float  # 38.2%
    level_500: float  # 50%
    level_618: float  # 61.8%
    level_786: float  # 78.6%
    level_100: float  # 100% (низкая)
    current_position: float  # Текущая позиция цены в фибоначчи (%)


@dataclass
class SupportResistance:
    """Уровни поддержки и сопротивления"""
    support_levels: List[float]
    resistance_levels: List[float]
    current_price: float
    nearest_support: float
    nearest_resistance: float
    support_distance_pct: float
    resistance_distance_pct: float
    strength: str  # 'strong', 'medium', 'weak'


class AdvancedIndicators:
    """🎯 Расширенные технические индикаторы"""
    
    def __init__(self):
        self.ichimoku_periods = {
            'conversion': 9,
            'base': 26,
            'leading_span_b': 52,
            'displacement': 26
        }
        logger.info("🎯 Advanced Indicators инициализированы")
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> Optional[IchimokuCloud]:
        """Расчет Ишимоку облака"""
        try:
            if len(df) < 52:
                return None
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 за 9 периодов
            periods_9_high = np.max(high[-9:])
            periods_9_low = np.min(low[-9:])
            tenkan_sen = (periods_9_high + periods_9_low) / 2
            
            # Kijun-sen (Base Line): (highest high + lowest low) / 2 за 26 периодов
            periods_26_high = np.max(high[-26:])
            periods_26_low = np.min(low[-26:])
            kijun_sen = (periods_26_high + periods_26_low) / 2
            
            # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2 (смещено вперед на 26 периодов)
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            
            # Senkou Span B: (highest high + lowest low) / 2 за 52 периода (смещено вперед на 26)
            periods_52_high = np.max(high[-52:])
            periods_52_low = np.min(low[-52:])
            senkou_span_b = (periods_52_high + periods_52_low) / 2
            
            # Chikou Span: текущая цена (смещена назад на 26 периодов)
            chikou_span = close[-26] if len(close) >= 26 else close[-1]
            
            cloud_top = max(senkou_span_a, senkou_span_b)
            cloud_bottom = min(senkou_span_a, senkou_span_b)
            
            current_price = close[-1]
            
            # Определение тренда
            if current_price > cloud_top:
                trend = 'bullish'
            elif current_price < cloud_bottom:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Сигнал
            if tenkan_sen > kijun_sen and current_price > cloud_top:
                signal = 'buy'
            elif tenkan_sen < kijun_sen and current_price < cloud_bottom:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return IchimokuCloud(
                tenkan_sen=tenkan_sen,
                kijun_sen=kijun_sen,
                senkou_span_a=senkou_span_a,
                senkou_span_b=senkou_span_b,
                chikou_span=chikou_span,
                cloud_top=cloud_top,
                cloud_bottom=cloud_bottom,
                trend=trend,
                signal=signal
            )
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета Ichimoku: {e}")
            return None
    
    def calculate_fibonacci(self, df: pd.DataFrame, lookback_periods: int = 100) -> Optional[FibonacciLevels]:
        """Расчет уровней Фибоначчи"""
        try:
            if len(df) < lookback_periods:
                lookback_periods = len(df)
            
            # Находим максимум и минимум за период
            high = df['high'].values[-lookback_periods:]
            low = df['low'].values[-lookback_periods:]
            
            price_high = np.max(high)
            price_low = np.min(low)
            
            price_range = price_high - price_low
            if price_range == 0:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Уровни Фибоначчи (нисходящий тренд от high к low)
            level_0 = price_high
            level_236 = price_high - (price_range * 0.236)
            level_382 = price_high - (price_range * 0.382)
            level_500 = price_high - (price_range * 0.500)
            level_618 = price_high - (price_range * 0.618)
            level_786 = price_high - (price_range * 0.786)
            level_100 = price_low
            
            # Позиция текущей цены
            if current_price >= price_high:
                current_position = 0.0
            elif current_price <= price_low:
                current_position = 100.0
            else:
                current_position = ((price_high - current_price) / price_range) * 100
            
            return FibonacciLevels(
                level_0=level_0,
                level_236=level_236,
                level_382=level_382,
                level_500=level_500,
                level_618=level_618,
                level_786=level_786,
                level_100=level_100,
                current_position=current_position
            )
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета Fibonacci: {e}")
            return None
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20, min_touches: int = 2) -> Optional[SupportResistance]:
        """Расчет уровней поддержки и сопротивления"""
        try:
            if len(df) < window * 2:
                return None
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            current_price = close[-1]
            
            # Находим локальные максимумы и минимумы
            support_levels = []
            resistance_levels = []
            
            # Используем скользящее окно для поиска экстремумов
            for i in range(window, len(df) - window):
                # Локальный максимум (сопротивление)
                if high[i] == np.max(high[i-window:i+window+1]):
                    resistance_levels.append(high[i])
                
                # Локальный минимум (поддержка)
                if low[i] == np.min(low[i-window:i+window+1]):
                    support_levels.append(low[i])
            
            # Фильтруем уровни (убираем слишком близкие)
            resistance_levels = self._filter_levels(resistance_levels, current_price)
            support_levels = self._filter_levels(support_levels, current_price)
            
            # Находим ближайшие уровни
            nearest_support = max([s for s in support_levels if s < current_price], default=0)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 2)
            
            if nearest_support == 0:
                nearest_support = current_price * 0.95  # Дефолт
            
            if nearest_resistance == current_price * 2:
                nearest_resistance = current_price * 1.05  # Дефолт
            
            support_distance_pct = ((current_price - nearest_support) / current_price) * 100
            resistance_distance_pct = ((nearest_resistance - current_price) / current_price) * 100
            
            # Определяем силу уровней по количеству касаний
            support_strength = len([s for s in support_levels if abs(s - nearest_support) / nearest_support < 0.01])
            resistance_strength = len([r for r in resistance_levels if abs(r - nearest_resistance) / nearest_resistance < 0.01])
            
            total_strength = support_strength + resistance_strength
            if total_strength >= min_touches * 2:
                strength = 'strong'
            elif total_strength >= min_touches:
                strength = 'medium'
            else:
                strength = 'weak'
            
            return SupportResistance(
                support_levels=sorted(support_levels, reverse=True)[:5],  # Топ-5
                resistance_levels=sorted(resistance_levels)[:5],  # Топ-5
                current_price=current_price,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                support_distance_pct=support_distance_pct,
                resistance_distance_pct=resistance_distance_pct,
                strength=strength
            )
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета Support/Resistance: {e}")
            return None
    
    def _filter_levels(self, levels: List[float], current_price: float, tolerance_pct: float = 1.0) -> List[float]:
        """Фильтрация уровней (убираем слишком близкие)"""
        if not levels:
            return []
        
        levels = sorted(levels)
        filtered = [levels[0]]
        
        for level in levels[1:]:
            # Проверяем расстояние до предыдущего уровня
            if abs(level - filtered[-1]) / filtered[-1] > tolerance_pct / 100:
                filtered.append(level)
        
        return filtered
    
    def get_all_indicators(self, df: pd.DataFrame) -> Dict[str, any]:
        """Получить все расширенные индикаторы"""
        indicators = {}
        
        ichimoku = self.calculate_ichimoku(df)
        if ichimoku:
            indicators['ichimoku'] = {
                'tenkan_sen': ichimoku.tenkan_sen,
                'kijun_sen': ichimoku.kijun_sen,
                'cloud_top': ichimoku.cloud_top,
                'cloud_bottom': ichimoku.cloud_bottom,
                'trend': ichimoku.trend,
                'signal': ichimoku.signal
            }
        
        fibonacci = self.calculate_fibonacci(df)
        if fibonacci:
            indicators['fibonacci'] = {
                'level_382': fibonacci.level_382,
                'level_500': fibonacci.level_500,
                'level_618': fibonacci.level_618,
                'current_position': fibonacci.current_position
            }
        
        support_resistance = self.calculate_support_resistance(df)
        if support_resistance:
            indicators['support_resistance'] = {
                'nearest_support': support_resistance.nearest_support,
                'nearest_resistance': support_resistance.nearest_resistance,
                'support_distance_pct': support_resistance.support_distance_pct,
                'resistance_distance_pct': support_resistance.resistance_distance_pct,
                'strength': support_resistance.strength
            }
        
        return indicators

    # Совместимость с тестами: алиас старого имени метода
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20, min_touches: int = 2) -> Optional[SupportResistance]:
        return self.calculate_support_resistance(df, window=window, min_touches=min_touches)


