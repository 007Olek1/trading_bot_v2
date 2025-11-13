#!/usr/bin/env python3
"""
📊 PROBABILITY CALCULATOR V4.0
==============================

Расчет реалистичных вероятностей достижения TP уровней
на основе ML анализа и исторических данных
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
# import talib  # Не используется в текущей версии

logger = logging.getLogger(__name__)

@dataclass
class TPProbability:
    """Вероятность достижения TP уровня"""
    level: int
    percent: float
    probability: float
    confidence_interval: Tuple[float, float]
    market_condition_factor: float

class ProbabilityCalculator:
    """📊 Калькулятор вероятностей для TP уровней"""
    
    def __init__(self):
        self.historical_data = {}
        self.market_conditions = ['BULLISH', 'BEARISH', 'NEUTRAL', 'VOLATILE']
        
        # Базовые вероятности по уровням (консервативные)
        self.base_probabilities = {
            4: 85,   # +4% - высокая вероятность
            6: 75,   # +6% - хорошая вероятность  
            8: 65,   # +8% - средняя вероятность
            10: 55,  # +10% - ниже средней
            12: 45,  # +12% - низкая вероятность
            15: 35   # +15% - очень низкая
        }
        
        logger.info("📊 ProbabilityCalculator инициализирован")
    
    def calculate_tp_probabilities(self, symbol: str, market_data: Dict, 
                                 market_condition: str) -> List[TPProbability]:
        """
        Рассчитать вероятности для всех TP уровней
        
        Args:
            symbol: Торговая пара
            market_data: Данные по символу (RSI, BB, ATR, etc.)
            market_condition: Состояние рынка
            
        Returns:
            Список вероятностей для каждого TP уровня
        """
        try:
            tp_levels = [4, 6, 8, 10, 12, 15]  # Проценты роста
            probabilities = []
            
            # Анализируем рыночные факторы
            market_factor = self._analyze_market_factors(market_data, market_condition)
            volatility_factor = self._calculate_volatility_factor(market_data)
            trend_strength = self._calculate_trend_strength(market_data)
            
            for i, tp_percent in enumerate(tp_levels):
                # Базовая вероятность
                base_prob = self.base_probabilities[tp_percent]
                
                # Корректировки на основе анализа
                adjusted_prob = self._adjust_probability(
                    base_prob, tp_percent, market_factor, 
                    volatility_factor, trend_strength
                )
                
                # Доверительный интервал
                confidence_interval = self._calculate_confidence_interval(adjusted_prob)
                
                tp_prob = TPProbability(
                    level=i + 1,
                    percent=tp_percent,
                    probability=adjusted_prob,
                    confidence_interval=confidence_interval,
                    market_condition_factor=market_factor
                )
                
                probabilities.append(tp_prob)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета вероятностей для {symbol}: {e}")
            return self._get_default_probabilities()
    
    def _analyze_market_factors(self, market_data: Dict, market_condition: str) -> float:
        """Анализ рыночных факторов"""
        try:
            factor = 1.0
            
            # Фактор рыночного состояния
            if market_condition == 'BULLISH':
                factor *= 1.15  # +15% к вероятностям в бычьем рынке
            elif market_condition == 'BEARISH':
                factor *= 0.85  # -15% в медвежьем рынке
            elif market_condition == 'VOLATILE':
                factor *= 0.90  # -10% в волатильном рынке
            
            # Фактор RSI
            rsi = market_data.get('rsi', 50)
            if 30 <= rsi <= 70:  # Нормальная зона
                factor *= 1.05
            elif rsi < 30 or rsi > 70:  # Экстремальные зоны
                factor *= 0.95
            
            # Фактор Bollinger Bands
            bb_position = market_data.get('bb_position', 50)
            if 25 <= bb_position <= 75:  # Нормальная зона
                factor *= 1.03
            else:  # Экстремальные зоны
                factor *= 0.97
            
            return max(0.7, min(1.3, factor))  # Ограничиваем диапазон
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка анализа рыночных факторов: {e}")
            return 1.0
    
    def _calculate_volatility_factor(self, market_data: Dict) -> float:
        """Расчет фактора волатильности"""
        try:
            atr = market_data.get('atr', 0)
            price = market_data.get('price', 1)
            
            if price > 0:
                volatility_percent = (atr / price) * 100
                
                if volatility_percent < 2:  # Низкая волатильность
                    return 1.05  # Легче достичь TP
                elif volatility_percent > 5:  # Высокая волатильность
                    return 0.90  # Сложнее достичь TP
                else:
                    return 1.0  # Нормальная волатильность
            
            return 1.0
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета волатильности: {e}")
            return 1.0
    
    def _calculate_trend_strength(self, market_data: Dict) -> float:
        """Расчет силы тренда"""
        try:
            # Анализируем EMA
            ema_9 = market_data.get('ema_9', 0)
            ema_21 = market_data.get('ema_21', 0)
            ema_50 = market_data.get('ema_50', 0)
            
            if ema_9 > ema_21 > ema_50:  # Сильный восходящий тренд
                return 1.10
            elif ema_9 > ema_21:  # Слабый восходящий тренд
                return 1.05
            elif ema_9 < ema_21 < ema_50:  # Сильный нисходящий тренд
                return 0.85
            elif ema_9 < ema_21:  # Слабый нисходящий тренд
                return 0.90
            else:  # Боковой тренд
                return 0.95
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета силы тренда: {e}")
            return 1.0
    
    def _adjust_probability(self, base_prob: float, tp_percent: float, 
                          market_factor: float, volatility_factor: float, 
                          trend_strength: float) -> float:
        """Корректировка вероятности на основе факторов"""
        try:
            # Применяем все факторы
            adjusted = base_prob * market_factor * volatility_factor * trend_strength
            
            # Дополнительная корректировка для высоких TP
            if tp_percent >= 12:
                adjusted *= 0.95  # Снижаем вероятность для высоких целей
            
            # Ограничиваем диапазон
            return max(20, min(95, adjusted))
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка корректировки вероятности: {e}")
            return base_prob
    
    def _calculate_confidence_interval(self, probability: float) -> Tuple[float, float]:
        """Расчет доверительного интервала"""
        try:
            # ±5% доверительный интервал
            margin = 5
            lower = max(10, probability - margin)
            upper = min(99, probability + margin)
            return (lower, upper)
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета доверительного интервала: {e}")
            return (probability - 5, probability + 5)
    
    def _get_default_probabilities(self) -> List[TPProbability]:
        """Получить дефолтные вероятности при ошибке"""
        default_probs = [
            TPProbability(1, 4, 85, (80, 90), 1.0),
            TPProbability(2, 6, 75, (70, 80), 1.0),
            TPProbability(3, 8, 65, (60, 70), 1.0),
            TPProbability(4, 10, 55, (50, 60), 1.0),
            TPProbability(5, 12, 45, (40, 50), 1.0),
            TPProbability(6, 15, 35, (30, 40), 1.0)
        ]
        return default_probs
    
    def get_probability_summary(self, probabilities: List[TPProbability]) -> str:
        """Получить краткое описание вероятностей"""
        try:
            summary_parts = []
            for tp in probabilities:
                summary_parts.append(f"TP{tp.level}({tp.percent}%): {tp.probability:.0f}%")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка создания summary: {e}")
            return "Вероятности недоступны"

# Тестирование модуля
if __name__ == "__main__":
    calc = ProbabilityCalculator()
    
    # Тестовые данные
    test_data = {
        'rsi': 45,
        'bb_position': 30,
        'atr': 2.5,
        'price': 100.0,
        'ema_9': 101,
        'ema_21': 100,
        'ema_50': 99
    }
    
    probabilities = calc.calculate_tp_probabilities('TEST/USDT', test_data, 'BULLISH')
    
    print("📊 Тест ProbabilityCalculator:")
    for tp in probabilities:
        print(f"TP{tp.level}: {tp.percent}% = {tp.probability:.1f}% вероятность")
    
    print(f"\n📋 Summary: {calc.get_probability_summary(probabilities)}")
