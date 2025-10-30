#!/usr/bin/env python3
"""
🏆 STRATEGY EVALUATOR V4.0
===========================

Оценка качества торговых стратегий и сигналов
Шкала оценки: 0-20 баллов (как у конкурентов)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StrategyScore:
    """Оценка стратегии"""
    total_score: float  # 0-20
    components: Dict[str, float]
    quality_level: str  # EXCELLENT, GOOD, AVERAGE, POOR
    recommendation: str

class StrategyEvaluator:
    """🏆 Оценщик качества торговых стратегий"""
    
    def __init__(self):
        # Веса компонентов оценки (сумма = 1.0)
        self.weights = {
            'trend_alignment': 0.25,      # Согласованность трендов
            'indicator_strength': 0.20,   # Сила индикаторов
            'market_condition': 0.15,     # Рыночные условия
            'risk_reward': 0.15,          # Соотношение риск/прибыль
            'ml_confidence': 0.10,        # ML уверенность
            'volume_quality': 0.10,       # Качество объемов
            'timing': 0.05               # Качество тайминга
        }
        
        # Пороги качества
        self.quality_thresholds = {
            'EXCELLENT': 16.0,  # 16-20 баллов
            'GOOD': 12.0,       # 12-16 баллов  
            'AVERAGE': 8.0,     # 8-12 баллов
            'POOR': 0.0         # 0-8 баллов
        }
        
        logger.info("🏆 StrategyEvaluator инициализирован")
    
    def evaluate_strategy(self, signal_data: Dict, market_data: Dict, 
                         market_condition: str) -> StrategyScore:
        """
        Оценить качество торговой стратегии
        
        Args:
            signal_data: Данные сигнала (direction, confidence, reasons)
            market_data: Рыночные данные (RSI, MACD, BB, etc.)
            market_condition: Состояние рынка
            
        Returns:
            StrategyScore с оценкой 0-20
        """
        try:
            components = {}
            
            # 1. Согласованность трендов (25%)
            components['trend_alignment'] = self._evaluate_trend_alignment(market_data)
            
            # 2. Сила индикаторов (20%)
            components['indicator_strength'] = self._evaluate_indicator_strength(market_data)
            
            # 3. Рыночные условия (15%)
            components['market_condition'] = self._evaluate_market_condition(
                market_condition, signal_data.get('direction', 'buy')
            )
            
            # 4. Соотношение риск/прибыль (15%)
            components['risk_reward'] = self._evaluate_risk_reward(signal_data)
            
            # 5. ML уверенность (10%)
            components['ml_confidence'] = self._evaluate_ml_confidence(signal_data)
            
            # 6. Качество объемов (10%)
            components['volume_quality'] = self._evaluate_volume_quality(market_data)
            
            # 7. Качество тайминга (5%)
            components['timing'] = self._evaluate_timing(market_data, market_condition)
            
            # Рассчитываем итоговую оценку
            total_score = sum(
                components[component] * self.weights[component] * 20
                for component in components
            )
            
            # Определяем уровень качества
            quality_level = self._determine_quality_level(total_score)
            
            # Генерируем рекомендацию
            recommendation = self._generate_recommendation(total_score, components)
            
            return StrategyScore(
                total_score=round(total_score, 2),
                components=components,
                quality_level=quality_level,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка оценки стратегии: {e}")
            return self._get_default_score()
    
    def _evaluate_trend_alignment(self, market_data: Dict) -> float:
        """Оценка согласованности трендов (0-1)"""
        try:
            score = 0.0
            
            # Проверяем EMA тренды на разных таймфреймах
            trends = []
            
            # Анализируем EMA 9 vs 21
            ema_9 = market_data.get('ema_9', 0)
            ema_21 = market_data.get('ema_21', 0)
            if ema_9 > ema_21:
                trends.append('up')
            elif ema_9 < ema_21:
                trends.append('down')
            
            # Анализируем EMA 21 vs 50
            ema_50 = market_data.get('ema_50', ema_21)
            if ema_21 > ema_50:
                trends.append('up')
            elif ema_21 < ema_50:
                trends.append('down')
            
            # Анализируем цену vs EMA
            price = market_data.get('price', 0)
            if price > ema_9:
                trends.append('up')
            elif price < ema_9:
                trends.append('down')
            
            # Рассчитываем согласованность
            if len(trends) > 0:
                up_count = trends.count('up')
                down_count = trends.count('down')
                total_count = len(trends)
                
                # Максимальная согласованность = все тренды в одном направлении
                max_alignment = max(up_count, down_count)
                score = max_alignment / total_count
            
            return score
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка оценки трендов: {e}")
            return 0.5
    
    def _evaluate_indicator_strength(self, market_data: Dict) -> float:
        """Оценка силы индикаторов (0-1)"""
        try:
            scores = []
            
            # RSI сила
            rsi = market_data.get('rsi', 50)
            if rsi <= 30 or rsi >= 70:  # Экстремальные зоны
                scores.append(0.9)
            elif rsi <= 35 or rsi >= 65:  # Сильные зоны
                scores.append(0.7)
            elif rsi <= 40 or rsi >= 60:  # Умеренные зоны
                scores.append(0.5)
            else:  # Нейтральная зона
                scores.append(0.3)
            
            # MACD сила
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            macd_histogram = market_data.get('macd_histogram', 0)
            
            if abs(macd - macd_signal) > abs(macd_signal) * 0.1:  # Сильное расхождение
                scores.append(0.8)
            elif macd_histogram != 0:  # Есть гистограмма
                scores.append(0.6)
            else:
                scores.append(0.4)
            
            # Bollinger Bands сила
            bb_position = market_data.get('bb_position', 50)
            if bb_position <= 20 or bb_position >= 80:  # Экстремальные позиции
                scores.append(0.8)
            elif bb_position <= 30 or bb_position >= 70:  # Сильные позиции
                scores.append(0.6)
            else:  # Нормальные позиции
                scores.append(0.4)
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка оценки индикаторов: {e}")
            return 0.5
    
    def _evaluate_market_condition(self, market_condition: str, direction: str) -> float:
        """Оценка соответствия рыночным условиям (0-1)"""
        try:
            # Соответствие направления сигнала рыночным условиям
            if market_condition == 'BULLISH' and direction == 'buy':
                return 0.9  # Отлично
            elif market_condition == 'BEARISH' and direction == 'sell':
                return 0.9  # Отлично
            elif market_condition == 'NEUTRAL':
                return 0.6  # Средне (любое направление)
            elif market_condition == 'VOLATILE':
                return 0.4  # Сложные условия
            else:
                return 0.3  # Против тренда
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка оценки рыночных условий: {e}")
            return 0.5
    
    def _evaluate_risk_reward(self, signal_data: Dict) -> float:
        """Оценка соотношения риск/прибыль (0-1)"""
        try:
            # Наши стандартные параметры
            stop_loss_percent = 20  # -20%
            min_tp_percent = 4      # +4% минимум
            
            # Рассчитываем R/R для первого TP
            risk_reward_ratio = min_tp_percent / stop_loss_percent  # 0.2
            
            # Оценка R/R
            if risk_reward_ratio >= 0.5:  # 1:2 или лучше
                return 0.9
            elif risk_reward_ratio >= 0.3:  # 1:3
                return 0.7
            elif risk_reward_ratio >= 0.2:  # 1:5 (наш случай)
                return 0.6
            elif risk_reward_ratio >= 0.1:  # 1:10
                return 0.4
            else:
                return 0.2
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка оценки R/R: {e}")
            return 0.6
    
    def _evaluate_ml_confidence(self, signal_data: Dict) -> float:
        """Оценка ML уверенности (0-1)"""
        try:
            confidence = signal_data.get('confidence', 50)
            
            # Нормализуем confidence (0-100) в score (0-1)
            if confidence >= 80:
                return 0.9
            elif confidence >= 70:
                return 0.8
            elif confidence >= 60:
                return 0.7
            elif confidence >= 50:
                return 0.6
            elif confidence >= 40:
                return 0.4
            else:
                return 0.2
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка оценки ML confidence: {e}")
            return 0.5
    
    def _evaluate_volume_quality(self, market_data: Dict) -> float:
        """Оценка качества объемов (0-1)"""
        try:
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            # Оценка объемов
            if volume_ratio >= 2.0:  # Очень высокий объем
                return 0.9
            elif volume_ratio >= 1.5:  # Высокий объем
                return 0.8
            elif volume_ratio >= 1.0:  # Нормальный объем
                return 0.7
            elif volume_ratio >= 0.8:  # Приемлемый объем
                return 0.6
            elif volume_ratio >= 0.5:  # Низкий объем
                return 0.4
            else:  # Очень низкий объем
                return 0.2
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка оценки объемов: {e}")
            return 0.5
    
    def _evaluate_timing(self, market_data: Dict, market_condition: str) -> float:
        """Оценка качества тайминга (0-1)"""
        try:
            score = 0.5  # Базовая оценка
            
            # Бонус за хорошие рыночные условия
            if market_condition in ['BULLISH', 'BEARISH']:
                score += 0.2
            
            # Бонус за экстремальные RSI
            rsi = market_data.get('rsi', 50)
            if rsi <= 30 or rsi >= 70:
                score += 0.2
            
            # Бонус за позицию в Bollinger Bands
            bb_position = market_data.get('bb_position', 50)
            if bb_position <= 25 or bb_position >= 75:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка оценки тайминга: {e}")
            return 0.5
    
    def _determine_quality_level(self, total_score: float) -> str:
        """Определить уровень качества"""
        if total_score >= self.quality_thresholds['EXCELLENT']:
            return 'EXCELLENT'
        elif total_score >= self.quality_thresholds['GOOD']:
            return 'GOOD'
        elif total_score >= self.quality_thresholds['AVERAGE']:
            return 'AVERAGE'
        else:
            return 'POOR'
    
    def _generate_recommendation(self, total_score: float, components: Dict) -> str:
        """Генерация рекомендации"""
        try:
            if total_score >= 16:
                return "🟢 ОТЛИЧНЫЙ сигнал - рекомендуется к исполнению"
            elif total_score >= 12:
                return "🟡 ХОРОШИЙ сигнал - можно исполнять с осторожностью"
            elif total_score >= 8:
                return "🟠 СРЕДНИЙ сигнал - требует дополнительного анализа"
            else:
                return "🔴 СЛАБЫЙ сигнал - не рекомендуется к исполнению"
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка генерации рекомендации: {e}")
            return "⚠️ Требуется дополнительный анализ"
    
    def _get_default_score(self) -> StrategyScore:
        """Получить дефолтную оценку при ошибке"""
        return StrategyScore(
            total_score=10.0,
            components={},
            quality_level='AVERAGE',
            recommendation='⚠️ Ошибка оценки - требуется ручная проверка'
        )
    
    def get_score_breakdown(self, score: StrategyScore) -> str:
        """Получить детальную разбивку оценки"""
        try:
            breakdown = [f"🏆 Оценка стратегии: {score.total_score:.1f}/20 ({score.quality_level})"]
            
            for component, value in score.components.items():
                component_score = value * self.weights[component] * 20
                breakdown.append(f"  • {component}: {component_score:.1f}")
            
            breakdown.append(f"📋 {score.recommendation}")
            
            return "\n".join(breakdown)
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка создания breakdown: {e}")
            return f"🏆 Оценка: {score.total_score:.1f}/20"

# Тестирование модуля
if __name__ == "__main__":
    evaluator = StrategyEvaluator()
    
    # Тестовые данные
    signal_data = {
        'direction': 'buy',
        'confidence': 75,
        'reasons': ['EMA_TREND', 'RSI_OVERSOLD']
    }
    
    market_data = {
        'rsi': 35,
        'bb_position': 25,
        'macd': 0.5,
        'macd_signal': 0.3,
        'macd_histogram': 0.2,
        'ema_9': 101,
        'ema_21': 100,
        'ema_50': 99,
        'price': 102,
        'volume_ratio': 1.2
    }
    
    score = evaluator.evaluate_strategy(signal_data, market_data, 'BULLISH')
    
    print("🏆 Тест StrategyEvaluator:")
    print(evaluator.get_score_breakdown(score))

