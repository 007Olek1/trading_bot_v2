#!/usr/bin/env python3
"""
🎯 СИСТЕМА АДАПТИВНЫХ ПАРАМЕТРОВ
================================

Функции:
- Автоматическая настройка порогов уверенности
- Динамическое изменение фильтров в зависимости от волатильности
- Адаптация к рыночным условиям (бычий/медвежий рынок)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketCondition:
    """📊 Состояние рынка"""
    trend: str  # 'bullish', 'bearish', 'sideways'
    volatility: float  # 0-1, где 1 = максимальная волатильность
    volume_trend: str  # 'high', 'normal', 'low'
    market_cap_dominance: float  # Доля BTC в общем рынке
    fear_greed_index: float  # 0-100, где 0 = страх, 100 = жадность

@dataclass
class AdaptiveParameters:
    """🎯 Адаптивные параметры"""
    min_confidence: float
    volume_filter: float
    rsi_oversold: float
    rsi_overbought: float
    bb_upper_threshold: float
    bb_lower_threshold: float
    macd_threshold: float
    stop_loss_percent: float
    take_profit_percent: float

class AdaptiveParameterSystem:
    """🎯 Система адаптивных параметров"""
    
    def __init__(self):
        self.market_history = []
        self.parameter_history = []
        self.current_market_condition = None
        
        # Базовые параметры (консервативные)
        self.base_parameters = AdaptiveParameters(
            min_confidence=85.0,
            volume_filter=1.5,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            bb_upper_threshold=0.8,
            bb_lower_threshold=0.2,
            macd_threshold=0.001,
            stop_loss_percent=2.0,
            take_profit_percent=3.0
        )
        
        # Агрессивные параметры (для бычьего рынка)
        self.aggressive_parameters = AdaptiveParameters(
            min_confidence=75.0,
            volume_filter=1.2,
            rsi_oversold=35.0,
            rsi_overbought=75.0,
            bb_upper_threshold=0.85,
            bb_lower_threshold=0.15,
            macd_threshold=0.0005,
            stop_loss_percent=1.5,
            take_profit_percent=4.0
        )
        
        # Консервативные параметры (для медвежьего рынка)
        self.conservative_parameters = AdaptiveParameters(
            min_confidence=90.0,
            volume_filter=2.0,
            rsi_oversold=25.0,
            rsi_overbought=75.0,
            bb_upper_threshold=0.9,
            bb_lower_threshold=0.1,
            macd_threshold=0.002,
            stop_loss_percent=1.0,
            take_profit_percent=2.0
        )
    
    def analyze_market_condition(self, market_data: Dict) -> MarketCondition:
        """📊 Анализ текущего состояния рынка"""
        
        # Анализ тренда
        trend = self._determine_trend(market_data)
        
        # Анализ волатильности
        volatility = self._calculate_volatility(market_data)
        
        # Анализ объёма
        volume_trend = self._analyze_volume_trend(market_data)
        
        # Доминирование BTC
        btc_dominance = market_data.get('btc_dominance', 0.4)
        
        # Индекс страха и жадности (упрощённый)
        fear_greed = self._calculate_fear_greed_index(market_data)
        
        condition = MarketCondition(
            trend=trend,
            volatility=volatility,
            volume_trend=volume_trend,
            market_cap_dominance=btc_dominance,
            fear_greed_index=fear_greed
        )
        
        self.market_history.append({
            'timestamp': datetime.now(),
            'condition': condition
        })
        
        # Ограничиваем историю последними 100 записями
        if len(self.market_history) > 100:
            self.market_history = self.market_history[-100:]
        
        self.current_market_condition = condition
        return condition
    
    def _determine_trend(self, market_data: Dict) -> str:
        """📈 Определение тренда рынка"""
        btc_change_24h = market_data.get('btc_change_24h', 0)
        eth_change_24h = market_data.get('eth_change_24h', 0)
        market_cap_change = market_data.get('market_cap_change', 0)
        
        # Среднее изменение основных активов
        avg_change = (btc_change_24h + eth_change_24h + market_cap_change) / 3
        
        if avg_change > 2:
            return 'bullish'
        elif avg_change < -2:
            return 'bearish'
        else:
            return 'sideways'
    
    def _calculate_volatility(self, market_data: Dict) -> float:
        """📊 Расчёт волатильности"""
        # Используем исторические данные для расчёта волатильности
        if len(self.market_history) < 10:
            return 0.5  # Средняя волатильность по умолчанию
        
        # Берём последние 10 записей
        recent_changes = []
        for i in range(1, min(11, len(self.market_history))):
            prev_condition = self.market_history[-i-1]['condition']
            curr_condition = self.market_history[-i]['condition']
            
            # Упрощённый расчёт изменения
            change = abs(curr_condition.fear_greed_index - prev_condition.fear_greed_index)
            recent_changes.append(change)
        
        volatility = np.std(recent_changes) / 100.0  # Нормализуем к 0-1
        return min(max(volatility, 0), 1)  # Ограничиваем диапазон
    
    def _analyze_volume_trend(self, market_data: Dict) -> str:
        """📊 Анализ тренда объёма"""
        total_volume_24h = market_data.get('total_volume_24h', 0)
        avg_volume_7d = market_data.get('avg_volume_7d', total_volume_24h)
        
        if avg_volume_7d == 0:
            return 'normal'
        
        volume_ratio = total_volume_24h / avg_volume_7d
        
        if volume_ratio > 1.5:
            return 'high'
        elif volume_ratio < 0.7:
            return 'low'
        else:
            return 'normal'
    
    def _calculate_fear_greed_index(self, market_data: Dict) -> float:
        """😨 Расчёт индекса страха и жадности"""
        # Упрощённый расчёт на основе доступных данных
        btc_change = market_data.get('btc_change_24h', 0)
        volatility = self._calculate_volatility(market_data)
        volume_trend = self._analyze_volume_trend(market_data)
        
        # Базовый индекс на основе изменения BTC
        base_index = 50 + (btc_change * 2)  # -50% до +50% изменения
        
        # Корректировка на волатильность
        if volatility > 0.7:
            base_index -= 20  # Высокая волатильность = страх
        elif volatility < 0.3:
            base_index += 10  # Низкая волатильность = спокойствие
        
        # Корректировка на объём
        if volume_trend == 'high':
            base_index += 15  # Высокий объём = интерес
        elif volume_trend == 'low':
            base_index -= 10  # Низкий объём = осторожность
        
        return min(max(base_index, 0), 100)  # Ограничиваем 0-100
    
    def get_adaptive_parameters(self, market_data: Dict) -> AdaptiveParameters:
        """🎯 Получение адаптивных параметров"""
        condition = self.analyze_market_condition(market_data)
        
        # Выбираем базовые параметры
        if condition.trend == 'bullish' and condition.fear_greed_index > 60:
            # Бычий рынок - используем агрессивные параметры
            base_params = self.aggressive_parameters
            logger.info("🐂 Бычий рынок: используем агрессивные параметры")
            
        elif condition.trend == 'bearish' or condition.fear_greed_index < 30:
            # Медвежий рынок - используем консервативные параметры
            base_params = self.conservative_parameters
            logger.info("🐻 Медвежий рынок: используем консервативные параметры")
            
        else:
            # Боковой рынок - используем базовые параметры
            base_params = self.base_parameters
            logger.info("↔️ Боковой рынок: используем базовые параметры")
        
        # Адаптируем параметры на основе волатильности
        adapted_params = self._adapt_to_volatility(base_params, condition.volatility)
        
        # Адаптируем параметры на основе объёма
        adapted_params = self._adapt_to_volume(adapted_params, condition.volume_trend)
        
        # Сохраняем историю параметров
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'condition': condition,
            'parameters': adapted_params
        })
        
        if len(self.parameter_history) > 50:
            self.parameter_history = self.parameter_history[-50:]
        
        return adapted_params
    
    def _adapt_to_volatility(self, params: AdaptiveParameters, volatility: float) -> AdaptiveParameters:
        """📊 Адаптация к волатильности"""
        # Высокая волатильность = более строгие фильтры
        volatility_factor = 1 + (volatility - 0.5) * 0.5  # 0.75 - 1.25
        
        return AdaptiveParameters(
            min_confidence=params.min_confidence * volatility_factor,
            volume_filter=params.volume_filter * volatility_factor,
            rsi_oversold=params.rsi_oversold * volatility_factor,
            rsi_overbought=params.rsi_overbought * volatility_factor,
            bb_upper_threshold=params.bb_upper_threshold * volatility_factor,
            bb_lower_threshold=params.bb_lower_threshold * volatility_factor,
            macd_threshold=params.macd_threshold * volatility_factor,
            stop_loss_percent=params.stop_loss_percent * volatility_factor,
            take_profit_percent=params.take_profit_percent * volatility_factor
        )
    
    def _adapt_to_volume(self, params: AdaptiveParameters, volume_trend: str) -> AdaptiveParameters:
        """📊 Адаптация к объёму"""
        if volume_trend == 'high':
            # Высокий объём = можем быть менее строгими
            volume_factor = 0.9
        elif volume_trend == 'low':
            # Низкий объём = нужно быть более строгими
            volume_factor = 1.1
        else:
            volume_factor = 1.0
        
        return AdaptiveParameters(
            min_confidence=params.min_confidence * volume_factor,
            volume_filter=params.volume_filter * volume_factor,
            rsi_oversold=params.rsi_oversold,
            rsi_overbought=params.rsi_overbought,
            bb_upper_threshold=params.bb_upper_threshold,
            bb_lower_threshold=params.bb_lower_threshold,
            macd_threshold=params.macd_threshold,
            stop_loss_percent=params.stop_loss_percent,
            take_profit_percent=params.take_profit_percent
        )
    
    def get_parameter_recommendations(self) -> Dict:
        """💡 Рекомендации по параметрам"""
        if not self.current_market_condition:
            return {"message": "Недостаточно данных для анализа"}
        
        condition = self.current_market_condition
        recommendations = {
            "market_condition": {
                "trend": condition.trend,
                "volatility": f"{condition.volatility:.2f}",
                "volume_trend": condition.volume_trend,
                "fear_greed_index": f"{condition.fear_greed_index:.1f}"
            },
            "recommendations": []
        }
        
        # Рекомендации на основе состояния рынка
        if condition.trend == 'bullish':
            recommendations["recommendations"].append("🐂 Бычий рынок: можно увеличить агрессивность")
        
        if condition.volatility > 0.7:
            recommendations["recommendations"].append("📊 Высокая волатильность: увеличить фильтры")
        
        if condition.volume_trend == 'low':
            recommendations["recommendations"].append("📉 Низкий объём: быть более осторожным")
        
        if condition.fear_greed_index < 30:
            recommendations["recommendations"].append("😨 Индекс страха: консервативный подход")
        
        return recommendations
    
    def save_parameters_to_file(self, filename: str = "adaptive_parameters.json"):
        """💾 Сохранение параметров в файл"""
        if not self.current_market_condition:
            return
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "market_condition": {
                "trend": self.current_market_condition.trend,
                "volatility": self.current_market_condition.volatility,
                "volume_trend": self.current_market_condition.volume_trend,
                "fear_greed_index": self.current_market_condition.fear_greed_index
            },
            "parameters": {
                "min_confidence": self.base_parameters.min_confidence,
                "volume_filter": self.base_parameters.volume_filter,
                "rsi_oversold": self.base_parameters.rsi_oversold,
                "rsi_overbought": self.base_parameters.rsi_overbought,
                "bb_upper_threshold": self.base_parameters.bb_upper_threshold,
                "bb_lower_threshold": self.base_parameters.bb_lower_threshold,
                "macd_threshold": self.base_parameters.macd_threshold,
                "stop_loss_percent": self.base_parameters.stop_loss_percent,
                "take_profit_percent": self.base_parameters.take_profit_percent
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Параметры сохранены в {filename}")

# Пример использования
if __name__ == "__main__":
    adaptive_system = AdaptiveParameterSystem()
    
    # Тестовые данные рынка
    test_market_data = {
        'btc_change_24h': 3.5,
        'eth_change_24h': 2.8,
        'market_cap_change': 2.1,
        'total_volume_24h': 50000000000,
        'avg_volume_7d': 45000000000,
        'btc_dominance': 0.42
    }
    
    # Получаем адаптивные параметры
    params = adaptive_system.get_adaptive_parameters(test_market_data)
    recommendations = adaptive_system.get_parameter_recommendations()
    
    print("🎯 АДАПТИВНЫЕ ПАРАМЕТРЫ:")
    print(f"Минимальная уверенность: {params.min_confidence:.1f}%")
    print(f"Фильтр объёма: {params.volume_filter:.1f}x")
    print(f"RSI перепроданность: {params.rsi_oversold:.1f}")
    print(f"RSI перекупленность: {params.rsi_overbought:.1f}")
    print(f"Stop Loss: {params.stop_loss_percent:.1f}%")
    print(f"Take Profit: {params.take_profit_percent:.1f}%")
    
    print("\n💡 РЕКОМЕНДАЦИИ:")
    for rec in recommendations["recommendations"]:
        print(f"• {rec}")






