#!/usr/bin/env python3
"""
🔄 ADAPTIVE TRADING SYSTEM
===========================

Полностью автономная система с адаптацией всех параметров
Фиксированные: Позиция ($25), SL (-20%), TP (min +$1)
Адаптивные: Все индикаторы, фильтры, пороги, веса

"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThreshold:
    """Адаптивный порог"""
    parameter: str
    current_value: float
    base_value: float
    min_value: float
    max_value: float
    success_rate: float = 0.5


class DynamicThresholds:
    """Динамические пороговые значения"""
    
    def __init__(self):
        self.thresholds = {
            'rsi_min': AdaptiveThreshold('rsi_min', 25, 25, 20, 35),
            'rsi_max': AdaptiveThreshold('rsi_max', 75, 75, 70, 85),
            'volume_ratio': AdaptiveThreshold('volume_ratio', 0.3, 0.3, 0.2, 0.5),
            'momentum': AdaptiveThreshold('momentum', 0.1, 0.1, 0.05, 0.3),
            'min_confidence': AdaptiveThreshold('min_confidence', 55, 55, 50, 70),
        }
    
    def adapt_to_market(self, market_condition: str, success_history: List[Dict]):
        """
        Адаптирует пороги на основе рыночных условий и успешности
        
        Args:
            market_condition: BULLISH, BEARISH, NEUTRAL, VOLATILE
            success_history: История сделок
        """
        if not success_history:
            return
        
        # Рассчитываем успешность
        total_trades = len(success_history)
        successful_trades = sum(1 for t in success_history if t.get('pnl', 0) > 0)
        success_rate = successful_trades / total_trades if total_trades > 0 else 0.5
        
        # Адаптация на основе успешности
        if success_rate < 0.4:
            # Если успешность низкая → смягчаем фильтры
            logger.info(f"⚠️ Низкая успешность ({success_rate:.1%}) → смягчаем фильтры")
            self._adjust_thresholds(0.85)
            
        elif success_rate > 0.7:
            # Если успешность высокая → немного ужесточаем
            logger.info(f"✅ Высокая успешность ({success_rate:.1%}) → немного ужесточаем")
            self._adjust_thresholds(1.05)
        
        # Адаптация на основе рыночных условий
        self._adapt_to_market_condition(market_condition)
    
    def _adjust_thresholds(self, factor: float):
        """Корректирует пороги на коэффициент"""
        for threshold in self.thresholds.values():
            if threshold.parameter in ['rsi_min', 'volume_ratio', 'momentum']:
                threshold.current_value *= factor
                threshold.current_value = max(threshold.min_value, min(threshold.max_value, threshold.current_value))
            elif threshold.parameter == 'rsi_max':
                threshold.current_value /= factor
                threshold.current_value = max(threshold.min_value, min(threshold.max_value, threshold.current_value))
    
    def _adapt_to_market_condition(self, market_condition: str):
        """Адаптирует пороги к условиям рынка"""
        if market_condition == 'BULLISH':
            # На бычьем рынке снижаем RSI мин, увеличиваем volume
            self.thresholds['rsi_min'].current_value = max(20, self.thresholds['rsi_min'].current_value - 5)
            self.thresholds['volume_ratio'].current_value = min(0.5, self.thresholds['volume_ratio'].current_value + 0.1)
        
        elif market_condition == 'BEARISH':
            # На медвежьем рынке повышаем RSI мин, снижаем требования
            self.thresholds['rsi_min'].current_value = max(30, self.thresholds['rsi_min'].current_value + 5)
            self.thresholds['volume_ratio'].current_value = max(0.2, self.thresholds['volume_ratio'].current_value - 0.1)
    
    def get_threshold(self, parameter: str) -> float:
        """Возвращает текущее значение порога"""
        return self.thresholds.get(parameter, self.thresholds['min_confidence']).current_value
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Возвращает все пороги"""
        return {name: th.current_value for name, th in self.thresholds.items()}


class IndicatorSelector:
    """Самообучающаяся система выбора индикаторов"""
    
    def __init__(self):
        self.indicators = {
            'RSI': {'weight': 0.2, 'enabled': True, 'success_count': 0, 'total_count': 0},
            'Volume': {'weight': 0.3, 'enabled': True, 'success_count': 0, 'total_count': 0},
            'Momentum': {'weight': 0.2, 'enabled': True, 'success_count': 0, 'total_count': 0},
            'MACD': {'weight': 0.1, 'enabled': True, 'success_count': 0, 'total_count': 0},
            'Bollinger': {'weight': 0.2, 'enabled': True, 'success_count': 0, 'total_count': 0},
        }
    
    def adapt_indicators(self, trade_results: List[Dict]):
        """
        Адаптирует веса индикаторов на основе успешности
        
        Args:
            trade_results: Список сделок с результатами
        """
        if not trade_results:
            return
        
        # Рассчитываем успешность для каждого индикатора
        for indicator_name, config in self.indicators.items():
            success_rate = config['success_count'] / config['total_count'] if config['total_count'] > 0 else 0.5
            
            # Адаптация веса
            if success_rate < 0.3 and config['weight'] > 0.05:
                # Низкая успешность → уменьшаем вес
                config['weight'] *= 0.8
                logger.debug(f"📉 {indicator_name}: вес снижен до {config['weight']:.2f}")
                
                if config['weight'] < 0.05:
                    config['enabled'] = False
                    logger.warning(f"⚠️ {indicator_name} отключен из-за низкой эффективности")
                    
            elif success_rate > 0.7 and config['weight'] < 0.5:
                # Высокая успешность → увеличиваем вес
                config['weight'] *= 1.2
                if config['weight'] > 0.5:
                    config['weight'] = 0.5  # Максимум
                logger.debug(f"📈 {indicator_name}: вес увеличен до {config['weight']:.2f}")
    
    def record_indicator_usage(self, indicator_name: str, was_successful: bool):
        """Записывает использование индикатора"""
        if indicator_name in self.indicators:
            config = self.indicators[indicator_name]
            config['total_count'] += 1
            if was_successful:
                config['success_count'] += 1
    
    def get_indicator_weight(self, indicator_name: str) -> float:
        """Возвращает вес индикатора"""
        config = self.indicators.get(indicator_name, {'weight': 0, 'enabled': False})
        return config['weight'] if config['enabled'] else 0
    
    def get_active_indicators(self) -> List[str]:
        """Возвращает список активных индикаторов"""
        return [name for name, config in self.indicators.items() if config['enabled']]


class SmartDecisionMaker:
    """Умная система принятия решений"""
    
    def __init__(self):
        self.ml_weight = 0.3  # Вес ML предсказаний
        self.rule_weight = 0.7  # Вес правил (индикаторов)
        
        self.ml_success_history = []
        self.rule_success_history = []
    
    def make_decision(self, indicator_score: float, ml_prediction: Optional[Dict]) -> bool:
        """
        Принимает решение на основе адаптивных правил
        
        Args:
            indicator_score: Оценка от индикаторов (0-1)
            ml_prediction: Предсказание ML (optional)
            
        Returns:
            True если открывать позицию
        """
        
        # ML предсказание
        if ml_prediction:
            ml_score = ml_prediction.get('confidence', 50) / 100
        else:
            ml_score = 0.5  # Нейтральное
        
        # Комбинируем с весами
        final_score = (
            indicator_score * self.rule_weight +
            ml_score * self.ml_weight
        )
        
        # MIN_CONFIDENCE
        min_confidence = 0.55
        decision = final_score >= min_confidence
        
        logger.debug(f"🧠 Decision: indicator={indicator_score:.2f} ml={ml_score:.2f} final={final_score:.2f} → {decision}")
        
        return decision
    
    def adapt_weights(self, trade_results: List[Dict]):
        """
        Адаптирует веса на основе успешности
        
        Args:
            trade_results: Список сделок с результатами
        """
        if not trade_results or len(trade_results) < 5:
            return
        
        # Рассчитываем успешность ML и правил отдельно
        ml_success_rate = self._calculate_ml_success(trade_results)
        rule_success_rate = self._calculate_rule_success(trade_results)
        
        # Адаптация весов
        if ml_success_rate > rule_success_rate + 0.1:
            # ML успешнее → увеличиваем его вес
            self.ml_weight = min(0.5, self.ml_weight + 0.05)
            self.rule_weight = max(0.5, self.rule_weight - 0.05)
            logger.info(f"🤖 ML более успешный ({ml_success_rate:.1%} vs {rule_success_rate:.1%}) → увеличиваем вес ML")
            
        elif rule_success_rate > ml_success_rate + 0.1:
            # Правила успешнее → увеличиваем их вес
            self.rule_weight = min(0.8, self.rule_weight + 0.05)
            self.ml_weight = max(0.2, self.ml_weight - 0.05)
            logger.info(f"📊 Правила более успешные ({rule_success_rate:.1%} vs {ml_success_rate:.1%}) → увеличиваем вес правил")
    
    def _calculate_ml_success(self, trade_results: List[Dict]) -> float:
        """Рассчитывает успешность ML"""
        ml_trades = [t for t in trade_results if t.get('used_ml', False)]
        if not ml_trades:
            return 0.5
        successful = sum(1 for t in ml_trades if t.get('pnl', 0) > 0)
        return successful / len(ml_trades)
    
    def _calculate_rule_success(self, trade_results: List[Dict]) -> float:
        """Рассчитывает успешность правил"""
        successful = sum(1 for t in trade_results if t.get('pnl', 0) > 0)
        return successful / len(trade_results) if trade_results else 0.5


class FullyAdaptiveSystem:
    """Полностью адаптивная система"""
    
    def __init__(self):
        self.dynamic_thresholds = DynamicThresholds()
        self.indicator_selector = IndicatorSelector()
        self.smart_decision = SmartDecisionMaker()
        
        # Фиксированные параметры
        self.position_size = 25  # $25
        self.leverage = 5  # 5x
        self.stop_loss = -0.20  # -20%
        self.min_tp_percent = 0.04  # +4% = +$1
        self.max_positions = 3
        
        # История для адаптации
        self.trade_history = []
        
        logger.info("🔄 FullyAdaptiveSystem инициализирована")
    
    def _adapt_thresholds(self, market_data: Dict, recent_trades: List[Dict]) -> Dict:
        """🎯 Адаптация порогов на основе производительности"""
        try:
            # Базовые пороги
            base_thresholds = {
                'min_confidence': 45,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'bb_threshold': 25,
                'volume_threshold': 0.3
            }
            
            if not recent_trades:
                return base_thresholds
            
            # Анализируем последние сделки
            wins = [t for t in recent_trades if t.get('result') == 'win']
            win_rate = len(wins) / len(recent_trades) if recent_trades else 0.5
            
            # Адаптируем пороги
            adapted_thresholds = base_thresholds.copy()
            
            if win_rate > 0.7:  # Высокая успешность - ужесточаем
                adapted_thresholds['min_confidence'] += 10
                adapted_thresholds['rsi_oversold'] -= 5
                adapted_thresholds['rsi_overbought'] += 5
            elif win_rate < 0.4:  # Низкая успешность - смягчаем
                adapted_thresholds['min_confidence'] -= 10
                adapted_thresholds['rsi_oversold'] += 5
                adapted_thresholds['rsi_overbought'] -= 5
            
            # Ограничиваем значения
            adapted_thresholds['min_confidence'] = max(30, min(70, adapted_thresholds['min_confidence']))
            adapted_thresholds['rsi_oversold'] = max(20, min(45, adapted_thresholds['rsi_oversold']))
            adapted_thresholds['rsi_overbought'] = max(55, min(80, adapted_thresholds['rsi_overbought']))
            
            return adapted_thresholds
            
        except Exception as e:
            logger.error(f"❌ Ошибка адаптации порогов: {e}")
            return base_thresholds
    
    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Возвращает адаптивные пороги"""
        return self.dynamic_thresholds.get_all_thresholds()
    
    def should_open_position(self, signal_data: Dict, ml_prediction: Optional[Dict] = None) -> tuple[bool, float]:
        """
        Проверяет, нужно ли открывать позицию
        
        Returns:
            (should_open, confidence)
        """
        
        # Рассчитываем оценку от индикаторов
        indicator_score = self._calculate_indicator_score(signal_data)
        
        # Принимаем решение
        should_open = self.smart_decision.make_decision(indicator_score, ml_prediction)
        
        # Рассчитываем финальную уверенность
        final_confidence = indicator_score * self.smart_decision.rule_weight
        if ml_prediction:
            final_confidence += (ml_prediction.get('confidence', 50) / 100) * self.smart_decision.ml_weight
        
        return should_open, final_confidence * 100
    
    def _calculate_indicator_score(self, signal_data: Dict) -> float:
        """Рассчитывает оценку от индикаторов"""
        total_score = 0
        total_weight = 0
        
        for indicator_name, config in self.indicator_selector.indicators.items():
            if not config['enabled']:
                continue
            
            # Проверяем индикатор
            weight = config['weight']
            if self._check_indicator(indicator_name, signal_data):
                total_score += weight
            
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _check_indicator(self, indicator_name: str, signal_data: Dict) -> bool:
        """Проверяет индикатор"""
        thresholds = self.get_adaptive_thresholds()
        
        if indicator_name == 'RSI':
            rsi = signal_data.get('rsi', 50)
            return thresholds['rsi_min'] <= rsi <= thresholds['rsi_max']
        
        elif indicator_name == 'Volume':
            vol = signal_data.get('volume_ratio', 0.5)
            return vol > thresholds['volume_ratio']
        
        elif indicator_name == 'Momentum':
            mom = signal_data.get('momentum', 0)
            return mom > thresholds['momentum']
        
        elif indicator_name == 'MACD':
            return signal_data.get('macd_bullish', False)
        
        elif indicator_name == 'Bollinger':
            bb_pos = signal_data.get('bb_position', 50)
            return 0 < bb_pos < 100  # Не на границах
        
        return False
    
    def adapt_to_results(self, trade_results: List[Dict], market_condition: str):
        """
        Адаптирует систему на основе результатов
        
        Args:
            trade_results: Список сделок с pnl
            market_condition: BULLISH, BEARISH, NEUTRAL, VOLATILE
        """
        self.trade_history.extend(trade_results)
        
        # Ограничиваем историю последними 50 сделками
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]
        
        # Адаптация порогов
        self.dynamic_thresholds.adapt_to_market(market_condition, self.trade_history)
        
        # Адаптация индикаторов
        self.indicator_selector.adapt_indicators(self.trade_history)
        
        # Адаптация весов решений
        self.smart_decision.adapt_weights(self.trade_history)
        
        logger.info(f"🔄 Адаптация завершена на основе {len(trade_results)} новых сделок")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Возвращает статистику системы"""
        return {
            'thresholds': self.get_adaptive_thresholds(),
            'indicator_weights': {name: config['weight'] for name, config in self.indicator_selector.indicators.items()},
            'decision_weights': {
                'ml': self.smart_decision.ml_weight,
                'rules': self.smart_decision.rule_weight
            },
            'total_trades': len(self.trade_history),
            'success_rate': sum(1 for t in self.trade_history if t.get('pnl', 0) > 0) / len(self.trade_history) if self.trade_history else 0
        }



