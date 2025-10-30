#!/usr/bin/env python3
"""
✅ REALISM VALIDATOR V4.0
=========================

Проверка реалистичности торговых сигналов
Защита от нереалистичных ожиданий и ложных сигналов
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RealismCheck:
    """Результат проверки реалистичности"""
    is_realistic: bool
    confidence_score: float  # 0-100
    warnings: List[str]
    recommendations: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME

class RealismValidator:
    """✅ Валидатор реалистичности торговых сигналов"""
    
    def __init__(self):
        # Реалистичные пороги
        self.realistic_limits = {
            'max_tp_percent': 20,        # Максимальный TP +20%
            'max_sl_percent': 25,        # Максимальный SL -25%
            'min_probability': 20,       # Минимальная вероятность 20%
            'max_probability': 95,       # Максимальная вероятность 95%
            'max_leverage': 10,          # Максимальное плечо 10x
            'min_volume_ratio': 0.1,     # Минимальный объем 0.1x
            'max_volatility': 15,        # Максимальная волатильность 15%
            'min_confidence': 30,        # Минимальная уверенность 30%
        }
        
        # Специфичные ограничения для крупных активов (абсолютные движения)
        self.major_assets_limits = {
            'BTCUSDT': {'max_tp_percent': 10, 'max_abs_move_usd': 12000},
            'ETHUSDT': {'max_tp_percent': 12, 'max_abs_move_usd': 3500},
            'BNBUSDT': {'max_tp_percent': 15, 'max_abs_move_usd': 1200},
        }
        
        # Пороги риска
        self.risk_thresholds = {
            'LOW': {'max_tp': 8, 'max_sl': 15, 'min_prob': 70},
            'MEDIUM': {'max_tp': 12, 'max_sl': 20, 'min_prob': 50},
            'HIGH': {'max_tp': 16, 'max_sl': 25, 'min_prob': 40},
            'EXTREME': {'max_tp': 20, 'max_sl': 30, 'min_prob': 20}
        }
        
        logger.info("✅ RealismValidator инициализирован")
    
    def validate_signal(self, signal_data: Dict, market_data: Dict, 
                       tp_probabilities: List[Any] = None) -> RealismCheck:
        """
        Проверить реалистичность торгового сигнала
        
        Args:
            signal_data: Данные сигнала
            market_data: Рыночные данные
            tp_probabilities: Вероятности TP уровней
            
        Returns:
            RealismCheck с результатами проверки
        """
        try:
            warnings = []
            recommendations = []
            confidence_score = 100.0
            
            # 1. Проверка TP/SL соотношений
            tp_sl_score = self._validate_tp_sl_ratios(signal_data, warnings, recommendations)
            confidence_score *= tp_sl_score
            
            # 2. Проверка вероятностей
            prob_score = self._validate_probabilities(tp_probabilities, warnings, recommendations)
            confidence_score *= prob_score
            
            # 3. Проверка рыночных условий
            market_score = self._validate_market_conditions(market_data, warnings, recommendations)
            confidence_score *= market_score
            
            # 3.1 Сопоставление TP уровней с текущей волатильностью (ATR)
            tp_vol_score = self._validate_tp_vs_volatility(signal_data, market_data, warnings, recommendations)
            confidence_score *= tp_vol_score

            # 4. Проверка объемов и волатильности
            volume_score = self._validate_volume_volatility(market_data, warnings, recommendations)
            confidence_score *= volume_score
            
            # 5. Проверка на манипуляции
            manipulation_score = self._check_manipulation_signs(market_data, warnings, recommendations)
            confidence_score *= manipulation_score
            
            # Определяем уровень риска
            risk_level = self._determine_risk_level(signal_data, market_data)
            
            # Определяем реалистичность (строже для крупных активов)
            symbol = str(signal_data.get('symbol', '')).upper()
            major_assets = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            
            # Для крупных активов более строгая проверка
            if symbol in major_assets:
                # Если есть предупреждения о превышении лимитов TP, сигнал нереалистичен
                tp_warnings = [w for w in warnings if 'НЕРЕАЛИСТИЧНЫЙ TP' in w or 'НЕРЕАЛИСТИЧНОЕ абсолютное движение' in w]
                is_realistic = confidence_score >= 70 and len(tp_warnings) == 0 and len(warnings) <= 1
            else:
                is_realistic = confidence_score >= 60 and len(warnings) <= 2
            
            return RealismCheck(
                is_realistic=is_realistic,
                confidence_score=round(confidence_score, 1),
                warnings=warnings,
                recommendations=recommendations,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации реалистичности: {e}")
            return self._get_default_check()

    def _validate_tp_vs_volatility(self, signal_data: Dict, market_data: Dict,
                                   warnings: List[str], recommendations: List[str]) -> float:
        """Проверка соответствия TP уровней текущей волатильности (ATR)."""
        try:
            price = float(market_data.get('price', 0) or 0)
            atr = float(market_data.get('atr', 0) or 0)
            if price <= 0 or atr <= 0:
                return 1.0

            atr_percent = (atr / price) * 100.0
            symbol = str(signal_data.get('symbol', '')).upper()
            tp_levels = signal_data.get('tp_levels', [])

            # Динамические лимиты по ATR
            if symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
                dynamic_limit = max(6.0, atr_percent * 2.5)
            else:
                dynamic_limit = max(12.0, atr_percent * 3.0)

            score = 1.0
            if tp_levels:
                max_tp = max([tp.get('percent', 0) for tp in tp_levels])
                if max_tp > dynamic_limit:
                    warnings.append(
                        f"⚠️ TP превышает текущую волатильность: +{max_tp}% > лимит {dynamic_limit:.1f}% (ATR={atr_percent:.2f}%)"
                    )
                    recommendations.append("Снизить TP уровни под текущую волатильность")
                    score *= 0.7

            return score
        except Exception:
            return 0.95
    
    def _validate_tp_sl_ratios(self, signal_data: Dict, warnings: List[str], 
                              recommendations: List[str]) -> float:
        """Проверка соотношений TP/SL (учет символа и абсолютных движений)"""
        try:
            score = 1.0
            symbol = str(signal_data.get('symbol', '')).upper()
            entry_price = float(signal_data.get('entry_price', 0) or 0)
            
            # Проверяем наличие SL
            stop_loss = signal_data.get('stop_loss_percent', 20)
            if stop_loss > self.realistic_limits['max_sl_percent']:
                warnings.append(f"⚠️ Слишком большой SL: -{stop_loss}%")
                recommendations.append("Уменьшить Stop Loss до -20%")
                score *= 0.8
            
            # Проверяем TP уровни
            tp_levels = signal_data.get('tp_levels', [])
            if tp_levels:
                max_tp = max([tp.get('percent', 0) for tp in tp_levels])
                if symbol in self.major_assets_limits:
                    limits = self.major_assets_limits[symbol]
                    if max_tp > limits['max_tp_percent']:
                        warnings.append(f"⚠️ НЕРЕАЛИСТИЧНЫЙ TP для {symbol}: +{max_tp}% > +{limits['max_tp_percent']}%")
                        recommendations.append(f"Снизить максимальный TP до +{limits['max_tp_percent']}% для {symbol}")
                        score *= 0.6
                    if entry_price > 0:
                        abs_move = entry_price * (max_tp / 100.0)
                        if abs_move > limits['max_abs_move_usd']:
                            warnings.append(f"⚠️ НЕРЕАЛИСТИЧНОЕ абсолютное движение по {symbol}: ${abs_move:,.0f} > ${limits['max_abs_move_usd']:,.0f}")
                            recommendations.append("Снизить TP уровни из-за чрезмерного абсолютного движения")
                            score *= 0.6
                elif max_tp > self.realistic_limits['max_tp_percent']:
                    warnings.append(f"⚠️ Слишком высокий TP: +{max_tp}%")
                    recommendations.append("Снизить максимальный TP до +15%")
                    score *= 0.9
            
            # Проверяем R/R соотношение
            min_tp = min([tp.get('percent', 0) for tp in tp_levels]) if tp_levels else 4
            risk_reward = min_tp / stop_loss if stop_loss > 0 else 0
            
            if risk_reward < 0.15:  # Меньше 1:6.7
                warnings.append(f"⚠️ Низкое R/R соотношение: 1:{1/risk_reward:.1f}")
                recommendations.append("Улучшить соотношение риск/прибыль")
                score *= 0.9
            
            return score
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки TP/SL: {e}")
            return 0.8
    
    def _validate_probabilities(self, tp_probabilities: List[Any], warnings: List[str], 
                               recommendations: List[str]) -> float:
        """Проверка реалистичности вероятностей"""
        try:
            if not tp_probabilities:
                return 1.0
            
            score = 1.0
            
            for tp in tp_probabilities:
                prob = getattr(tp, 'probability', 50)
                tp_percent = getattr(tp, 'percent', 0)
                
                # Проверка на нереалистично высокие вероятности
                if prob > 90 and tp_percent > 8:
                    warnings.append(f"⚠️ Нереалистично высокая вероятность: {prob}% для TP+{tp_percent}%")
                    recommendations.append("Снизить ожидания по вероятностям")
                    score *= 0.85
                
                # Проверка на слишком низкие вероятности
                elif prob < 30 and tp_percent < 10:
                    warnings.append(f"⚠️ Слишком низкая вероятность: {prob}% для TP+{tp_percent}%")
                    recommendations.append("Пересмотреть TP уровни")
                    score *= 0.9
                
                # Проверка логичности убывания вероятностей
                if hasattr(tp, 'level') and tp.level > 1:
                    prev_tp = tp_probabilities[tp.level - 2] if tp.level <= len(tp_probabilities) else None
                    if prev_tp and prob > getattr(prev_tp, 'probability', 0):
                        warnings.append(f"⚠️ Нелогичное возрастание вероятности на TP{tp.level}")
                        score *= 0.9
            
            return score
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки вероятностей: {e}")
            return 0.9
    
    def _validate_market_conditions(self, market_data: Dict, warnings: List[str], 
                                   recommendations: List[str]) -> float:
        """Проверка рыночных условий"""
        try:
            score = 1.0
            
            # Проверка RSI экстремумов
            rsi = market_data.get('rsi', 50)
            if rsi > 80:
                warnings.append(f"⚠️ Экстремальная перекупленность: RSI={rsi}")
                recommendations.append("Дождаться коррекции RSI")
                score *= 0.9
            elif rsi < 20:
                warnings.append(f"⚠️ Экстремальная перепроданность: RSI={rsi}")
                recommendations.append("Дождаться восстановления RSI")
                score *= 0.9
            
            # Проверка Bollinger Bands
            bb_position = market_data.get('bb_position', 50)
            if bb_position > 95:
                warnings.append(f"⚠️ Цена далеко за верхней границей BB: {bb_position}%")
                recommendations.append("Высокий риск коррекции")
                score *= 0.85
            elif bb_position < 5:
                warnings.append(f"⚠️ Цена далеко за нижней границей BB: {bb_position}%")
                recommendations.append("Высокий риск отскока")
                score *= 0.85
            
            # Проверка волатильности
            atr = market_data.get('atr', 0)
            price = market_data.get('price', 1)
            if price > 0:
                volatility = (atr / price) * 100
                if volatility > self.realistic_limits['max_volatility']:
                    warnings.append(f"⚠️ Высокая волатильность: {volatility:.1f}%")
                    recommendations.append("Снизить размер позиции из-за волатильности")
                    score *= 0.9
            
            return score
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки рыночных условий: {e}")
            return 0.9
    
    def _validate_volume_volatility(self, market_data: Dict, warnings: List[str], 
                                   recommendations: List[str]) -> float:
        """Проверка объемов и волатильности"""
        try:
            score = 1.0
            
            # Проверка объемов
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio < self.realistic_limits['min_volume_ratio']:
                warnings.append(f"⚠️ Очень низкий объем: {volume_ratio:.1f}x")
                recommendations.append("Дождаться увеличения объемов")
                score *= 0.8
            elif volume_ratio > 10:  # Аномально высокий объем
                warnings.append(f"⚠️ Аномально высокий объем: {volume_ratio:.1f}x")
                recommendations.append("Проверить на наличие новостей")
                score *= 0.9
            
            return score
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки объемов: {e}")
            return 0.9
    
    def _check_manipulation_signs(self, market_data: Dict, warnings: List[str], 
                                 recommendations: List[str]) -> float:
        """Проверка признаков манипуляций"""
        try:
            score = 1.0
            
            # Проверка на pump & dump
            volume_ratio = market_data.get('volume_ratio', 1.0)
            momentum = market_data.get('momentum', 0)
            
            if volume_ratio > 5 and abs(momentum) > 10:
                warnings.append("⚠️ Возможные признаки манипуляций (Pump&Dump)")
                recommendations.append("Избегать входа при подозрении на манипуляции")
                score *= 0.7
            
            # Проверка на fakeout
            bb_position = market_data.get('bb_position', 50)
            rsi = market_data.get('rsi', 50)
            
            if (bb_position > 90 and rsi < 60) or (bb_position < 10 and rsi > 40):
                warnings.append("⚠️ Возможный Fakeout (расхождение BB и RSI)")
                recommendations.append("Дождаться подтверждения сигнала")
                score *= 0.8
            
            return score
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки манипуляций: {e}")
            return 0.9
    
    def _determine_risk_level(self, signal_data: Dict, market_data: Dict) -> str:
        """Определить уровень риска"""
        try:
            # Факторы риска
            risk_factors = 0
            
            # TP уровни
            tp_levels = signal_data.get('tp_levels', [])
            if tp_levels:
                max_tp = max([tp.get('percent', 0) for tp in tp_levels])
                if max_tp > 15:
                    risk_factors += 2
                elif max_tp > 10:
                    risk_factors += 1
            
            # SL уровень
            stop_loss = signal_data.get('stop_loss_percent', 20)
            if stop_loss > 20:
                risk_factors += 2
            elif stop_loss > 15:
                risk_factors += 1
            
            # Волатильность
            atr = market_data.get('atr', 0)
            price = market_data.get('price', 1)
            if price > 0:
                volatility = (atr / price) * 100
                if volatility > 10:
                    risk_factors += 2
                elif volatility > 5:
                    risk_factors += 1
            
            # Объемы
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:
                risk_factors += 1
            elif volume_ratio > 5:
                risk_factors += 1
            
            # Определяем уровень
            if risk_factors >= 5:
                return 'EXTREME'
            elif risk_factors >= 3:
                return 'HIGH'
            elif risk_factors >= 1:
                return 'MEDIUM'
            else:
                return 'LOW'
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка определения риска: {e}")
            return 'MEDIUM'
    
    def _get_default_check(self) -> RealismCheck:
        """Получить дефолтную проверку при ошибке"""
        return RealismCheck(
            is_realistic=False,
            confidence_score=50.0,
            warnings=["⚠️ Ошибка валидации"],
            recommendations=["Требуется ручная проверка"],
            risk_level='HIGH'
        )
    
    def get_validation_summary(self, check: RealismCheck) -> str:
        """Получить краткое описание валидации"""
        try:
            status = "✅ РЕАЛИСТИЧЕН" if check.is_realistic else "❌ НЕ РЕАЛИСТИЧЕН"
            risk_emoji = {
                'LOW': '🟢',
                'MEDIUM': '🟡', 
                'HIGH': '🟠',
                'EXTREME': '🔴'
            }
            
            summary = [
                f"{status} (Уверенность: {check.confidence_score:.0f}%)",
                f"{risk_emoji.get(check.risk_level, '⚪')} Риск: {check.risk_level}"
            ]
            
            if check.warnings:
                summary.append(f"⚠️ Предупреждения: {len(check.warnings)}")
            
            return " | ".join(summary)
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка создания summary: {e}")
            return f"Реалистичность: {check.confidence_score:.0f}%"

# Тестирование модуля
if __name__ == "__main__":
    validator = RealismValidator()
    
    # Тестовые данные - реалистичный сигнал
    realistic_signal = {
        'direction': 'buy',
        'stop_loss_percent': 20,
        'tp_levels': [
            {'percent': 4}, {'percent': 6}, {'percent': 8}
        ]
    }
    
    realistic_market = {
        'rsi': 35,
        'bb_position': 25,
        'volume_ratio': 1.2,
        'atr': 2.0,
        'price': 100.0,
        'momentum': 2.0
    }
    
    # Тестовые данные - нереалистичный сигнал
    unrealistic_signal = {
        'direction': 'buy',
        'stop_loss_percent': 30,
        'tp_levels': [
            {'percent': 25}, {'percent': 50}, {'percent': 100}
        ]
    }
    
    unrealistic_market = {
        'rsi': 85,
        'bb_position': 95,
        'volume_ratio': 0.1,
        'atr': 15.0,
        'price': 100.0,
        'momentum': 15.0
    }
    
    print("✅ Тест RealismValidator:")
    
    # Тест реалистичного сигнала
    check1 = validator.validate_signal(realistic_signal, realistic_market)
    print(f"\n🟢 Реалистичный сигнал:")
    print(validator.get_validation_summary(check1))
    
    # Тест нереалистичного сигнала
    check2 = validator.validate_signal(unrealistic_signal, unrealistic_market)
    print(f"\n🔴 Нереалистичный сигнал:")
    print(validator.get_validation_summary(check2))
    print(f"Предупреждения: {check2.warnings}")

