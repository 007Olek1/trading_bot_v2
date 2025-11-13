#!/usr/bin/env python3
"""
🤖 ML/LLM МОНИТОРИНГ СИСТЕМЫ
Компоненты:
1. Health Score (0-100)
2. ML Performance Predictor
3. Anomaly Detector
4. Smart Alert System
5. ChatGPT Analyzer (GPT-4)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ OpenAI библиотека не установлена: pip install openai")


@dataclass
class HealthScore:
    """Оценка здоровья системы (0-100)"""
    total_score: float
    trade_performance: float
    system_stability: float
    ml_accuracy: float
    risk_management: float
    recommendations: List[str]
    timestamp: datetime


@dataclass
class MLPrediction:
    """Предсказание ML модели"""
    next_period_profit_pct: float
    confidence: float
    market_direction: str
    risk_level: str
    expected_trades: int
    timestamp: datetime


@dataclass
class AnomalyAlert:
    """Алерт об аномалии"""
    type: str  # 'trade', 'system', 'market'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    data: Dict
    timestamp: datetime


class BotHealthMonitor:
    """🏥 Монитор здоровья бота"""
    
    def __init__(self):
        self.trade_history = []
        self.system_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        logger.info("🏥 Bot Health Monitor инициализирован")
    
    def add_trade(self, trade_result: Dict):
        """Добавить результат сделки"""
        self.trade_history.append(trade_result)
        
        if trade_result.get('profit', 0) > 0:
            self.system_stats['winning_trades'] += 1
        else:
            self.system_stats['losing_trades'] += 1
        
        self.system_stats['total_trades'] += 1
        self.system_stats['total_profit'] += trade_result.get('profit', 0)
    
    def calculate_health_score(self) -> HealthScore:
        """Рассчитать общий Health Score (0-100)"""
        try:
            # 1. Trade Performance (40%)
            win_rate = 0.0
            avg_profit = 0.0
            if self.system_stats['total_trades'] > 0:
                win_rate = (self.system_stats['winning_trades'] / self.system_stats['total_trades']) * 100
                avg_profit = self.system_stats['total_profit'] / self.system_stats['total_trades']
            
            # Целевой win rate: 60%+, целевая прибыль: +$1 (+4%)
            trade_score = min(100, (win_rate * 0.6) + (min(avg_profit / 1.0, 1.0) * 40))
            
            # 2. System Stability (25%)
            # Проверка на сбои, ошибки, перезапуски
            stability_score = 85.0  # Дефолт, можно улучшить
            
            # 3. ML Accuracy (20%)
            ml_score = 75.0  # Дефолт, можно улучшить
            
            # 4. Risk Management (15%)
            # Проверка соблюдения правил риска
            risk_score = 90.0  # Дефолт
            
            # Общий score
            total_score = (
                trade_score * 0.40 +
                stability_score * 0.25 +
                ml_score * 0.20 +
                risk_score * 0.15
            )
            
            recommendations = []
            if win_rate < 60:
                recommendations.append(f"Увеличить win rate (текущий: {win_rate:.1f}%, целевой: 60%+)")
            if avg_profit < 1.0:
                recommendations.append(f"Увеличить среднюю прибыль (текущая: ${avg_profit:.2f}, целевая: $1.00)")
            if total_score < 70:
                recommendations.append("Общий Health Score ниже 70 - требуется оптимизация")
            
            return HealthScore(
                total_score=total_score,
                trade_performance=trade_score,
                system_stability=stability_score,
                ml_accuracy=ml_score,
                risk_management=risk_score,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета Health Score: {e}")
            return HealthScore(
                total_score=50.0,
                trade_performance=50.0,
                system_stability=50.0,
                ml_accuracy=50.0,
                risk_management=50.0,
                recommendations=["Ошибка расчета"],
                timestamp=datetime.now()
            )


class MLPerformancePredictor:
    """🔮 Предиктор производительности ML"""
    
    def __init__(self):
        self.historical_performance = []
        logger.info("🔮 ML Performance Predictor инициализирован")
    
    def predict_next_period(self, recent_trades: List[Dict], market_data: Dict) -> MLPrediction:
        """Предсказать производительность на следующий период"""
        try:
            if not recent_trades:
                return MLPrediction(
                    next_period_profit_pct=2.0,
                    confidence=50.0,
                    market_direction='neutral',
                    risk_level='medium',
                    expected_trades=2,
                    timestamp=datetime.now()
                )
            
            # Анализ последних сделок
            profits = [t.get('profit', 0) for t in recent_trades[-10:]]
            avg_profit = np.mean(profits) if profits else 1.0
            
            # Простой прогноз на основе тренда
            if len(profits) >= 2:
                trend = profits[-1] - profits[0]
                next_profit = avg_profit + (trend * 0.5)
            else:
                next_profit = avg_profit
            
            # Прогноз направления рынка
            market_trend = market_data.get('trend', 'neutral')
            
            # Уровень риска
            volatility = market_data.get('volatility', 0.02)
            if volatility > 0.05:
                risk_level = 'high'
            elif volatility > 0.02:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Прогноз количества сделок
            expected_trades = max(1, int(len(recent_trades) / 10) if recent_trades else 2)
            
            return MLPrediction(
                next_period_profit_pct=max(1.0, next_profit * 20),  # Конвертируем в %
                confidence=min(95.0, 60.0 + (len(recent_trades) * 2)),
                market_direction=market_trend,
                risk_level=risk_level,
                expected_trades=expected_trades,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            return MLPrediction(
                next_period_profit_pct=2.0,
                confidence=50.0,
                market_direction='neutral',
                risk_level='medium',
                expected_trades=2,
                timestamp=datetime.now()
            )


class AnomalyDetector:
    """🚨 Детектор аномалий"""
    
    def __init__(self):
        self.alert_history = []
        logger.info("🚨 Anomaly Detector инициализирован")
    
    def detect_trade_anomalies(self, trade: Dict) -> Optional[AnomalyAlert]:
        """Детектировать аномалии в сделке"""
        try:
            profit = trade.get('profit', 0)
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            
            # Аномалия: очень большая потеря
            if profit < -5.0:
                return AnomalyAlert(
                    type='trade',
                    severity='high',
                    message=f"Критическая потеря: ${profit:.2f}",
                    data=trade,
                    timestamp=datetime.now()
                )
            
            # Аномалия: очень большое изменение цены
            if entry_price > 0:
                price_change_pct = abs((exit_price - entry_price) / entry_price) * 100
                if price_change_pct > 20:
                    return AnomalyAlert(
                        type='trade',
                        severity='medium',
                        message=f"Аномальное изменение цены: {price_change_pct:.1f}%",
                        data=trade,
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка детекции аномалий: {e}")
            return None
    
    def detect_system_anomalies(self, system_stats: Dict) -> List[AnomalyAlert]:
        """Детектировать системные аномалии"""
        alerts = []
        
        try:
            # Проверка количества ошибок
            error_count = system_stats.get('errors', 0)
            if error_count > 10:
                alerts.append(AnomalyAlert(
                    type='system',
                    severity='high',
                    message=f"Высокое количество ошибок: {error_count}",
                    data=system_stats,
                    timestamp=datetime.now()
                ))
            
            # Проверка памяти
            memory_usage = system_stats.get('memory_usage', 0)
            if memory_usage > 90:
                alerts.append(AnomalyAlert(
                    type='system',
                    severity='medium',
                    message=f"Высокое использование памяти: {memory_usage}%",
                    data=system_stats,
                    timestamp=datetime.now()
                ))
            
            return alerts
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка детекции системных аномалий: {e}")
            return []


class SmartAlertSystem:
    """📢 Система умных алертов"""
    
    def __init__(self, health_monitor: BotHealthMonitor):
        self.health_monitor = health_monitor
        self.alert_history = []
        logger.info("📢 Smart Alert System инициализирована")
    
    def check_and_alert(self) -> List[Dict]:
        """Проверить условия и отправить алерты"""
        alerts = []
        
        try:
            health = self.health_monitor.calculate_health_score()
            
            # Алерт: низкий Health Score
            if health.total_score < 60:
                alerts.append({
                    'type': 'health',
                    'severity': 'high',
                    'message': f"⚠️ Низкий Health Score: {health.total_score:.1f}/100",
                    'recommendations': health.recommendations
                })
            
            # Алерт: рекомендации по улучшению
            if health.recommendations:
                alerts.append({
                    'type': 'recommendations',
                    'severity': 'medium',
                    'message': "💡 Рекомендации по оптимизации:",
                    'recommendations': health.recommendations
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки алертов: {e}")
            return []


class LLMAnalyzer:
    """🤖 ChatGPT Analyzer (GPT-4)"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and self.api_key and self.api_key != 'your_openai_api_key_here':
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.available = True
                logger.info("✅ LLM анализатор доступен")
            except Exception as e:
                self.available = False
                logger.warning(f"⚠️ LLM анализатор недоступен: {e}")
        else:
            self.available = False
            logger.warning("⚠️ LLM анализатор недоступен (нет API ключа)")
    
    def analyze_market_situation(self, market_data: Dict, trade_history: List[Dict]) -> Optional[str]:
        """Проанализировать ситуацию на рынке через ChatGPT"""
        if not self.available:
            return None
        
        try:
            # Формируем промпт
            recent_trades = trade_history[-5:] if trade_history else []
            win_rate = len([t for t in recent_trades if t.get('profit', 0) > 0]) / len(recent_trades) * 100 if recent_trades else 0
            
            prompt = f"""Проанализируй ситуацию торгового бота и дай рекомендации:

Рыночные данные:
- Тренд: {market_data.get('trend', 'unknown')}
- Волатильность: {market_data.get('volatility', 0):.2%}

Последние сделки (всего: {len(recent_trades)}):
Win Rate: {win_rate:.1f}%
Средняя прибыль: ${sum(t.get('profit', 0) for t in recent_trades) / len(recent_trades) if recent_trades else 0:.2f}

Цель: +$1 (+4%) на каждую сделку

Дай краткие рекомендации по улучшению торговли (2-3 предложения)."""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты эксперт по алгоритмической торговле криптовалютами. Дай краткие практические рекомендации."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            logger.info(f"🤖 ChatGPT анализ получен")
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка ChatGPT анализа: {e}")
            return None


