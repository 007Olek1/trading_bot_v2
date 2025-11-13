#!/usr/bin/env python3
"""
🎭 ПРОДВИНУТАЯ СИСТЕМА ОБНАРУЖЕНИЯ МАНИПУЛЯЦИЙ
==============================================

Функции:
- Анализ паттернов мошенничества
- Обнаружение необычной активности
- Защита от pump & dump схем
- Детекция манипуляций в реальном времени
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from collections import deque
import statistics
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ManipulationAlert:
    """🚨 Алерт о манипуляции"""
    timestamp: datetime
    symbol: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    description: str
    evidence: Dict[str, Any]
    recommended_action: str

@dataclass
class MarketAnomaly:
    """🔍 Аномалия рынка"""
    timestamp: datetime
    symbol: str
    anomaly_type: str
    score: float
    features: Dict[str, float]
    explanation: str

class AdvancedManipulationDetector:
    """🎭 Продвинутый детектор манипуляций"""
    
    def __init__(self):
        # Настройки детекции
        self.settings = {
            'volume_spike_threshold': 5.0,      # Порог всплеска объёма
            'price_spike_threshold': 0.15,      # Порог всплеска цены (15%)
            'unusual_pattern_threshold': 0.8,   # Порог необычных паттернов
            'pump_dump_timeframe_minutes': 30,  # Временное окно для pump&dump
            'wash_trading_threshold': 0.7,      # Порог для wash trading
            'spoofing_threshold': 0.6,          # Порог для spoofing
            'min_data_points': 100,             # Минимальное количество точек данных
            'lookback_periods': [5, 10, 20, 50] # Периоды для анализа
        }
        
        # История данных для анализа
        self.price_history = {}
        self.volume_history = {}
        self.order_book_history = {}
        self.trade_history = {}
        
        # Модели для детекции аномалий
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Статистика
        self.stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'false_positives': 0,
            'true_positives': 0,
            'last_analysis': None
        }
        
        logger.info("🎭 Продвинутый детектор манипуляций инициализирован")
    
    def analyze_market_data(self, symbol: str, market_data: Dict) -> List[ManipulationAlert]:
        """🔍 Анализ рыночных данных на предмет манипуляций"""
        alerts = []
        
        # Обновляем историю данных
        self._update_data_history(symbol, market_data)
        
        # Проверяем достаточность данных
        if len(self.price_history.get(symbol, [])) < self.settings['min_data_points']:
            return alerts
        
        # 1. Детекция Pump & Dump
        pump_dump_alerts = self._detect_pump_and_dump(symbol)
        alerts.extend(pump_dump_alerts)
        
        # 2. Детекция Wash Trading
        wash_trading_alerts = self._detect_wash_trading(symbol)
        alerts.extend(wash_trading_alerts)
        
        # 3. Детекция Spoofing
        spoofing_alerts = self._detect_spoofing(symbol)
        alerts.extend(spoofing_alerts)
        
        # 4. Детекция необычных паттернов объёма
        volume_pattern_alerts = self._detect_unusual_volume_patterns(symbol)
        alerts.extend(volume_pattern_alerts)
        
        # 5. Детекция ценовых манипуляций
        price_manipulation_alerts = self._detect_price_manipulation(symbol)
        alerts.extend(price_manipulation_alerts)
        
        # 6. Детекция синхронизированных торгов
        synchronized_trading_alerts = self._detect_synchronized_trading(symbol)
        alerts.extend(synchronized_trading_alerts)
        
        # Обновляем статистику
        self.stats['total_alerts'] += len(alerts)
        self.stats['last_analysis'] = datetime.now()
        
        for alert in alerts:
            alert_type = alert.alert_type
            severity = alert.severity
            
            self.stats['alerts_by_type'][alert_type] = self.stats['alerts_by_type'].get(alert_type, 0) + 1
            self.stats['alerts_by_severity'][severity] = self.stats['alerts_by_severity'].get(severity, 0) + 1
        
        return alerts
    
    def _update_data_history(self, symbol: str, market_data: Dict):
        """📊 Обновление истории данных"""
        timestamp = datetime.now()
        
        # История цен
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=1000)
        
        self.price_history[symbol].append({
            'timestamp': timestamp,
            'price': market_data.get('price', 0),
            'high': market_data.get('high', 0),
            'low': market_data.get('low', 0),
            'open': market_data.get('open', 0),
            'close': market_data.get('close', 0)
        })
        
        # История объёмов
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=1000)
        
        self.volume_history[symbol].append({
            'timestamp': timestamp,
            'volume': market_data.get('volume', 0),
            'quote_volume': market_data.get('quote_volume', 0),
            'trades_count': market_data.get('trades_count', 0)
        })
        
        # История ордербука (если доступна)
        if 'order_book' in market_data:
            if symbol not in self.order_book_history:
                self.order_book_history[symbol] = deque(maxlen=500)
            
            self.order_book_history[symbol].append({
                'timestamp': timestamp,
                'bids': market_data['order_book'].get('bids', []),
                'asks': market_data['order_book'].get('asks', []),
                'bid_ask_spread': market_data['order_book'].get('spread', 0)
            })
    
    def _detect_pump_and_dump(self, symbol: str) -> List[ManipulationAlert]:
        """🚀 Детекция Pump & Dump схем"""
        alerts = []
        
        if symbol not in self.price_history or symbol not in self.volume_history:
            return alerts
        
        price_data = list(self.price_history[symbol])
        volume_data = list(self.volume_history[symbol])
        
        if len(price_data) < 20:
            return alerts
        
        # Анализируем последние 30 минут
        recent_prices = price_data[-30:]
        recent_volumes = volume_data[-30:]
        
        if len(recent_prices) < 10:
            return alerts
        
        # Рассчитываем изменения цены и объёма
        price_changes = []
        volume_changes = []
        
        for i in range(1, len(recent_prices)):
            price_change = (recent_prices[i]['price'] - recent_prices[i-1]['price']) / recent_prices[i-1]['price']
            volume_change = recent_volumes[i]['volume'] / recent_volumes[i-1]['volume'] if recent_volumes[i-1]['volume'] > 0 else 1
            
            price_changes.append(price_change)
            volume_changes.append(volume_change)
        
        # Проверяем на pump (резкий рост цены с большим объёмом)
        max_price_change = max(price_changes) if price_changes else 0
        max_volume_change = max(volume_changes) if volume_changes else 0
        
        if max_price_change > self.settings['price_spike_threshold'] and max_volume_change > self.settings['volume_spike_threshold']:
            # Проверяем на dump (резкое падение после роста)
            price_after_pump = recent_prices[-1]['price']
            max_price = max(p['price'] for p in recent_prices)
            
            if price_after_pump < max_price * 0.9:  # Падение на 10% от максимума
                confidence = min(max_price_change * max_volume_change / 10, 1.0)
                
                alert = ManipulationAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type='pump_and_dump',
                    severity='high' if confidence > 0.7 else 'medium',
                    confidence=confidence,
                    description=f"Обнаружена схема Pump & Dump: рост на {max_price_change:.1%} с объёмом {max_volume_change:.1f}x",
                    evidence={
                        'max_price_change': max_price_change,
                        'max_volume_change': max_volume_change,
                        'price_drop': (max_price - price_after_pump) / max_price,
                        'timeframe_minutes': len(recent_prices)
                    },
                    recommended_action='Избегать торговли до стабилизации'
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_wash_trading(self, symbol: str) -> List[ManipulationAlert]:
        """🔄 Детекция Wash Trading"""
        alerts = []
        
        if symbol not in self.trade_history:
            return alerts
        
        trade_data = list(self.trade_history[symbol])
        
        if len(trade_data) < 50:
            return alerts
        
        # Анализируем паттерны торгов
        recent_trades = trade_data[-100:]  # Последние 100 сделок
        
        # Проверяем на подозрительные паттерны
        suspicious_patterns = []
        
        # 1. Проверка на одинаковые размеры ордеров
        order_sizes = [trade.get('amount', 0) for trade in recent_trades]
        if len(set(order_sizes)) < len(order_sizes) * 0.3:  # Менее 30% уникальных размеров
            suspicious_patterns.append('identical_order_sizes')
        
        # 2. Проверка на чередование покупок и продаж
        buy_sell_pattern = [trade.get('side', '') for trade in recent_trades]
        alternating_patterns = 0
        
        for i in range(1, len(buy_sell_pattern)):
            if buy_sell_pattern[i] != buy_sell_pattern[i-1]:
                alternating_patterns += 1
        
        if alternating_patterns / len(buy_sell_pattern) > 0.8:  # Более 80% чередований
            suspicious_patterns.append('alternating_buy_sell')
        
        # 3. Проверка на подозрительную регулярность времени
        timestamps = [trade.get('timestamp', datetime.now()) for trade in recent_trades]
        time_intervals = []
        
        for i in range(1, len(timestamps)):
            if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_intervals.append(interval)
        
        if time_intervals:
            interval_std = statistics.stdev(time_intervals) if len(time_intervals) > 1 else 0
            interval_mean = statistics.mean(time_intervals)
            
            if interval_std < interval_mean * 0.1:  # Очень низкая вариативность
                suspicious_patterns.append('regular_timing')
        
        # Если найдено несколько подозрительных паттернов
        if len(suspicious_patterns) >= 2:
            confidence = len(suspicious_patterns) / 3.0
            
            alert = ManipulationAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='wash_trading',
                severity='high' if confidence > 0.7 else 'medium',
                confidence=confidence,
                description=f"Подозрение на Wash Trading: {', '.join(suspicious_patterns)}",
                evidence={
                    'suspicious_patterns': suspicious_patterns,
                    'alternating_ratio': alternating_patterns / len(buy_sell_pattern),
                    'unique_order_sizes': len(set(order_sizes)) / len(order_sizes),
                    'time_regularity': interval_std / interval_mean if interval_mean > 0 else 0
                },
                recommended_action='Проверить источники объёма и избегать торговли'
            )
            alerts.append(alert)
        
        return alerts
    
    def _detect_spoofing(self, symbol: str) -> List[ManipulationAlert]:
        """🎭 Детекция Spoofing"""
        alerts = []
        
        if symbol not in self.order_book_history:
            return alerts
        
        order_book_data = list(self.order_book_history[symbol])
        
        if len(order_book_data) < 20:
            return alerts
        
        # Анализируем последние данные ордербука
        recent_books = order_book_data[-20:]
        
        spoofing_indicators = []
        
        for i in range(1, len(recent_books)):
            current_book = recent_books[i]
            previous_book = recent_books[i-1]
            
            # Проверяем на большие ордера, которые быстро исчезают
            current_bids = current_book.get('bids', [])
            previous_bids = previous_book.get('bids', [])
            
            if current_bids and previous_bids:
                # Ищем большие ордера, которые исчезли
                large_orders_disappeared = 0
                
                for prev_bid in previous_bids:
                    if len(prev_bid) >= 2 and prev_bid[1] > 1000:  # Большой ордер
                        # Проверяем, есть ли он в текущем ордербуке
                        found = False
                        for curr_bid in current_bids:
                            if len(curr_bid) >= 2 and abs(curr_bid[0] - prev_bid[0]) < 0.01:
                                found = True
                                break
                        
                        if not found:
                            large_orders_disappeared += 1
                
                if large_orders_disappeared > 2:
                    spoofing_indicators.append('large_orders_disappeared')
            
            # Проверяем на подозрительные изменения в спреде
            current_spread = current_book.get('bid_ask_spread', 0)
            previous_spread = previous_book.get('bid_ask_spread', 0)
            
            if previous_spread > 0 and abs(current_spread - previous_spread) / previous_spread > 0.5:
                spoofing_indicators.append('spread_manipulation')
        
        # Если найдены индикаторы spoofing
        if len(spoofing_indicators) >= 2:
            confidence = min(len(spoofing_indicators) / 3.0, 1.0)
            
            alert = ManipulationAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='spoofing',
                severity='high' if confidence > 0.7 else 'medium',
                confidence=confidence,
                description=f"Подозрение на Spoofing: {', '.join(spoofing_indicators)}",
                evidence={
                    'spoofing_indicators': spoofing_indicators,
                    'analysis_period': len(recent_books)
                },
                recommended_action='Осторожность при торговле, возможны ложные сигналы'
            )
            alerts.append(alert)
        
        return alerts
    
    def _detect_unusual_volume_patterns(self, symbol: str) -> List[ManipulationAlert]:
        """📊 Детекция необычных паттернов объёма"""
        alerts = []
        
        if symbol not in self.volume_history:
            return alerts
        
        volume_data = list(self.volume_history[symbol])
        
        if len(volume_data) < 50:
            return alerts
        
        # Рассчитываем статистики объёма
        volumes = [v['volume'] for v in volume_data]
        recent_volumes = volumes[-20:]
        historical_volumes = volumes[:-20] if len(volumes) > 20 else volumes
        
        if not historical_volumes:
            return alerts
        
        # Статистики исторического объёма
        hist_mean = statistics.mean(historical_volumes)
        hist_std = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0
        
        # Проверяем на аномалии в объёме
        anomalies = []
        
        for i, volume in enumerate(recent_volumes):
            # Z-score для объёма
            if hist_std > 0:
                z_score = abs(volume - hist_mean) / hist_std
                
                if z_score > 3:  # Сильное отклонение
                    anomalies.append({
                        'index': i,
                        'volume': volume,
                        'z_score': z_score,
                        'deviation': (volume - hist_mean) / hist_mean
                    })
        
        # Если найдено много аномалий
        if len(anomalies) > 3:
            max_deviation = max(anomaly['deviation'] for anomaly in anomalies)
            confidence = min(len(anomalies) / 10.0, 1.0)
            
            alert = ManipulationAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='unusual_volume_patterns',
                severity='medium' if confidence > 0.5 else 'low',
                confidence=confidence,
                description=f"Необычные паттерны объёма: {len(anomalies)} аномалий за последние 20 периодов",
                evidence={
                    'anomalies_count': len(anomalies),
                    'max_deviation': max_deviation,
                    'historical_mean': hist_mean,
                    'historical_std': hist_std
                },
                recommended_action='Мониторить объём и избегать торговли при экстремальных значениях'
            )
            alerts.append(alert)
        
        return alerts
    
    def _detect_price_manipulation(self, symbol: str) -> List[ManipulationAlert]:
        """💰 Детекция ценовых манипуляций"""
        alerts = []
        
        if symbol not in self.price_history:
            return alerts
        
        price_data = list(self.price_history[symbol])
        
        if len(price_data) < 100:
            return alerts
        
        # Анализируем ценовые паттерны
        prices = [p['price'] for p in price_data]
        recent_prices = prices[-50:]
        
        # Проверяем на подозрительные ценовые движения
        manipulation_indicators = []
        
        # 1. Проверка на "лестничные" движения цены
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        # Ищем последовательные движения в одном направлении
        consecutive_moves = 0
        max_consecutive = 0
        current_direction = 0
        
        for change in price_changes:
            if change > 0:
                if current_direction == 1:
                    consecutive_moves += 1
                else:
                    consecutive_moves = 1
                    current_direction = 1
            elif change < 0:
                if current_direction == -1:
                    consecutive_moves += 1
                else:
                    consecutive_moves = 1
                    current_direction = -1
            else:
                consecutive_moves = 0
                current_direction = 0
            
            max_consecutive = max(max_consecutive, consecutive_moves)
        
        if max_consecutive > 8:  # Более 8 последовательных движений
            manipulation_indicators.append('ladder_pattern')
        
        # 2. Проверка на подозрительную волатильность
        price_volatility = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
        historical_volatility = statistics.stdev(prices[:-50]) if len(prices) > 50 else price_volatility
        
        if historical_volatility > 0 and price_volatility > historical_volatility * 2:
            manipulation_indicators.append('excessive_volatility')
        
        # 3. Проверка на подозрительные ценовые уровни
        # Ищем повторяющиеся ценовые уровни (возможные манипуляции)
        price_levels = {}
        for price in recent_prices:
            rounded_price = round(price, 2)  # Округляем до 2 знаков
            price_levels[rounded_price] = price_levels.get(rounded_price, 0) + 1
        
        # Если много сделок на одном уровне
        max_level_count = max(price_levels.values()) if price_levels else 0
        if max_level_count > len(recent_prices) * 0.3:  # Более 30% сделок на одном уровне
            manipulation_indicators.append('price_level_manipulation')
        
        # Если найдены индикаторы манипуляции
        if manipulation_indicators:
            confidence = len(manipulation_indicators) / 3.0
            
            alert = ManipulationAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='price_manipulation',
                severity='high' if confidence > 0.7 else 'medium',
                confidence=confidence,
                description=f"Подозрение на ценовые манипуляции: {', '.join(manipulation_indicators)}",
                evidence={
                    'manipulation_indicators': manipulation_indicators,
                    'max_consecutive_moves': max_consecutive,
                    'volatility_ratio': price_volatility / historical_volatility if historical_volatility > 0 else 1,
                    'price_level_concentration': max_level_count / len(recent_prices)
                },
                recommended_action='Осторожность при торговле, возможны искусственные движения цены'
            )
            alerts.append(alert)
        
        return alerts
    
    def _detect_synchronized_trading(self, symbol: str) -> List[ManipulationAlert]:
        """🔄 Детекция синхронизированных торгов"""
        alerts = []
        
        if symbol not in self.trade_history:
            return alerts
        
        trade_data = list(self.trade_history[symbol])
        
        if len(trade_data) < 100:
            return alerts
        
        # Анализируем временные паттерны торгов
        recent_trades = trade_data[-200:]  # Последние 200 сделок
        
        # Группируем сделки по временным интервалам
        time_intervals = {}
        
        for trade in recent_trades:
            timestamp = trade.get('timestamp', datetime.now())
            if isinstance(timestamp, datetime):
                # Группируем по 5-минутным интервалам
                interval_key = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
                
                if interval_key not in time_intervals:
                    time_intervals[interval_key] = []
                
                time_intervals[interval_key].append(trade)
        
        # Анализируем синхронизацию
        synchronized_periods = 0
        
        for interval, trades in time_intervals.items():
            if len(trades) > 10:  # Много сделок в одном интервале
                # Проверяем на подозрительную синхронизацию
                trade_sizes = [trade.get('amount', 0) for trade in trades]
                trade_sides = [trade.get('side', '') for trade in trades]
                
                # Если много сделок одинакового размера
                unique_sizes = len(set(trade_sizes))
                if unique_sizes < len(trade_sizes) * 0.3:  # Менее 30% уникальных размеров
                    synchronized_periods += 1
                
                # Если много сделок в одном направлении
                buy_count = trade_sides.count('buy')
                sell_count = trade_sides.count('sell')
                
                if buy_count > len(trades) * 0.8 or sell_count > len(trades) * 0.8:
                    synchronized_periods += 1
        
        # Если найдено много синхронизированных периодов
        if synchronized_periods > len(time_intervals) * 0.3:  # Более 30% периодов подозрительны
            confidence = synchronized_periods / len(time_intervals)
            
            alert = ManipulationAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='synchronized_trading',
                severity='medium' if confidence > 0.5 else 'low',
                confidence=confidence,
                description=f"Подозрение на синхронизированные торги: {synchronized_periods} подозрительных периодов",
                evidence={
                    'synchronized_periods': synchronized_periods,
                    'total_periods': len(time_intervals),
                    'analysis_trades': len(recent_trades)
                },
                recommended_action='Мониторить торговые паттерны и источники объёма'
            )
            alerts.append(alert)
        
        return alerts
    
    def get_manipulation_statistics(self) -> Dict:
        """📊 Статистика детекции манипуляций"""
        return {
            'total_alerts': self.stats['total_alerts'],
            'alerts_by_type': self.stats['alerts_by_type'],
            'alerts_by_severity': self.stats['alerts_by_severity'],
            'false_positives': self.stats['false_positives'],
            'true_positives': self.stats['true_positives'],
            'last_analysis': self.stats['last_analysis'].isoformat() if self.stats['last_analysis'] else None,
            'symbols_monitored': list(self.price_history.keys()),
            'settings': self.settings
        }
    
    def update_trade_history(self, symbol: str, trade_data: Dict):
        """📝 Обновление истории торгов"""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=1000)
        
        self.trade_history[symbol].append({
            'timestamp': datetime.now(),
            'amount': trade_data.get('amount', 0),
            'price': trade_data.get('price', 0),
            'side': trade_data.get('side', ''),
            'trade_id': trade_data.get('id', '')
        })
    
    def get_risk_assessment(self, symbol: str) -> Dict:
        """⚠️ Оценка риска манипуляций"""
        if symbol not in self.price_history:
            return {'risk_level': 'unknown', 'confidence': 0.0}
        
        # Анализируем последние данные
        recent_alerts = self.analyze_market_data(symbol, {})
        
        # Рассчитываем уровень риска
        high_severity_alerts = [alert for alert in recent_alerts if alert.severity == 'high']
        medium_severity_alerts = [alert for alert in recent_alerts if alert.severity == 'medium']
        
        risk_score = len(high_severity_alerts) * 3 + len(medium_severity_alerts) * 1
        
        if risk_score >= 6:
            risk_level = 'critical'
        elif risk_score >= 3:
            risk_level = 'high'
        elif risk_score >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        confidence = min(risk_score / 10.0, 1.0)
        
        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'risk_score': risk_score,
            'recent_alerts': len(recent_alerts),
            'high_severity_alerts': len(high_severity_alerts),
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """💡 Рекомендация по уровню риска"""
        recommendations = {
            'critical': 'Избегать торговли, высокий риск манипуляций',
            'high': 'Осторожность при торговле, мониторить объём и цену',
            'medium': 'Стандартная осторожность, следить за аномалиями',
            'low': 'Низкий риск, можно торговать с обычными мерами предосторожности',
            'unknown': 'Недостаточно данных для оценки риска'
        }
        
        return recommendations.get(risk_level, 'Неизвестный уровень риска')

# Пример использования
if __name__ == "__main__":
    detector = AdvancedManipulationDetector()
    
    print("🎭 ТЕСТ ПРОДВИНУТОГО ДЕТЕКТОРА МАНИПУЛЯЦИЙ")
    print("=" * 50)
    
    # Тестовые данные
    test_symbol = "BTCUSDT"
    
    # Симулируем рыночные данные
    for i in range(100):
        test_data = {
            'price': 50000 + np.random.randn() * 1000,
            'volume': np.random.randint(1000, 10000),
            'high': 50000 + np.random.randint(0, 500),
            'low': 50000 - np.random.randint(0, 500),
            'open': 50000 + np.random.randn() * 100,
            'close': 50000 + np.random.randn() * 100,
            'quote_volume': np.random.randint(1000000, 10000000),
            'trades_count': np.random.randint(100, 1000)
        }
        
        # Обновляем историю
        detector._update_data_history(test_symbol, test_data)
        
        # Добавляем тестовые сделки
        detector.update_trade_history(test_symbol, {
            'amount': np.random.uniform(0.1, 10.0),
            'price': test_data['price'],
            'side': np.random.choice(['buy', 'sell']),
            'id': f'trade_{i}'
        })
    
    # Анализируем данные
    alerts = detector.analyze_market_data(test_symbol, {
        'price': 51000,
        'volume': 15000,
        'high': 52000,
        'low': 50000,
        'open': 50500,
        'close': 51000
    })
    
    print(f"Найдено алертов: {len(alerts)}")
    
    for alert in alerts:
        print(f"🚨 {alert.alert_type.upper()}: {alert.description}")
        print(f"   Уверенность: {alert.confidence:.2%}")
        print(f"   Рекомендация: {alert.recommended_action}")
        print()
    
    # Оценка риска
    risk_assessment = detector.get_risk_assessment(test_symbol)
    print(f"⚠️ ОЦЕНКА РИСКА:")
    print(f"Уровень риска: {risk_assessment['risk_level']}")
    print(f"Уверенность: {risk_assessment['confidence']:.2%}")
    print(f"Рекомендация: {risk_assessment['recommendation']}")
    
    # Статистика
    stats = detector.get_manipulation_statistics()
    print(f"\n📊 СТАТИСТИКА:")
    print(f"Всего алертов: {stats['total_alerts']}")
    print(f"По типам: {stats['alerts_by_type']}")
    print(f"По серьёзности: {stats['alerts_by_severity']}")
    
    print("✅ Тест завершён!")






