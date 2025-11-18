"""
🧠 MARKET REGIME DETECTOR - ML-классификатор режимов рынка
Определяет: BULL (бычий), BEAR (медвежий), SIDEWAYS (боковой), VOLATILE (волатильный)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple
import logging
from enum import Enum
import json
from pathlib import Path


class MarketRegime(Enum):
    """Режимы рынка"""
    BULL = "bull"           # Бычий рынок - агрессивная торговля
    BEAR = "bear"           # Медвежий рынок - консервативная торговля
    SIDEWAYS = "sideways"   # Боковой рынок - сбалансированная торговля
    VOLATILE = "volatile"   # Высокая волатильность - осторожная торговля


class MarketRegimeDetector:
    """
    Детектор режима рынка с ML-классификацией
    
    Анализирует:
    - BTC тренд (главный индикатор)
    - Волатильность (ATR, BB width)
    - Объёмы
    - Корреляции между монетами
    - Исторические паттерны
    """
    
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        
        # История режимов
        self.regime_history = []
        self.current_regime = None
        self.regime_confidence = 0.0
        self.regime_changed_at = None
        
        # Параметры стабильности
        self.min_confirmation_time = 1800  # 30 минут подтверждения
        self.regime_change_threshold = 0.75  # 75% уверенность для смены
        
        # Кэш данных
        self.btc_data_cache = None
        self.market_data_cache = None
        self.last_update = None
        
        # Путь для сохранения истории
        self.history_file = Path("logs/regime_history.json")
        self.history_file.parent.mkdir(exist_ok=True)
        
        # Загружаем историю
        self._load_history()
    
    def detect_regime(self) -> Tuple[MarketRegime, float, Dict]:
        """
        Определяет текущий режим рынка
        
        Returns:
            (режим, уверенность, детали)
        """
        try:
            # Получаем данные BTC
            btc_data = self._get_btc_data()
            if btc_data is None:
                return self.current_regime or MarketRegime.SIDEWAYS, 0.5, {}
            
            # Анализируем различные аспекты
            trend_score = self._analyze_trend(btc_data)
            volatility_score = self._analyze_volatility(btc_data)
            volume_score = self._analyze_volume(btc_data)
            momentum_score = self._analyze_momentum(btc_data)
            
            # Комбинируем оценки
            regime, confidence = self._classify_regime(
                trend_score,
                volatility_score,
                volume_score,
                momentum_score
            )
            
            # Детали для логирования
            details = {
                'trend_score': trend_score,
                'volatility_score': volatility_score,
                'volume_score': volume_score,
                'momentum_score': momentum_score,
                'btc_price': btc_data['close'].iloc[-1],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Проверяем стабильность режима
            regime = self._stabilize_regime(regime, confidence)
            
            # Сохраняем в историю
            self._save_to_history(regime, confidence, details)
            
            return regime, confidence, details
            
        except Exception as e:
            self.logger.error(f"Ошибка определения режима рынка: {e}")
            return self.current_regime or MarketRegime.SIDEWAYS, 0.5, {}
    
    def _get_btc_data(self) -> Optional[pd.DataFrame]:
        """Получает данные BTC для анализа"""
        try:
            # Используем кэш если данные свежие (< 5 минут)
            if (self.btc_data_cache is not None and 
                self.last_update and 
                (datetime.now(timezone.utc) - self.last_update).seconds < 300):
                return self.btc_data_cache
            
            # Получаем дневные данные BTC
            response = self.client.get_kline(
                category="linear",
                symbol="BTCUSDT",
                interval="D",  # Дневной таймфрейм
                limit=100
            )
            
            if response['retCode'] != 0:
                return None
            
            klines = response['result']['list']
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Кэшируем
            self.btc_data_cache = df
            self.last_update = datetime.now(timezone.utc)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка получения BTC данных: {e}")
            return None
    
    def _analyze_trend(self, df: pd.DataFrame) -> float:
        """
        Анализ тренда
        Returns: -1.0 (сильный медвежий) до +1.0 (сильный бычий)
        """
        # EMA
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        ema_200 = df['close'].ewm(span=200).mean() if len(df) >= 200 else ema_50
        
        current_price = df['close'].iloc[-1]
        
        # Оценка позиции цены относительно EMA
        score = 0.0
        
        # Цена vs EMA20
        if current_price > ema_20.iloc[-1]:
            score += 0.3
        else:
            score -= 0.3
        
        # EMA20 vs EMA50
        if ema_20.iloc[-1] > ema_50.iloc[-1]:
            score += 0.3
        else:
            score -= 0.3
        
        # EMA50 vs EMA200
        if ema_50.iloc[-1] > ema_200.iloc[-1]:
            score += 0.4
        else:
            score -= 0.4
        
        return np.clip(score, -1.0, 1.0)
    
    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        """
        Анализ волатильности
        Returns: 0.0 (низкая) до 1.0 (высокая)
        """
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Нормализуем ATR относительно цены
        atr_percent = (atr / df['close']) * 100
        current_atr_percent = atr_percent.iloc[-1]
        
        # Bollinger Bands width
        bb_sma = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        bb_width = (bb_std / bb_sma) * 100
        current_bb_width = bb_width.iloc[-1]
        
        # Комбинируем показатели
        volatility_score = (current_atr_percent / 5.0 + current_bb_width / 10.0) / 2
        
        return np.clip(volatility_score, 0.0, 1.0)
    
    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """
        Анализ объёмов
        Returns: -1.0 (падающие) до +1.0 (растущие)
        """
        # Средний объём
        volume_sma = df['volume'].rolling(window=20).mean()
        current_volume = df['volume'].iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        
        # Отношение текущего к среднему
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Тренд объёма (растёт или падает)
        volume_trend = (volume_sma.iloc[-1] - volume_sma.iloc[-10]) / volume_sma.iloc[-10] if volume_sma.iloc[-10] > 0 else 0
        
        # Комбинируем
        score = (volume_ratio - 1.0) * 0.5 + volume_trend
        
        return np.clip(score, -1.0, 1.0)
    
    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """
        Анализ моментума
        Returns: -1.0 (сильный медвежий) до +1.0 (сильный бычий)
        """
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Нормализуем RSI к диапазону -1 до +1
        rsi_score = (current_rsi - 50) / 50
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        macd_histogram = macd - signal
        
        # Нормализуем MACD
        macd_score = np.sign(macd_histogram.iloc[-1])
        
        # Комбинируем
        momentum_score = (rsi_score * 0.6 + macd_score * 0.4)
        
        return np.clip(momentum_score, -1.0, 1.0)
    
    def _classify_regime(self, trend: float, volatility: float, 
                        volume: float, momentum: float) -> Tuple[MarketRegime, float]:
        """
        Классифицирует режим рынка на основе оценок
        
        Returns:
            (режим, уверенность)
        """
        # Веса для разных факторов
        weights = {
            'trend': 0.35,
            'momentum': 0.30,
            'volume': 0.20,
            'volatility': 0.15
        }
        
        # Общий score
        overall_score = (
            trend * weights['trend'] +
            momentum * weights['momentum'] +
            volume * weights['volume']
        )
        
        # Определяем режим
        if volatility > 0.7:
            # Высокая волатильность - особый режим
            regime = MarketRegime.VOLATILE
            confidence = volatility
            
        elif overall_score > 0.4:
            # Бычий рынок
            regime = MarketRegime.BULL
            confidence = abs(overall_score)
            
        elif overall_score < -0.4:
            # Медвежий рынок
            regime = MarketRegime.BEAR
            confidence = abs(overall_score)
            
        else:
            # Боковой рынок
            regime = MarketRegime.SIDEWAYS
            confidence = 1.0 - abs(overall_score)
        
        return regime, min(confidence, 1.0)
    
    def _stabilize_regime(self, new_regime: MarketRegime, confidence: float) -> MarketRegime:
        """
        Стабилизирует режим, избегая частых переключений
        """
        # Если нет текущего режима - устанавливаем новый
        if self.current_regime is None:
            self.current_regime = new_regime
            self.regime_confidence = confidence
            self.regime_changed_at = datetime.now(timezone.utc)
            self.logger.info(f"🎯 Установлен начальный режим: {new_regime.value.upper()} ({confidence:.1%})")
            return new_regime
        
        # Если режим не изменился - обновляем уверенность
        if new_regime == self.current_regime:
            self.regime_confidence = confidence
            return new_regime
        
        # Режим изменился - проверяем условия для смены
        time_since_change = (datetime.now(timezone.utc) - self.regime_changed_at).seconds
        
        # Требуем высокую уверенность и минимальное время для смены
        if (confidence >= self.regime_change_threshold and 
            time_since_change >= self.min_confirmation_time):
            
            old_regime = self.current_regime
            self.current_regime = new_regime
            self.regime_confidence = confidence
            self.regime_changed_at = datetime.now(timezone.utc)
            
            self.logger.info(
                f"🔄 СМЕНА РЕЖИМА: {old_regime.value.upper()} → {new_regime.value.upper()} "
                f"(уверенность: {confidence:.1%})"
            )
            
            return new_regime
        
        # Условия не выполнены - оставляем текущий режим
        return self.current_regime
    
    def _save_to_history(self, regime: MarketRegime, confidence: float, details: Dict):
        """Сохраняет режим в историю"""
        entry = {
            'regime': regime.value,
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details
        }
        
        self.regime_history.append(entry)
        
        # Ограничиваем размер истории (последние 1000 записей)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        # Сохраняем в файл
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.regime_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения истории режимов: {e}")
    
    def _load_history(self):
        """Загружает историю режимов"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    self.regime_history = json.load(f)
                
                # Восстанавливаем последний режим
                if self.regime_history:
                    last_entry = self.regime_history[-1]
                    self.current_regime = MarketRegime(last_entry['regime'])
                    self.regime_confidence = last_entry['confidence']
                    self.regime_changed_at = datetime.fromisoformat(last_entry['timestamp'])
                    
                    self.logger.info(
                        f"📊 Восстановлен режим из истории: {self.current_regime.value.upper()} "
                        f"({self.regime_confidence:.1%})"
                    )
        except Exception as e:
            self.logger.error(f"Ошибка загрузки истории режимов: {e}")
    
    def get_regime_stats(self) -> Dict:
        """Возвращает статистику по режимам"""
        if not self.regime_history:
            return {}
        
        stats = {regime.value: 0 for regime in MarketRegime}
        
        for entry in self.regime_history[-100:]:  # Последние 100 записей
            regime = entry['regime']
            stats[regime] = stats.get(regime, 0) + 1
        
        total = sum(stats.values())
        percentages = {k: (v/total*100) if total > 0 else 0 for k, v in stats.items()}
        
        return {
            'current_regime': self.current_regime.value if self.current_regime else None,
            'confidence': self.regime_confidence,
            'changed_at': self.regime_changed_at.isoformat() if self.regime_changed_at else None,
            'distribution': percentages,
            'total_records': len(self.regime_history)
        }
