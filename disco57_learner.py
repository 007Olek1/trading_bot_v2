#!/usr/bin/env python3
"""
Disco57 (DiscoRL) - Адаптивная система обучения
Обучается на каждой свече, адаптируется к рынку
"""

import logging
import json
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TradeFeatures:
    """Признаки для обучения"""
    ema_trend: float      # EMA9 vs EMA21 (1 = бычий, -1 = медвежий)
    momentum: float       # Momentum в %
    volume_ratio: float   # Объем vs средний
    rsi: float           # RSI нормализованный (0-1)
    spread: float        # Спред в %
    volatility: float    # ATR / price
    price_position: float # Позиция цены в диапазоне (0-1)
    candle_strength: float # Сила последней свечи
    stop_speed: float = 0.0      # Скорость ухода цены против позиции (0-1)
    book_imbalance: float = 0.5 # Доля бидов в стакане (0-1)
    book_delta: float = 0.5     # Нормализованная разница лучших цен (0-1)


class Disco57Learner:
    """
    DiscoRL - Reinforcement Learning на основе результатов сделок
    
    Обучается после каждой закрытой сделки:
    - Прибыльная сделка → увеличивает веса признаков
    - Убыточная сделка → уменьшает веса признаков
    """
    
    def __init__(self, model_path: str = '/opt/bot/data/disco57_model.json'):
        self.model_path = model_path
        self.learning_rate = 0.05
        self.min_confidence = 0.6
        
        # Веса для признаков (начальные значения)
        self.weights = {
            'ema_trend': 0.5,
            'momentum': 0.5,
            'volume_ratio': 0.5,
            'rsi': 0.5,
            'spread': 0.5,
            'volatility': 0.5,
            'price_position': 0.5,
            'candle_strength': 0.5,
            'stop_speed': 0.4,
            'book_imbalance': 0.4,
            'book_delta': 0.4,
        }
        
        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_history: List[Dict] = []
        
        # Загружаем модель
        self.load_model()
        logger.info(f"Disco57 инициализирован. Win Rate: {self.get_win_rate():.1f}%")
    
    def load_model(self):
        """Загрузить модель из файла"""
        try:
            path = Path(self.model_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.weights = data.get('weights', self.weights)
                    self.total_trades = data.get('total_trades', 0)
                    self.winning_trades = data.get('winning_trades', 0)
                    self.total_pnl = data.get('total_pnl', 0.0)
                    self.trade_history = data.get('trade_history', [])[-100:]  # Последние 100
                logger.info(f"Disco57 модель загружена: {self.total_trades} сделок")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель Disco57: {e}")
    
    def save_model(self):
        """Сохранить модель в файл"""
        try:
            path = Path(self.model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'weights': self.weights,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'total_pnl': self.total_pnl,
                'trade_history': self.trade_history[-100:],
                'updated_at': time.time()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Disco57 модель сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели Disco57: {e}")
    
    def extract_features(self, candles: List, ticker: Dict) -> TradeFeatures:
        """Извлечь признаки из данных"""
        closes = [float(c[4]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        volumes = [float(c[5]) for c in candles]
        
        price = closes[-1]
        
        # EMA
        ema9 = self._calc_ema(closes, 9)
        ema21 = self._calc_ema(closes, 21)
        ema_trend = 1.0 if ema9 > ema21 else -1.0
        
        # Momentum
        momentum = (closes[-1] - closes[-14]) / closes[-14] if len(closes) >= 14 else 0
        
        # Volume
        avg_vol = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 1
        volume_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        
        # RSI
        rsi = self._calc_rsi(closes, 14) / 100.0  # Нормализуем 0-1
        
        # Spread
        bid = float(ticker.get('bid', price))
        ask = float(ticker.get('ask', price))
        spread = (ask - bid) / price if price > 0 else 0
        
        # Volatility (ATR)
        atr = self._calc_atr(highs, lows, closes, 14)
        volatility = atr / price if price > 0 else 0
        
        # Price position in range
        high_20 = max(highs[-20:]) if len(highs) >= 20 else highs[-1]
        low_20 = min(lows[-20:]) if len(lows) >= 20 else lows[-1]
        price_range = high_20 - low_20
        price_position = (price - low_20) / price_range if price_range > 0 else 0.5
        
        # Candle strength
        open_price = float(candles[-1][1])
        candle_body = abs(price - open_price)
        candle_range = highs[-1] - lows[-1]
        candle_strength = candle_body / candle_range if candle_range > 0 else 0
        
        return TradeFeatures(
            ema_trend=ema_trend,
            momentum=momentum,
            volume_ratio=min(volume_ratio, 3.0) / 3.0,  # Нормализуем 0-1
            rsi=rsi,
            spread=min(spread * 1000, 1.0),  # Нормализуем
            volatility=min(volatility * 100, 1.0),  # Нормализуем
            price_position=price_position,
            candle_strength=candle_strength,
            stop_speed=0.0,
            book_imbalance=0.5,
            book_delta=0.5
        )
    
    def predict(self, features: TradeFeatures, direction: str) -> tuple:
        """
        Предсказать вероятность успеха сделки
        
        Returns:
            (allow: bool, confidence: float)
        """
        # Вычисляем взвешенную сумму признаков
        score = 0.0
        
        # Для LONG
        if direction == 'long':
            score += self.weights['ema_trend'] * (1 if features.ema_trend > 0 else 0)
            score += self.weights['momentum'] * (1 if features.momentum > 0.005 else 0)
            score += self.weights['volume_ratio'] * features.volume_ratio
            score += self.weights['rsi'] * (1 if 0.3 < features.rsi < 0.7 else 0)
            score += self.weights['spread'] * (1 - features.spread)  # Меньше спред = лучше
            score += self.weights['volatility'] * features.volatility
            score += self.weights['price_position'] * (1 - features.price_position)  # Покупаем внизу
            score += self.weights['candle_strength'] * features.candle_strength
            score += self.weights['book_imbalance'] * features.book_imbalance
            score += self.weights['book_delta'] * features.book_delta
        
        # Для SHORT
        else:
            score += self.weights['ema_trend'] * (1 if features.ema_trend < 0 else 0)
            score += self.weights['momentum'] * (1 if features.momentum < -0.005 else 0)
            score += self.weights['volume_ratio'] * features.volume_ratio
            score += self.weights['rsi'] * (1 if 0.3 < features.rsi < 0.7 else 0)
            score += self.weights['spread'] * (1 - features.spread)
            score += self.weights['volatility'] * features.volatility
            score += self.weights['price_position'] * features.price_position  # Продаем вверху
            score += self.weights['candle_strength'] * features.candle_strength
            score += self.weights['book_imbalance'] * (1 - features.book_imbalance)
            score += self.weights['book_delta'] * (1 - features.book_delta)
        
        # Stop speed (0 быстрый стоп, 1 медленный) - одинаково для обоих направлений
        stop_component = 1 - max(0.0, min(features.stop_speed, 1.0))
        score += self.weights['stop_speed'] * stop_component
        
        # Нормализуем в диапазон 0-1
        max_score = sum(self.weights.values())
        confidence = score / max_score if max_score > 0 else 0.5
        
        # Решение
        allow = confidence >= self.min_confidence
        
        return allow, confidence
    
    def learn(self, features: TradeFeatures, direction: str, pnl: float):
        """
        Обучение на результате сделки
        
        Args:
            features: Признаки при входе
            direction: 'long' или 'short'
            pnl: Результат в USD
        """
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            reward = 1.0
        else:
            reward = -1.0
        
        # Обновляем веса
        feature_dict = asdict(features)
        
        for key in self.weights:
            if key in feature_dict:
                feature_value = feature_dict[key]
                
                # Для LONG
                if direction == 'long':
                    if key == 'ema_trend':
                        feature_value = 1 if feature_value > 0 else 0
                    elif key == 'momentum':
                        feature_value = 1 if feature_value > 0.005 else 0
                    elif key == 'price_position':
                        feature_value = 1 - feature_value
                    elif key == 'book_imbalance':
                        feature_value = feature_value
                    elif key == 'book_delta':
                        feature_value = feature_value
                
                # Для SHORT
                else:
                    if key == 'ema_trend':
                        feature_value = 1 if feature_value < 0 else 0
                    elif key == 'momentum':
                        feature_value = 1 if feature_value < -0.005 else 0
                    elif key == 'book_imbalance':
                        feature_value = 1 - feature_value
                    elif key == 'book_delta':
                        feature_value = 1 - feature_value
                
                if key == 'stop_speed':
                    feature_value = 1 - max(0.0, min(feature_value, 1.0))
                
                # Обновление веса
                adjustment = self.learning_rate * reward * feature_value
                self.weights[key] = max(0.1, min(1.0, self.weights[key] + adjustment))
        
        # Сохраняем историю
        self.trade_history.append({
            'time': time.time(),
            'direction': direction,
            'pnl': pnl,
            'features': asdict(features)
        })
        
        # Сохраняем модель
        self.save_model()
        
        logger.info(f"Disco57 обучен: PnL ${pnl:.2f} | Win Rate: {self.get_win_rate():.1f}%")
    
    def get_win_rate(self) -> float:
        """Получить текущий Win Rate"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def get_stats(self) -> Dict:
        """Получить статистику"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.get_win_rate(),
            'total_pnl': self.total_pnl,
            'weights': self.weights
        }
    
    def _calc_ema(self, data: List[float], period: int) -> float:
        if len(data) < period:
            return sum(data) / len(data) if data else 0
        mult = 2 / (period + 1)
        ema = sum(data[:period]) / period
        for price in data[period:]:
            ema = (price - ema) * mult + ema
        return ema
    
    def _calc_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calc_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0
        return sum(trs[-period:]) / period


# Тест
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    disco = Disco57Learner(model_path='./test_disco57.json')
    print(f"Stats: {disco.get_stats()}")
