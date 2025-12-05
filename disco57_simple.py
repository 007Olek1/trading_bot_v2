#!/usr/bin/env python3
"""
Disco57 Simple - Упрощенный ML для TradeGPT Scalper
Только бинарное решение: ALLOW или BLOCK
Обучение на исторических данных
"""

import logging
import pickle
import os
from typing import Dict, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Результат сделки для обучения"""
    price: float
    volume_ratio: float
    momentum: float
    volatility: float
    success: bool  # True если прибыльная


class Disco57Simple:
    """
    Упрощенный DiscoRL - только ALLOW/BLOCK
    Без confidence, без промежуточных оценок
    """
    
    def __init__(self, model_path: str = 'disco57_model.pkl'):
        self.model_path = model_path
        self.learning_rate = 0.05
        
        # Простые веса для признаков
        self.weights = {
            'volume_high': 0.5,      # Объем выше среднего
            'momentum_positive': 0.5,  # Положительный импульс
            'volatility_ok': 0.5,    # Волатильность в норме
        }
        
        # Статистика
        self.total_trades = 0
        self.winning_trades = 0
        self.history: List[TradeResult] = []
        
        # Загружаем модель если есть
        self.load_model()
        
        logger.info(f"Disco57Simple инициализирован | Сделок: {self.total_trades} | "
                   f"WR: {self.win_rate:.1f}%")
    
    @property
    def win_rate(self) -> float:
        """Винрейт"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def load_model(self):
        """Загрузить модель"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.weights = data.get('weights', self.weights)
                    self.total_trades = data.get('total_trades', 0)
                    self.winning_trades = data.get('winning_trades', 0)
                logger.info(f"Модель загружена: {self.total_trades} сделок")
            except Exception as e:
                logger.warning(f"Ошибка загрузки модели: {e}")
    
    def save_model(self):
        """Сохранить модель"""
        try:
            data = {
                'weights': self.weights,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug("Модель сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
    
    def extract_features(self, price: float, volume_ratio: float, 
                        momentum: float, volatility: float) -> Dict[str, float]:
        """Извлечь признаки"""
        features = {}
        
        # Объем выше среднего
        features['volume_high'] = 1.0 if volume_ratio > 1.0 else 0.0
        
        # Положительный импульс (для long) или отрицательный (для short)
        features['momentum_positive'] = 1.0 if abs(momentum) > 0 else 0.0
        
        # Волатильность в норме (не слишком высокая)
        features['volatility_ok'] = 1.0 if 0.001 < volatility < 0.05 else 0.0
        
        return features
    
    def predict(self, price: float, volume_ratio: float, 
                momentum: float, volatility: float) -> str:
        """
        Предсказание: ALLOW или BLOCK
        
        Returns:
            'ALLOW' - разрешить вход
            'BLOCK' - заблокировать вход
        """
        features = self.extract_features(price, volume_ratio, momentum, volatility)
        
        # Взвешенная сумма
        score = 0.0
        for feature_name, feature_value in features.items():
            weight = self.weights.get(feature_name, 0.5)
            score += weight * feature_value
        
        # Нормализуем (максимум 3 признака)
        score = score / 3.0
        
        # Порог 0.5 - если выше, разрешаем
        if score >= 0.5:
            return 'ALLOW'
        else:
            return 'BLOCK'
    
    def learn(self, price: float, volume_ratio: float, momentum: float, 
              volatility: float, success: bool):
        """
        Обучение на результате сделки
        
        Args:
            price: Цена входа
            volume_ratio: Отношение объема к среднему
            momentum: Импульс
            volatility: Волатильность
            success: True если сделка прибыльная
        """
        features = self.extract_features(price, volume_ratio, momentum, volatility)
        
        # Reward: +1 для прибыльной, -1 для убыточной
        reward = 1.0 if success else -1.0
        
        # Обновляем веса активных признаков
        for feature_name, feature_value in features.items():
            if feature_value > 0:
                old_weight = self.weights.get(feature_name, 0.5)
                new_weight = old_weight + self.learning_rate * reward * feature_value
                # Ограничиваем [0.1, 0.9]
                self.weights[feature_name] = max(0.1, min(0.9, new_weight))
        
        # Обновляем статистику
        self.total_trades += 1
        if success:
            self.winning_trades += 1
        
        # Сохраняем результат
        result = TradeResult(
            price=price,
            volume_ratio=volume_ratio,
            momentum=momentum,
            volatility=volatility,
            success=success
        )
        self.history.append(result)
        
        # Сохраняем модель каждые 10 сделок
        if self.total_trades % 10 == 0:
            self.save_model()
            logger.info(f"Disco57: {self.total_trades} сделок | WR: {self.win_rate:.1f}%")
    
    def train_on_history(self, trades: List[Dict]):
        """
        Обучение на исторических данных
        
        Args:
            trades: Список словарей с ключами:
                - price
                - volume_ratio
                - momentum
                - volatility
                - pnl (для определения success)
        """
        logger.info(f"Обучение на {len(trades)} исторических сделках...")
        
        for trade in trades:
            try:
                self.learn(
                    price=float(trade.get('price', 0)),
                    volume_ratio=float(trade.get('volume_ratio', 1.0)),
                    momentum=float(trade.get('momentum', 0)),
                    volatility=float(trade.get('volatility', 0.01)),
                    success=float(trade.get('pnl', 0)) > 0
                )
            except Exception as e:
                logger.debug(f"Ошибка обработки сделки: {e}")
        
        self.save_model()
        logger.info(f"Обучение завершено | Всего: {self.total_trades} | WR: {self.win_rate:.1f}%")
        logger.info(f"Веса: {self.weights}")
    
    def get_stats(self) -> Dict:
        """Получить статистику"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'weights': self.weights
        }


if __name__ == '__main__':
    # Тест
    logging.basicConfig(level=logging.INFO)
    
    disco = Disco57Simple()
    
    # Тестовые данные
    test_trades = [
        {'price': 100, 'volume_ratio': 1.5, 'momentum': 2.0, 'volatility': 0.02, 'pnl': 0.6},
        {'price': 101, 'volume_ratio': 1.2, 'momentum': 1.5, 'volatility': 0.015, 'pnl': 0.4},
        {'price': 102, 'volume_ratio': 0.8, 'momentum': -0.5, 'volatility': 0.03, 'pnl': -0.2},
        {'price': 103, 'volume_ratio': 1.8, 'momentum': 3.0, 'volatility': 0.025, 'pnl': 0.8},
        {'price': 104, 'volume_ratio': 0.6, 'momentum': -1.0, 'volatility': 0.01, 'pnl': -0.3},
    ]
    
    disco.train_on_history(test_trades)
    
    # Тест предсказания
    print("\n=== ТЕСТ ПРЕДСКАЗАНИЙ ===")
    
    test_cases = [
        {'volume_ratio': 1.5, 'momentum': 2.0, 'volatility': 0.02},
        {'volume_ratio': 0.8, 'momentum': -0.5, 'volatility': 0.03},
        {'volume_ratio': 1.2, 'momentum': 1.0, 'volatility': 0.015},
    ]
    
    for i, case in enumerate(test_cases, 1):
        decision = disco.predict(
            price=100,
            volume_ratio=case['volume_ratio'],
            momentum=case['momentum'],
            volatility=case['volatility']
        )
        print(f"Тест {i}: {case} -> {decision}")
