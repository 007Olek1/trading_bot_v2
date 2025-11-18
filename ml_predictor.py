"""
🤖 ML МОДЕЛЬ ДЛЯ ПРЕДСКАЗАНИЙ
Машинное обучение для улучшения точности сигналов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json


class MLPredictor:
    """ML модель для предсказания успешности сделок"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Путь к модели
        self.model_path = Path(__file__).parent / "models" / "ml_model.json"
        self.model_path.parent.mkdir(exist_ok=True)
        
        # Веса признаков (обучаются на исторических данных)
        self.feature_weights = {
            'confidence': 0.35,
            'timeframes_aligned': 0.25,
            'volume_ratio': 0.15,
            'trend_strength': 0.15,
            'volatility': 0.10,
        }
        
        # Статистика для нормализации
        self.feature_stats = {
            'confidence': {'mean': 0.70, 'std': 0.10},
            'timeframes_aligned': {'mean': 3.5, 'std': 0.8},
            'volume_ratio': {'mean': 1.5, 'std': 0.5},
            'trend_strength': {'mean': 0.02, 'std': 0.01},
            'volatility': {'mean': 0.03, 'std': 0.02},
        }
        
        # История для обучения
        self.training_data = []
        
        # Загружаем модель если есть
        self.load_model()
    
    def extract_features(self, signal_data: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Извлекает признаки для ML модели
        
        Args:
            signal_data: Данные сигнала
            market_data: Рыночные данные
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        # 1. Уверенность стратегий
        features['confidence'] = signal_data.get('confidence', 0.65)
        
        # 2. Количество совпадающих таймфреймов
        features['timeframes_aligned'] = signal_data.get('timeframes_aligned', 3)
        
        # 3. Отношение объёма к среднему
        if 'volume_ratio' in signal_data:
            features['volume_ratio'] = signal_data['volume_ratio']
        else:
            features['volume_ratio'] = 1.0
        
        # 4. Сила тренда
        if len(market_data) >= 50:
            close = market_data['close']
            ema20 = close.ewm(span=20).mean().iloc[-1]
            ema50 = close.ewm(span=50).mean().iloc[-1]
            features['trend_strength'] = abs(ema20 - ema50) / ema50
        else:
            features['trend_strength'] = 0.01
        
        # 5. Волатильность
        if len(market_data) >= 14:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(window=14).mean().iloc[-1]
            features['volatility'] = atr / close.iloc[-1]
        else:
            features['volatility'] = 0.02
        
        return features
    
    def normalize_features(self, features: Dict) -> Dict:
        """
        Нормализует признаки (z-score)
        
        Args:
            features: Исходные признаки
            
        Returns:
            Нормализованные признаки
        """
        normalized = {}
        
        for key, value in features.items():
            if key in self.feature_stats:
                mean = self.feature_stats[key]['mean']
                std = self.feature_stats[key]['std']
                normalized[key] = (value - mean) / std
            else:
                normalized[key] = value
        
        return normalized
    
    def predict_success_probability(self, features: Dict) -> float:
        """
        Предсказывает вероятность успеха сделки
        
        Args:
            features: Признаки сигнала
            
        Returns:
            Вероятность успеха (0-1)
        """
        # Нормализуем признаки
        norm_features = self.normalize_features(features)
        
        # Взвешенная сумма
        score = 0.0
        for key, weight in self.feature_weights.items():
            if key in norm_features:
                # Сигмоида для нормализации в [0, 1]
                feature_score = 1 / (1 + np.exp(-norm_features[key]))
                score += feature_score * weight
        
        # Клиппинг в [0, 1]
        probability = np.clip(score, 0.0, 1.0)
        
        return probability
    
    def enhance_signal(self, signal_data: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Улучшает сигнал с помощью ML предсказания
        
        Args:
            signal_data: Исходный сигнал
            market_data: Рыночные данные
            
        Returns:
            Улучшенный сигнал с ML оценкой
        """
        # Извлекаем признаки
        features = self.extract_features(signal_data, market_data)
        
        # Предсказываем вероятность успеха
        ml_probability = self.predict_success_probability(features)
        
        # Комбинируем с исходной уверенностью
        original_confidence = signal_data.get('confidence', 0.65)
        
        # Взвешенное среднее (70% ML, 30% стратегии)
        enhanced_confidence = (ml_probability * 0.70) + (original_confidence * 0.30)
        
        # Обновляем сигнал
        enhanced_signal = signal_data.copy()
        enhanced_signal['ml_probability'] = ml_probability
        enhanced_signal['original_confidence'] = original_confidence
        enhanced_signal['confidence'] = enhanced_confidence
        enhanced_signal['ml_features'] = features
        
        self.logger.info(
            f"🤖 ML Enhancement: "
            f"Original: {original_confidence:.1%} | "
            f"ML: {ml_probability:.1%} | "
            f"Enhanced: {enhanced_confidence:.1%}"
        )
        
        return enhanced_signal
    
    def add_training_sample(self, features: Dict, success: bool):
        """
        Добавляет образец для обучения
        
        Args:
            features: Признаки сделки
            success: Успешна ли была сделка
        """
        sample = {
            'features': features,
            'success': success,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.training_data.append(sample)
        
        # Сохраняем каждые 10 образцов
        if len(self.training_data) % 10 == 0:
            self.save_training_data()
    
    def update_model(self):
        """
        Обновляет веса модели на основе накопленных данных
        """
        if len(self.training_data) < 20:
            self.logger.warning("Недостаточно данных для обучения (минимум 20)")
            return
        
        # Простое обновление весов на основе корреляции с успехом
        df = pd.DataFrame([
            {**sample['features'], 'success': sample['success']}
            for sample in self.training_data
        ])
        
        # Рассчитываем корреляцию каждого признака с успехом
        correlations = {}
        for feature in self.feature_weights.keys():
            if feature in df.columns:
                corr = df[feature].corr(df['success'])
                correlations[feature] = abs(corr) if not np.isnan(corr) else 0
        
        # Нормализуем корреляции в веса
        total_corr = sum(correlations.values())
        if total_corr > 0:
            for feature in self.feature_weights.keys():
                if feature in correlations:
                    self.feature_weights[feature] = correlations[feature] / total_corr
        
        # Обновляем статистику признаков
        for feature in self.feature_weights.keys():
            if feature in df.columns:
                self.feature_stats[feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std() or 0.01
                }
        
        self.logger.info(f"🎓 Модель обновлена на {len(self.training_data)} образцах")
        self.save_model()
    
    def save_model(self):
        """Сохраняет модель в файл"""
        model_data = {
            'feature_weights': self.feature_weights,
            'feature_stats': self.feature_stats,
            'training_samples': len(self.training_data),
            'last_updated': pd.Timestamp.now().isoformat()
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"💾 Модель сохранена: {self.model_path}")
    
    def load_model(self):
        """Загружает модель из файла"""
        if not self.model_path.exists():
            self.logger.info("📝 Модель не найдена, используются значения по умолчанию")
            return
        
        try:
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
            
            self.feature_weights = model_data.get('feature_weights', self.feature_weights)
            self.feature_stats = model_data.get('feature_stats', self.feature_stats)
            
            self.logger.info(
                f"✅ Модель загружена: {model_data.get('training_samples', 0)} образцов"
            )
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
    
    def save_training_data(self):
        """Сохраняет обучающие данные"""
        data_path = self.model_path.parent / "training_data.json"
        
        with open(data_path, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        
        self.logger.info(f"💾 Обучающие данные сохранены: {len(self.training_data)} образцов")


if __name__ == "__main__":
    # Тест
    print("🧪 Тестирование ML предиктора\n")
    
    predictor = MLPredictor()
    
    # Тестовые данные
    signal = {
        'confidence': 0.75,
        'timeframes_aligned': 4,
        'volume_ratio': 1.8,
    }
    
    market_data = pd.DataFrame({
        'high': 100 + np.random.randn(100),
        'low': 99 + np.random.randn(100),
        'close': 99.5 + np.random.randn(100),
    })
    
    # Улучшаем сигнал
    enhanced = predictor.enhance_signal(signal, market_data)
    
    print(f"Исходная уверенность: {signal['confidence']:.1%}")
    print(f"ML вероятность: {enhanced['ml_probability']:.1%}")
    print(f"Улучшенная уверенность: {enhanced['confidence']:.1%}")
    print()
    
    # Добавляем обучающие образцы
    print("Добавление обучающих образцов...")
    for i in range(25):
        features = predictor.extract_features(signal, market_data)
        success = np.random.random() > 0.4  # 60% успешных
        predictor.add_training_sample(features, success)
    
    # Обновляем модель
    print("\nОбновление модели...")
    predictor.update_model()
    
    print("\n✅ Тест завершён!")
