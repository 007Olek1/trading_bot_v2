#!/usr/bin/env python3
"""
🧠 ПРОДВИНУТАЯ СИСТЕМА МАШИННОГО ОБУЧЕНИЯ
==========================================

Функции:
- Обучение на исторических данных
- Предсказание рыночных трендов
- Автоматическая оптимизация стратегий
- Ансамбль моделей для максимальной точности
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
import joblib
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """📊 Производительность модели"""
    model_name: str
    mse: float
    mae: float
    r2: float
    accuracy: float
    last_trained: datetime
    training_samples: int

@dataclass
class PredictionResult:
    """🔮 Результат предсказания"""
    symbol: str
    predicted_price: float
    confidence: float
    trend_direction: str  # 'bullish', 'bearish', 'sideways'
    time_horizon: str    # 'short', 'medium', 'long'
    model_used: str
    features_importance: Dict[str, float]

class AdvancedMLSystem:
    """🧠 Продвинутая система машинного обучения"""
    
    def __init__(self):
        # Модели для разных задач
        self.price_prediction_models = {}
        self.trend_prediction_models = {}
        self.strategy_optimization_models = {}
        
        # Предобработчики данных
        self.scalers = {}
        self.feature_importance = {}
        
        # Настройки
        self.settings = {
            'lookback_periods': [5, 10, 20, 50],  # Периоды для анализа
            'prediction_horizons': [1, 3, 6, 12],  # Горизонты предсказания (часы)
            'min_training_samples': 1000,
            'validation_split': 0.2,
            'retrain_frequency_hours': 24,
            'ensemble_weights': {
                'random_forest': 0.3,
                'gradient_boosting': 0.3,
                'neural_network': 0.2,
                'lstm': 0.2
            }
        }
        
        # Статистика
        self.stats = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'models_trained': 0,
            'last_training': None,
            'best_model': None,
            'avg_accuracy': 0.0
        }
        
        # Отключаем CUDA для TensorFlow
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        logger.info("🧠 Продвинутая система ML инициализирована")
    
    def create_features(self, ohlcv_data: pd.DataFrame, technical_indicators: Dict) -> pd.DataFrame:
        """🔧 Создание признаков для ML"""
        features = pd.DataFrame()
        
        # Базовые ценовые признаки
        features['price_change_1h'] = ohlcv_data['close'].pct_change(1)
        features['price_change_4h'] = ohlcv_data['close'].pct_change(4)
        features['price_change_24h'] = ohlcv_data['close'].pct_change(24)
        
        # Волатильность
        features['volatility_1h'] = ohlcv_data['close'].rolling(4).std()
        features['volatility_4h'] = ohlcv_data['close'].rolling(16).std()
        features['volatility_24h'] = ohlcv_data['close'].rolling(96).std()
        
        # Объёмные признаки
        features['volume_change_1h'] = ohlcv_data['volume'].pct_change(1)
        features['volume_ma_ratio'] = ohlcv_data['volume'] / ohlcv_data['volume'].rolling(20).mean()
        features['volume_spike'] = (ohlcv_data['volume'] > ohlcv_data['volume'].rolling(20).mean() * 2).astype(int)
        
        # Технические индикаторы
        if 'rsi' in technical_indicators:
            features['rsi'] = technical_indicators['rsi']
            features['rsi_overbought'] = (technical_indicators['rsi'] > 70).astype(int)
            features['rsi_oversold'] = (technical_indicators['rsi'] < 30).astype(int)
        
        if 'bb_upper' in technical_indicators and 'bb_lower' in technical_indicators:
            bb_position = (ohlcv_data['close'] - technical_indicators['bb_lower']) / (technical_indicators['bb_upper'] - technical_indicators['bb_lower'])
            features['bb_position'] = bb_position
            features['bb_squeeze'] = ((technical_indicators['bb_upper'] - technical_indicators['bb_lower']) / ohlcv_data['close'] < 0.1).astype(int)
        
        if 'ema_21' in technical_indicators and 'ema_50' in technical_indicators:
            features['ema_trend'] = (technical_indicators['ema_21'] > technical_indicators['ema_50']).astype(int)
            features['ema_distance'] = (ohlcv_data['close'] - technical_indicators['ema_21']) / technical_indicators['ema_21']
        
        # MACD
        if 'macd' in technical_indicators and 'macd_signal' in technical_indicators:
            macd_values = technical_indicators['macd']
            macd_signal_values = technical_indicators['macd_signal']
            
            # Преобразуем в pandas Series если это numpy array
            if isinstance(macd_values, np.ndarray):
                macd_series = pd.Series(macd_values, index=ohlcv_data.index)
                macd_signal_series = pd.Series(macd_signal_values, index=ohlcv_data.index)
            else:
                macd_series = macd_values
                macd_signal_series = macd_signal_values
            
            features['macd_diff'] = macd_series - macd_signal_series
            features['macd_cross'] = ((macd_series > macd_signal_series) & 
                                     (macd_series.shift(1) <= macd_signal_series.shift(1))).astype(int)
        
        # Временные признаки
        features['hour_of_day'] = pd.to_datetime(ohlcv_data.index).hour
        features['day_of_week'] = pd.to_datetime(ohlcv_data.index).dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # Лаговые признаки
        for lag in [1, 2, 3, 5, 10]:
            features[f'price_lag_{lag}'] = ohlcv_data['close'].shift(lag)
            features[f'volume_lag_{lag}'] = ohlcv_data['volume'].shift(lag)
        
        # Скользящие средние разных периодов
        for period in [5, 10, 20, 50]:
            sma = ohlcv_data['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_vs_sma_{period}'] = ohlcv_data['close'] / sma - 1
        
        # Удаляем NaN значения
        features = features.dropna()
        
        return features
    
    def create_targets(self, ohlcv_data: pd.DataFrame, prediction_horizons: List[int]) -> Dict[str, pd.Series]:
        """🎯 Создание целевых переменных"""
        targets = {}
        
        for horizon in prediction_horizons:
            # Предсказание цены
            targets[f'price_target_{horizon}h'] = ohlcv_data['close'].shift(-horizon)
            
            # Предсказание направления тренда
            price_change = ohlcv_data['close'].shift(-horizon) / ohlcv_data['close'] - 1
            targets[f'trend_target_{horizon}h'] = (price_change > 0.01).astype(int)  # 1% порог
            
            # Предсказание волатильности
            targets[f'volatility_target_{horizon}h'] = ohlcv_data['close'].rolling(horizon).std().shift(-horizon)
        
        return targets
    
    def train_price_prediction_model(self, symbol: str, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """🎯 Обучение модели предсказания цен"""
        logger.info(f"🧠 Обучение модели предсказания цен для {symbol}")
        
        models = {}
        performances = {}
        
        # Подготавливаем данные
        X = features.values
        y_price = targets['price_target_1h'].values
        
        # Проверяем совместимость размеров
        min_length = min(len(X), len(y_price))
        X = X[:min_length]
        y_price = y_price[:min_length]
        
        # Разделяем на обучающую и тестовую выборки
        split_idx = int(len(X) * (1 - self.settings['validation_split']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_price[:split_idx], y_price[split_idx:]
        
        # Масштабирование
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[f'{symbol}_price'] = scaler
        
        # 1. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        models['random_forest'] = rf_model
        performances['random_forest'] = ModelPerformance(
            model_name='Random Forest',
            mse=mean_squared_error(y_test, rf_pred),
            mae=mean_absolute_error(y_test, rf_pred),
            r2=r2_score(y_test, rf_pred),
            accuracy=self._calculate_direction_accuracy(y_test, rf_pred),
            last_trained=datetime.now(),
            training_samples=len(X_train)
        )
        
        # 2. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        models['gradient_boosting'] = gb_model
        performances['gradient_boosting'] = ModelPerformance(
            model_name='Gradient Boosting',
            mse=mean_squared_error(y_test, gb_pred),
            mae=mean_absolute_error(y_test, gb_pred),
            r2=r2_score(y_test, gb_pred),
            accuracy=self._calculate_direction_accuracy(y_test, gb_pred),
            last_trained=datetime.now(),
            training_samples=len(X_train)
        )
        
        # 3. Neural Network
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        nn_pred = nn_model.predict(X_test_scaled)
        
        models['neural_network'] = nn_model
        performances['neural_network'] = ModelPerformance(
            model_name='Neural Network',
            mse=mean_squared_error(y_test, nn_pred),
            mae=mean_absolute_error(y_test, nn_pred),
            r2=r2_score(y_test, nn_pred),
            accuracy=self._calculate_direction_accuracy(y_test, nn_pred),
            last_trained=datetime.now(),
            training_samples=len(X_train)
        )
        
        # 4. LSTM (для временных рядов)
        lstm_model = self._create_lstm_model(X_train_scaled.shape[1])
        lstm_model.fit(
            X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1]),
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=0
        )
        
        lstm_pred = lstm_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])).flatten()
        
        models['lstm'] = lstm_model
        performances['lstm'] = ModelPerformance(
            model_name='LSTM',
            mse=mean_squared_error(y_test, lstm_pred),
            mae=mean_absolute_error(y_test, lstm_pred),
            r2=r2_score(y_test, lstm_pred),
            accuracy=self._calculate_direction_accuracy(y_test, lstm_pred),
            last_trained=datetime.now(),
            training_samples=len(X_train)
        )
        
        # Создаём ансамбль
        ensemble_model = VotingRegressor([
            ('rf', rf_model),
            ('gb', gb_model),
            ('nn', nn_model)
        ])
        ensemble_model.fit(X_train_scaled, y_train)
        ensemble_pred = ensemble_model.predict(X_test_scaled)
        
        models['ensemble'] = ensemble_model
        performances['ensemble'] = ModelPerformance(
            model_name='Ensemble',
            mse=mean_squared_error(y_test, ensemble_pred),
            mae=mean_absolute_error(y_test, ensemble_pred),
            r2=r2_score(y_test, ensemble_pred),
            accuracy=self._calculate_direction_accuracy(y_test, ensemble_pred),
            last_trained=datetime.now(),
            training_samples=len(X_train)
        )
        
        # Сохраняем модели
        self.price_prediction_models[symbol] = models
        
        # Обновляем статистику
        self.stats['models_trained'] += len(models)
        self.stats['last_training'] = datetime.now()
        
        # Определяем лучшую модель
        best_model_name = max(performances.keys(), key=lambda x: performances[x].accuracy)
        self.stats['best_model'] = best_model_name
        
        logger.info(f"✅ Обучение завершено. Лучшая модель: {best_model_name} (точность: {performances[best_model_name].accuracy:.2%})")
        
        return {
            'models': models,
            'performances': performances,
            'best_model': best_model_name,
            'feature_names': features.columns.tolist()
        }
    
    def _create_lstm_model(self, input_dim: int) -> tf.keras.Model:
        """🏗️ Создание LSTM модели"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, input_dim)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """📊 Расчёт точности направления"""
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        if len(true_direction) == 0:
            return 0.0
        
        return np.mean(true_direction == pred_direction)
    
    def predict_price(self, symbol: str, features: pd.DataFrame) -> PredictionResult:
        """🔮 Предсказание цены"""
        if symbol not in self.price_prediction_models:
            logger.warning(f"⚠️ Модель для {symbol} не обучена")
            return None
        
        models = self.price_prediction_models[symbol]
        scaler = self.scalers.get(f'{symbol}_price')
        
        if scaler is None:
            logger.error(f"❌ Scaler для {symbol} не найден")
            return None
        
        # Подготавливаем данные
        X = features.iloc[-1:].values  # Последняя строка
        X_scaled = scaler.transform(X)
        
        # Предсказания от всех моделей
        predictions = {}
        
        for model_name, model in models.items():
            if model_name == 'lstm':
                pred = model.predict(X_scaled.reshape(1, 1, X_scaled.shape[1])).flatten()[0]
            else:
                pred = model.predict(X_scaled)[0]
            
            predictions[model_name] = pred
        
        # Взвешенное среднее предсказаний
        weights = self.settings['ensemble_weights']
        weighted_prediction = sum(predictions[model] * weights.get(model, 0.25) for model in predictions)
        
        # Рассчитываем уверенность на основе согласованности моделей
        predictions_array = np.array(list(predictions.values()))
        confidence = 1.0 - (np.std(predictions_array) / np.mean(predictions_array)) if np.mean(predictions_array) != 0 else 0.5
        
        # Определяем направление тренда
        current_price = features['price_change_1h'].iloc[-1] if 'price_change_1h' in features.columns else 0
        trend_direction = 'bullish' if weighted_prediction > current_price * 1.01 else 'bearish' if weighted_prediction < current_price * 0.99 else 'sideways'
        
        # Обновляем статистику
        self.stats['total_predictions'] += 1
        
        return PredictionResult(
            symbol=symbol,
            predicted_price=weighted_prediction,
            confidence=min(max(confidence, 0.0), 1.0),
            trend_direction=trend_direction,
            time_horizon='short',
            model_used='ensemble',
            features_importance=self._get_feature_importance(symbol, features)
        )
    
    def _get_feature_importance(self, symbol: str, features: pd.DataFrame) -> Dict[str, float]:
        """📊 Важность признаков"""
        if symbol not in self.price_prediction_models:
            return {}
        
        models = self.price_prediction_models[symbol]
        
        # Используем Random Forest для получения важности признаков
        if 'random_forest' in models:
            rf_model = models['random_forest']
            importance = rf_model.feature_importances_
            feature_names = features.columns.tolist()
            
            return dict(zip(feature_names, importance))
        
        return {}
    
    def optimize_strategy_parameters(self, historical_trades: List[Dict]) -> Dict[str, Any]:
        """⚙️ Оптимизация параметров стратегии"""
        logger.info("⚙️ Оптимизация параметров стратегии")
        
        if len(historical_trades) < self.settings['min_training_samples']:
            logger.warning(f"⚠️ Недостаточно данных для оптимизации: {len(historical_trades)} < {self.settings['min_training_samples']}")
            return {}
        
        # Создаём DataFrame из исторических сделок
        df = pd.DataFrame(historical_trades)
        
        # Создаём признаки для оптимизации
        features = self._create_strategy_features(df)
        
        # Целевая переменная - прибыльность
        target = df['pnl_percent'].values
        
        # Обучение модели для оптимизации
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scores = cross_val_score(model, features, target, cv=5, scoring='neg_mean_squared_error')
        
        model.fit(features, target)
        
        # Получаем важность параметров
        feature_importance = dict(zip(features.columns, model.feature_importances_))
        
        # Рекомендации по оптимизации
        recommendations = {}
        
        for param, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            if importance > 0.1:  # Только значимые параметры
                recommendations[param] = {
                    'importance': importance,
                    'current_value': df[param].mean() if param in df.columns else 'unknown',
                    'suggestion': 'optimize' if importance > 0.2 else 'monitor'
                }
        
        logger.info(f"✅ Оптимизация завершена. Найдено {len(recommendations)} значимых параметров")
        
        return {
            'model_score': scores.mean(),
            'feature_importance': feature_importance,
            'recommendations': recommendations,
            'optimization_date': datetime.now().isoformat()
        }
    
    def _create_strategy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Создание признаков для оптимизации стратегии"""
        features = pd.DataFrame()
        
        # Параметры стратегии
        if 'confidence' in df.columns:
            features['avg_confidence'] = df['confidence']
            features['confidence_std'] = df['confidence'].rolling(10).std()
        
        if 'volume_ratio' in df.columns:
            features['avg_volume_ratio'] = df['volume_ratio']
            features['volume_ratio_std'] = df['volume_ratio'].rolling(10).std()
        
        # Временные признаки
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour_of_day'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # Рыночные условия
        if 'market_condition' in df.columns:
            features['market_bullish'] = (df['market_condition'] == 'bullish').astype(int)
            features['market_bearish'] = (df['market_condition'] == 'bearish').astype(int)
        
        # Результаты торгов
        features['win_rate'] = (df['pnl_percent'] > 0).rolling(20).mean()
        features['avg_pnl'] = df['pnl_percent'].rolling(20).mean()
        features['pnl_volatility'] = df['pnl_percent'].rolling(20).std()
        
        return features.fillna(0)
    
    def get_ml_statistics(self) -> Dict:
        """📊 Статистика ML системы"""
        return {
            'total_predictions': self.stats['total_predictions'],
            'models_trained': self.stats['models_trained'],
            'last_training': self.stats['last_training'].isoformat() if self.stats['last_training'] else None,
            'best_model': self.stats['best_model'],
            'avg_accuracy': self.stats['avg_accuracy'],
            'symbols_with_models': list(self.price_prediction_models.keys()),
            'settings': self.settings
        }
    
    def save_models(self, filepath: str = "ml_models.pkl"):
        """💾 Сохранение моделей"""
        model_data = {
            'price_prediction_models': self.price_prediction_models,
            'scalers': self.scalers,
            'stats': self.stats,
            'settings': self.settings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Модели сохранены в {filepath}")
    
    def load_models(self, filepath: str = "ml_models.pkl"):
        """📂 Загрузка моделей"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.price_prediction_models = model_data.get('price_prediction_models', {})
            self.scalers = model_data.get('scalers', {})
            self.stats = model_data.get('stats', self.stats)
            self.settings = model_data.get('settings', self.settings)
            
            logger.info(f"📂 Модели загружены из {filepath}")
            
        except FileNotFoundError:
            logger.warning(f"⚠️ Файл {filepath} не найден")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")

# Пример использования
if __name__ == "__main__":
    ml_system = AdvancedMLSystem()
    
    # Тестовые данные
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
    test_data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Тестовые индикаторы
    test_indicators = {
        'rsi': np.random.uniform(20, 80, len(dates)),
        'bb_upper': test_data['close'] * 1.02,
        'bb_lower': test_data['close'] * 0.98,
        'ema_21': test_data['close'].rolling(21).mean(),
        'ema_50': test_data['close'].rolling(50).mean(),
        'macd': np.random.randn(len(dates)),
        'macd_signal': np.random.randn(len(dates))
    }
    
    print("🧠 ТЕСТ ПРОДВИНУТОЙ СИСТЕМЫ ML")
    print("=" * 40)
    
    # Создаём признаки
    features = ml_system.create_features(test_data, test_indicators)
    targets = ml_system.create_targets(test_data, [1, 3, 6])
    
    print(f"Создано признаков: {features.shape[1]}")
    print(f"Создано целей: {len(targets)}")
    
    # Обучаем модель
    if len(features) > 1000:
        result = ml_system.train_price_prediction_model('BTCUSDT', features, targets)
        print(f"Лучшая модель: {result['best_model']}")
        
        # Делаем предсказание
        prediction = ml_system.predict_price('BTCUSDT', features)
        if prediction:
            print(f"Предсказанная цена: {prediction.predicted_price:.2f}")
            print(f"Уверенность: {prediction.confidence:.2%}")
            print(f"Направление тренда: {prediction.trend_direction}")
    
    # Показываем статистику
    stats = ml_system.get_ml_statistics()
    print(f"\n📊 Статистика:")
    print(f"Обучено моделей: {stats['models_trained']}")
    print(f"Символов с моделями: {len(stats['symbols_with_models'])}")
    
    print("✅ Тест завершён!")
