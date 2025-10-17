"""
🧠 СИСТЕМА САМООБУЧЕНИЯ БОТА V2.0
Сбор данных, обучение ML модели и улучшение торговых сигналов
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TradingDataCollector:
    """Сборщик данных для обучения ML модели"""
    
    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Файлы данных
        self.trades_file = self.data_dir / "trades_history.json"
        self.features_file = self.data_dir / "features_dataset.json"
        self.model_file = self.data_dir / "trading_model.pkl"
        self.scaler_file = self.data_dir / "feature_scaler.pkl"
        
        # Загружаем существующие данные
        self.trades_data = self._load_trades_data()
        self.features_data = self._load_features_data()
    
    def _load_trades_data(self) -> List[Dict]:
        """Загружаем историю сделок"""
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки сделок: {e}")
        return []
    
    def _load_features_data(self) -> List[Dict]:
        """Загружаем данные признаков"""
        if self.features_file.exists():
            try:
                with open(self.features_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки признаков: {e}")
        return []
    
    def save_trades_data(self):
        """Сохраняем историю сделок"""
        try:
            with open(self.trades_file, 'w', encoding='utf-8') as f:
                json.dump(self.trades_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения сделок: {e}")
    
    def save_features_data(self):
        """Сохраняем данные признаков"""
        try:
            with open(self.features_file, 'w', encoding='utf-8') as f:
                json.dump(self.features_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения признаков: {e}")
    
    def add_trade_result(self, trade_data: Dict[str, Any]):
        """Добавляем результат сделки"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_data.get('symbol'),
            'side': trade_data.get('side'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('exit_price'),
            'pnl': trade_data.get('pnl'),
            'pnl_percent': trade_data.get('pnl_percent'),
            'confidence': trade_data.get('confidence'),
            'reason': trade_data.get('reason'),
            'duration_minutes': trade_data.get('duration_minutes', 0),
            'success': trade_data.get('pnl', 0) > 0  # True если прибыль > 0
        }
        
        self.trades_data.append(trade_record)
        logger.info(f"📊 Добавлена сделка: {trade_record['symbol']} PnL=${trade_record['pnl']:.2f}")
        
        # Сохраняем каждые 10 сделок
        if len(self.trades_data) % 10 == 0:
            self.save_trades_data()
    
    def add_signal_features(self, symbol: str, signal_data: Dict[str, Any], market_data: Dict[str, Any]):
        """Добавляем признаки сигнала для обучения"""
        feature_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal_data.get('signal'),
            'confidence': signal_data.get('confidence'),
            'signal_strength': signal_data.get('signal_strength', 0),
            
            # Технические индикаторы
            'rsi': market_data.get('rsi', 0),
            'macd_signal': market_data.get('macd_signal', 0),
            'bollinger_position': market_data.get('bollinger_position', 0),
            'ema_trend': market_data.get('ema_trend', 0),
            'volume_ratio': market_data.get('volume_ratio', 0),
            'stochastic': market_data.get('stochastic', 0),
            
            # Рыночные условия
            'price': market_data.get('price', 0),
            'volume_24h': market_data.get('volume_24h', 0),
            'volatility': market_data.get('volatility', 0),
            'atr': market_data.get('atr', 0),
            
            # Временные факторы
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': datetime.now().weekday() >= 5,
            
            # Результат (будет заполнен позже)
            'actual_success': None,
            'actual_pnl': None
        }
        
        self.features_data.append(feature_record)
        logger.debug(f"🔍 Добавлены признаки: {symbol} {signal_data.get('signal', 'NONE')}")
    
    def update_signal_result(self, symbol: str, timestamp: str, success: bool, pnl: float):
        """Обновляем результат сигнала"""
        for feature in self.features_data:
            if (feature['symbol'] == symbol and 
                feature['timestamp'] == timestamp and 
                feature['actual_success'] is None):
                feature['actual_success'] = success
                feature['actual_pnl'] = pnl
                break
        
        logger.debug(f"📈 Обновлён результат: {symbol} Success={success} PnL=${pnl:.2f}")
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Получаем данные для обучения"""
        if len(self.features_data) < 50:
            logger.warning("⚠️ Недостаточно данных для обучения (< 50 записей)")
            return np.array([]), np.array([])
        
        # Фильтруем записи с результатами
        completed_features = [f for f in self.features_data if f['actual_success'] is not None]
        
        if len(completed_features) < 30:
            logger.warning("⚠️ Недостаточно завершённых сделок для обучения (< 30)")
            return np.array([]), np.array([])
        
        # Подготавливаем признаки
        feature_columns = [
            'confidence', 'signal_strength', 'rsi', 'macd_signal', 
            'bollinger_position', 'ema_trend', 'volume_ratio', 'stochastic',
            'volume_24h', 'volatility', 'atr', 'hour', 'day_of_week'
        ]
        
        X = []
        y = []
        
        for record in completed_features:
            features = [record.get(col, 0) for col in feature_columns]
            X.append(features)
            y.append(1 if record['actual_success'] else 0)
        
        return np.array(X), np.array(y)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получаем статистику по данным"""
        if not self.trades_data:
            return {"total_trades": 0, "success_rate": 0}
        
        total_trades = len(self.trades_data)
        successful_trades = sum(1 for t in self.trades_data if t['success'])
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trades_data)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "success_rate": success_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "features_count": len(self.features_data),
            "completed_features": len([f for f in self.features_data if f['actual_success'] is not None])
        }


class MLTradingPredictor:
    """ML модель для предсказания успешности торговых сигналов"""
    
    def __init__(self, data_collector: TradingDataCollector):
        self.data_collector = data_collector
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_time = None
        
        # Загружаем существующую модель
        self._load_model()
    
    def _load_model(self):
        """Загружаем обученную модель"""
        try:
            if self.data_collector.model_file.exists():
                self.model = joblib.load(self.data_collector.model_file)
                self.is_trained = True
                logger.info("✅ ML модель загружена")
            
            if self.data_collector.scaler_file.exists():
                self.scaler = joblib.load(self.data_collector.scaler_file)
                logger.info("✅ Scaler загружен")
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
    
    def _save_model(self):
        """Сохраняем обученную модель"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.data_collector.model_file)
                joblib.dump(self.scaler, self.data_collector.scaler_file)
                logger.info("💾 ML модель сохранена")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
    
    def train_model(self) -> Dict[str, Any]:
        """Обучаем ML модель"""
        logger.info("🧠 Начинаю обучение ML модели...")
        
        X, y = self.data_collector.get_training_data()
        
        if len(X) == 0:
            return {"error": "Недостаточно данных для обучения"}
        
        try:
            # Разделяем на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Нормализуем признаки
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучаем модель (используем GradientBoosting для лучшей точности)
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Оцениваем качество
            y_pred = self.model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            # Сохраняем модель
            self._save_model()
            
            results = {
                "success": True,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            logger.info(f"✅ Модель обучена! Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения модели: {e}")
            return {"error": str(e)}
    
    def predict_signal_success(self, signal_features: Dict[str, Any]) -> Dict[str, Any]:
        """Предсказываем успешность сигнала"""
        if not self.is_trained or self.model is None:
            return {
                "prediction": None,
                "confidence": 0,
                "error": "Модель не обучена"
            }
        
        try:
            # Подготавливаем признаки в том же порядке что и при обучении
            feature_columns = [
                'confidence', 'signal_strength', 'rsi', 'macd_signal', 
                'bollinger_position', 'ema_trend', 'volume_ratio', 'stochastic',
                'volume_24h', 'volatility', 'atr', 'hour', 'day_of_week'
            ]
            
            features = np.array([[signal_features.get(col, 0) for col in feature_columns]])
            features_scaled = self.scaler.transform(features)
            
            # Предсказание
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Вероятность успеха
            success_prob = probability[1] if len(probability) > 1 else 0.5
            
            return {
                "prediction": bool(prediction),
                "confidence": success_prob,
                "probability_success": success_prob,
                "probability_failure": probability[0] if len(probability) > 1 else 0.5
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            return {
                "prediction": None,
                "confidence": 0,
                "error": str(e)
            }
    
    def should_retrain(self) -> bool:
        """Проверяем нужно ли переобучать модель"""
        if not self.is_trained:
            return True
        
        if self.last_training_time is None:
            return True
        
        # Переобучаем если прошло больше 24 часов
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training > timedelta(hours=24)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получаем информацию о модели"""
        stats = self.data_collector.get_statistics()
        
        return {
            "is_trained": self.is_trained,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
            "should_retrain": self.should_retrain(),
            "training_data_stats": stats
        }


class SelfLearningSystem:
    """Главная система самообучения"""
    
    def __init__(self):
        self.data_collector = TradingDataCollector()
        self.ml_predictor = MLTradingPredictor(self.data_collector)
        
        # Настройки
        self.min_trades_for_training = 50
        self.retrain_interval_hours = 24
        self.ml_confidence_threshold = 0.7
        
        logger.info("🧠 Система самообучения инициализирована")
    
    def record_trade_result(self, trade_data: Dict[str, Any]):
        """Записываем результат сделки"""
        self.data_collector.add_trade_result(trade_data)
        
        # Проверяем нужно ли переобучать модель
        if self.ml_predictor.should_retrain():
            stats = self.data_collector.get_statistics()
            if stats['total_trades'] >= self.min_trades_for_training:
                logger.info("🔄 Автоматическое переобучение модели...")
                self.train_model()
    
    def record_signal_features(self, symbol: str, signal_data: Dict[str, Any], market_data: Dict[str, Any]):
        """Записываем признаки сигнала"""
        self.data_collector.add_signal_features(symbol, signal_data, market_data)
    
    def update_signal_result(self, symbol: str, timestamp: str, success: bool, pnl: float):
        """Обновляем результат сигнала"""
        self.data_collector.update_signal_result(symbol, timestamp, success, pnl)
    
    def predict_signal_quality(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Предсказываем качество сигнала"""
        if not self.ml_predictor.is_trained:
            return {
                "ml_prediction": None,
                "ml_confidence": 0,
                "recommendation": "trade",  # Торгуем без ML если модель не обучена
                "reason": "ML модель не обучена"
            }
        
        # Подготавливаем признаки
        features = {
            'confidence': signal_data.get('confidence', 0),
            'signal_strength': signal_data.get('signal_strength', 0),
            'rsi': market_data.get('rsi', 50),
            'macd_signal': market_data.get('macd_signal', 0),
            'bollinger_position': market_data.get('bollinger_position', 0),
            'ema_trend': market_data.get('ema_trend', 0),
            'volume_ratio': market_data.get('volume_ratio', 1),
            'stochastic': market_data.get('stochastic', 50),
            'volume_24h': market_data.get('volume_24h', 0),
            'volatility': market_data.get('volatility', 0),
            'atr': market_data.get('atr', 0),
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
        
        prediction = self.ml_predictor.predict_signal_success(features)
        
        if prediction.get('error'):
            return {
                "ml_prediction": None,
                "ml_confidence": 0,
                "recommendation": "trade",
                "reason": f"ML ошибка: {prediction['error']}"
            }
        
        ml_confidence = prediction.get('confidence', 0)
        
        # Рекомендация на основе ML предсказания
        if ml_confidence >= self.ml_confidence_threshold:
            recommendation = "trade"
            reason = f"ML рекомендует торговать (уверенность: {ml_confidence:.2f})"
        else:
            recommendation = "skip"
            reason = f"ML не рекомендует торговать (уверенность: {ml_confidence:.2f})"
        
        return {
            "ml_prediction": prediction.get('prediction'),
            "ml_confidence": ml_confidence,
            "recommendation": recommendation,
            "reason": reason,
            "probability_success": prediction.get('probability_success', 0),
            "probability_failure": prediction.get('probability_failure', 0)
        }
    
    def train_model(self) -> Dict[str, Any]:
        """Обучаем модель"""
        return self.ml_predictor.train_model()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Получаем статистику обучения"""
        stats = self.data_collector.get_statistics()
        model_info = self.ml_predictor.get_model_info()
        
        return {
            **stats,
            **model_info,
            "min_trades_for_training": self.min_trades_for_training,
            "ml_confidence_threshold": self.ml_confidence_threshold
        }
    
    def save_all_data(self):
        """Сохраняем все данные"""
        self.data_collector.save_trades_data()
        self.data_collector.save_features_data()


# Глобальный экземпляр системы самообучения
self_learning_system = SelfLearningSystem()
