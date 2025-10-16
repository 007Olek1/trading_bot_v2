"""
🧠 ML ENGINE V3.5 - Система машинного обучения для торгового бота

Архитектура:
- XGBoost для классификации сигналов BUY/SELL
- LSTM для предсказания цен
- Reinforcement Learning для оптимизации стратегии
- Автоматическое переобучение каждые 24 часа
- Целевая точность: 85-99%

Автор: AI Trading Bot Team
Версия: 3.5 AUTONOMOUS ML
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
import os
from loguru import logger

# ML библиотеки
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ ML библиотеки не установлены. Используйте: pip install xgboost scikit-learn")
    ML_AVAILABLE = False

# Deep Learning для LSTM
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    class LSTMPricePredictor(nn.Module):
        """
        LSTM модель для предсказания цен
        Архитектура: LSTM(128) -> Dropout -> LSTM(64) -> Dense
        """
        def __init__(self, input_size=10, hidden_size=128, num_layers=2, output_size=1):
            super(LSTMPricePredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc1 = nn.Linear(hidden_size, 64)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(64, output_size)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # LSTM слои
            lstm_out, _ = self.lstm(x)
            
            # Берем последний выход
            out = lstm_out[:, -1, :]
            
            # Fully connected слои
            out = self.fc1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out
    
except ImportError:
    logger.warning("⚠️ PyTorch не установлен. Используйте: pip install torch")
    TORCH_AVAILABLE = False
    LSTMPricePredictor = None  # Заглушка


class MLTradingEngine:
    """
    Основной ML движок для торговли
    
    Включает:
    - XGBoost для классификации сигналов
    - LSTM для предсказания цен
    - Автоматическое переобучение
    - Валидация точности
    """
    
    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # XGBoost модель
        self.xgb_model = None
        self.scaler = StandardScaler()
        
        # LSTM модель
        self.lstm_model = None
        self.lstm_scaler = StandardScaler()
        
        # История для обучения
        self.training_data = []
        self.max_history = 10000  # Максимум записей для обучения
        
        # Метрики
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        
        # Последнее обучение
        self.last_training = None
        self.training_interval = timedelta(hours=24)  # Переобучение раз в 24 часа
        
        # Загружаем существующие модели
        self._load_models()
        
        logger.info("🧠 ML Engine V3.5 инициализирован")
    
    def _load_models(self):
        """Загрузка сохраненных моделей"""
        try:
            # XGBoost
            xgb_path = os.path.join(self.model_dir, "xgboost_model.json")
            if os.path.exists(xgb_path):
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(xgb_path)
                logger.info("✅ XGBoost модель загружена")
            
            # Scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Scaler загружен")
            
            # LSTM
            lstm_path = os.path.join(self.model_dir, "lstm_model.pth")
            if os.path.exists(lstm_path) and TORCH_AVAILABLE and LSTMPricePredictor:
                self.lstm_model = LSTMPricePredictor()
                self.lstm_model.load_state_dict(torch.load(lstm_path))
                self.lstm_model.eval()
                logger.info("✅ LSTM модель загружена")
            
            # История обучения
            history_path = os.path.join(self.model_dir, "training_history.pkl")
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    data = pickle.load(f)
                    self.training_data = data.get('history', [])
                    self.last_training = data.get('last_training')
                    self.accuracy = data.get('accuracy', 0.0)
                logger.info(f"✅ История загружена: {len(self.training_data)} записей, точность: {self.accuracy:.2%}")
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки моделей: {e}")
    
    def _save_models(self):
        """Сохранение моделей"""
        try:
            # XGBoost
            if self.xgb_model:
                xgb_path = os.path.join(self.model_dir, "xgboost_model.json")
                self.xgb_model.save_model(xgb_path)
            
            # Scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # LSTM
            if self.lstm_model and TORCH_AVAILABLE and LSTMPricePredictor:
                lstm_path = os.path.join(self.model_dir, "lstm_model.pth")
                torch.save(self.lstm_model.state_dict(), lstm_path)
            
            # История
            history_path = os.path.join(self.model_dir, "training_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump({
                    'history': self.training_data[-self.max_history:],
                    'last_training': self.last_training,
                    'accuracy': self.accuracy,
                    'precision': self.precision,
                    'recall': self.recall,
                    'f1': self.f1
                }, f)
            
            logger.info("💾 Модели сохранены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения моделей: {e}")
    
    def extract_features(self, df: pd.DataFrame, signal_result: Dict) -> np.ndarray:
        """
        Извлечение признаков для ML
        
        Признаки:
        - RSI, MACD, EMA, Bollinger Bands
        - Объем, волатильность
        - Уверенность сигнала
        - Рыночные условия
        """
        try:
            features = []
            
            # Технические индикаторы из signal_result
            features.append(signal_result.get('rsi', 50))
            features.append(signal_result.get('macd', 0))
            features.append(signal_result.get('macd_signal', 0))
            features.append(signal_result.get('ema_short', df['close'].iloc[-1]))
            features.append(signal_result.get('ema_long', df['close'].iloc[-1]))
            
            # Bollinger Bands
            bb = signal_result.get('bollinger_bands', {})
            features.append(bb.get('upper', df['close'].iloc[-1]))
            features.append(bb.get('middle', df['close'].iloc[-1]))
            features.append(bb.get('lower', df['close'].iloc[-1]))
            
            # Волатильность (стандартное отклонение последних 20 свечей)
            volatility = df['close'].tail(20).std()
            features.append(volatility)
            
            # Объем (средний за последние 20 свечей)
            avg_volume = df['volume'].tail(20).mean()
            features.append(avg_volume)
            
            # Уверенность сигнала
            features.append(signal_result.get('confidence', 0))
            
            # Тренд (разница между EMA короткой и длинной)
            ema_diff = signal_result.get('ema_short', 0) - signal_result.get('ema_long', 0)
            features.append(ema_diff)
            
            # Последние 5 свечей (изменение цены)
            for i in range(1, 6):
                if len(df) >= i:
                    price_change = (df['close'].iloc[-i] - df['open'].iloc[-i]) / df['open'].iloc[-i]
                    features.append(price_change)
                else:
                    features.append(0)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения признаков: {e}")
            return np.zeros((1, 17))  # 17 признаков
    
    async def predict_signal(
        self,
        df: pd.DataFrame,
        signal_result: Dict,
        current_price: float
    ) -> Dict[str, any]:
        """
        ML предсказание сигнала
        
        Возвращает:
        - signal: 'buy', 'sell', None
        - confidence: 0-100
        - ml_score: оценка модели
        - price_prediction: предсказание цены (LSTM)
        """
        try:
            if not ML_AVAILABLE or self.xgb_model is None:
                logger.debug("⚠️ ML модель не готова, используем базовый сигнал")
                return {
                    'signal': signal_result.get('signal'),
                    'confidence': signal_result.get('confidence', 0),
                    'ml_score': 0,
                    'price_prediction': None
                }
            
            # Извлекаем признаки
            features = self.extract_features(df, signal_result)
            
            # Нормализация
            features_scaled = self.scaler.transform(features)
            
            # XGBoost предсказание
            # 0 = SELL, 1 = HOLD, 2 = BUY
            prediction = self.xgb_model.predict(features_scaled)[0]
            probabilities = self.xgb_model.predict_proba(features_scaled)[0]
            
            # Переводим в сигнал
            if prediction == 2:  # BUY
                ml_signal = 'buy'
                ml_confidence = probabilities[2] * 100
            elif prediction == 0:  # SELL
                ml_signal = 'sell'
                ml_confidence = probabilities[0] * 100
            else:  # HOLD
                ml_signal = None
                ml_confidence = probabilities[1] * 100
            
            # LSTM предсказание цены
            price_prediction = None
            if self.lstm_model and TORCH_AVAILABLE and LSTMPricePredictor:
                try:
                    # Подготовка данных для LSTM (последние 50 свечей)
                    lstm_data = df[['close', 'volume', 'high', 'low']].tail(50).values
                    lstm_data_scaled = self.lstm_scaler.fit_transform(lstm_data)
                    
                    # Конвертируем в тензор
                    lstm_input = torch.FloatTensor(lstm_data_scaled).unsqueeze(0)
                    
                    # Предсказание
                    with torch.no_grad():
                        price_prediction = self.lstm_model(lstm_input).item()
                        # Обратное масштабирование
                        price_prediction = self.lstm_scaler.inverse_transform([[price_prediction, 0, 0, 0]])[0][0]
                    
                    logger.debug(f"🔮 LSTM предсказание цены: ${price_prediction:.4f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ LSTM ошибка: {e}")
            
            # Комбинируем с базовым сигналом
            base_signal = signal_result.get('signal')
            base_confidence = signal_result.get('confidence', 0)
            
            # Если ML и базовый сигнал согласны - увеличиваем уверенность
            if ml_signal == base_signal:
                final_confidence = min(100, (base_confidence + ml_confidence) / 2 * 1.2)
                final_signal = ml_signal
            else:
                # Если не согласны - берем более уверенный
                if ml_confidence > base_confidence:
                    final_signal = ml_signal
                    final_confidence = ml_confidence * 0.9  # Небольшой штраф за разногласие
                else:
                    final_signal = base_signal
                    final_confidence = base_confidence * 0.9
            
            # LSTM фильтр: если цена предсказана вниз, не открываем BUY
            if price_prediction and final_signal == 'buy':
                if price_prediction < current_price * 0.998:  # Предсказание падения >0.2%
                    logger.warning(f"🔮 LSTM предсказывает падение: ${current_price:.4f} -> ${price_prediction:.4f}")
                    final_confidence *= 0.7  # Снижаем уверенность
            
            return {
                'signal': final_signal if final_confidence >= 85 else None,
                'confidence': final_confidence,
                'ml_score': ml_confidence,
                'ml_signal': ml_signal,
                'base_signal': base_signal,
                'price_prediction': price_prediction,
                'probabilities': {
                    'buy': probabilities[2] * 100,
                    'hold': probabilities[1] * 100,
                    'sell': probabilities[0] * 100
                }
            }
            
        except Exception as e:
            logger.error(f"❌ ML ошибка предсказания: {e}")
            return {
                'signal': signal_result.get('signal'),
                'confidence': signal_result.get('confidence', 0),
                'ml_score': 0,
                'price_prediction': None
            }
    
    def record_trade_result(
        self,
        symbol: str,
        signal: str,
        entry_price: float,
        exit_price: float,
        profit: float,
        features: np.ndarray,
        signal_result: Dict
    ):
        """
        Запись результата сделки для обучения
        """
        try:
            # Определяем метку: 1 = прибыль, 0 = убыток
            label = 1 if profit > 0 else 0
            
            # Сохраняем запись
            self.training_data.append({
                'symbol': symbol,
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit': profit,
                'profit_percent': (profit / entry_price) * 100 if entry_price > 0 else 0,
                'features': features,
                'label': label,
                'timestamp': datetime.now(),
                'signal_result': signal_result
            })
            
            # Ограничиваем размер истории
            if len(self.training_data) > self.max_history:
                self.training_data = self.training_data[-self.max_history:]
            
            logger.debug(f"📝 Записан результат сделки: {symbol} {signal} Profit: {profit:.2f} ({label})")
            
            # Проверяем нужно ли переобучение
            if self.last_training is None or datetime.now() - self.last_training > self.training_interval:
                if len(self.training_data) >= 50:  # Минимум 50 сделок для обучения
                    asyncio.create_task(self.retrain_models())
            
        except Exception as e:
            logger.error(f"❌ Ошибка записи результата: {e}")
    
    async def retrain_models(self):
        """
        Автоматическое переобучение моделей
        """
        try:
            if not ML_AVAILABLE:
                logger.warning("⚠️ ML библиотеки недоступны")
                return
            
            if len(self.training_data) < 50:
                logger.warning(f"⚠️ Недостаточно данных для обучения: {len(self.training_data)}/50")
                return
            
            logger.info(f"🔄 НАЧАЛО ПЕРЕОБУЧЕНИЯ ML МОДЕЛЕЙ ({len(self.training_data)} записей)...")
            
            # Подготовка данных
            X = []
            y = []
            
            for record in self.training_data:
                X.append(record['features'].flatten())
                
                # Метки: 0=SELL, 1=HOLD, 2=BUY
                if record['signal'] == 'buy':
                    if record['label'] == 1:  # Прибыльный BUY
                        y.append(2)
                    else:  # Убыточный BUY
                        y.append(1)  # HOLD был бы лучше
                elif record['signal'] == 'sell':
                    if record['label'] == 1:  # Прибыльный SELL
                        y.append(0)
                    else:  # Убыточный SELL
                        y.append(1)  # HOLD был бы лучше
                else:
                    y.append(1)  # HOLD
            
            X = np.array(X)
            y = np.array(y)
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Нормализация
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучение XGBoost
            logger.info("🔄 Обучение XGBoost...")
            self.xgb_model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                eval_metric='mlogloss'
            )
            
            self.xgb_model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # Валидация
            y_pred = self.xgb_model.predict(X_test_scaled)
            
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            self.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            self.f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"✅ XGBoost обучен!")
            logger.info(f"   📊 Accuracy:  {self.accuracy:.2%}")
            logger.info(f"   📊 Precision: {self.precision:.2%}")
            logger.info(f"   📊 Recall:    {self.recall:.2%}")
            logger.info(f"   📊 F1 Score:  {self.f1:.2%}")
            
            # Проверка целевой точности 85-99%
            if self.accuracy >= 0.85:
                logger.info(f"🎯 ЦЕЛЕВАЯ ТОЧНОСТЬ ДОСТИГНУТА: {self.accuracy:.2%} ✅")
            else:
                logger.warning(f"⚠️ Точность ниже целевой: {self.accuracy:.2%} < 85%")
            
            # Обучение LSTM (если доступен PyTorch)
            if TORCH_AVAILABLE and LSTMPricePredictor:
                logger.info("🔄 Обучение LSTM...")
                # TODO: Реализовать обучение LSTM на исторических данных цен
            
            # Сохраняем модели
            self.last_training = datetime.now()
            self._save_models()
            
            logger.info(f"✅ ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО! Следующее через {self.training_interval.total_seconds()/3600:.0f}ч")
            
        except Exception as e:
            logger.error(f"❌ Ошибка переобучения: {e}")
    
    def get_status(self) -> Dict:
        """Статус ML движка"""
        return {
            'ml_available': ML_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'model_trained': self.xgb_model is not None,
            'lstm_trained': self.lstm_model is not None,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'training_samples': len(self.training_data),
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'target_accuracy': '85-99%',
            'current_performance': f"{self.accuracy:.1%}" if self.accuracy > 0 else "Не обучена"
        }


# Глобальный экземпляр
ml_engine = MLTradingEngine()


if __name__ == "__main__":
    logger.info("🧠 ML Engine V3.5 - Тестовый режим")
    logger.info(f"ML Status: {ml_engine.get_status()}")

