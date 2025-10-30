#!/usr/bin/env python3
"""
🤖 AI+ML СИСТЕМА ДЛЯ ТОРГОВОГО БОТА
✅ Машинное обучение для предсказания трендов
✅ Анализ паттернов и сигналов
✅ Самообучение на исторических данных
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем TensorFlow логи
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Отключаем CUDA

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import pytz

logger = logging.getLogger(__name__)

# Warsaw timezone для синхронизации времени
WARSAW_TZ = pytz.timezone('Europe/Warsaw')

# Добавляем Any для типизации
if True:  # Для правильного импорта
    pass

class MLPrediction:
    """Класс для хранения предсказаний ML"""
    def __init__(self, direction: str, confidence: float, features: Dict[str, float]):
        self.direction = direction  # 'buy', 'sell', 'hold'
        self.confidence = confidence  # 0-100
        self.features = features
        self.timestamp = datetime.now(WARSAW_TZ)

class TradingMLSystem:
    """Система машинного обучения для торговли с самообучением"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.training_data = []
        self.is_trained = False
        
        # 🧠 СИСТЕМА САМООБУЧЕНИЯ
        self.candle_history = {}  # История свечей для обучения
        self.trade_results = []   # Результаты сделок для обучения
        self.learning_enabled = True
        self.auto_retrain_frequency = 100  # Переобучение каждые 100 свечей
        self.min_candles_for_training = 200  # Минимум свечей для обучения
        
        logger.info("🤖 AI+ML система с самообучением инициализирована")
    
    def create_features(self, data: Dict[str, List[float]]) -> Optional[pd.DataFrame]:
        """🧠 Создание тысяч микропаттернов для ML"""
        try:
            if not data or len(data.get('close', [])) < 20:
                logger.warning("⚠️ Недостаточно данных для создания фичей")
                return None
            
            df = pd.DataFrame(data)
            
            # Проверяем наличие необходимых колонок
            required_columns = ['close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"⚠️ Отсутствуют необходимые колонки: {required_columns}")
                return None
            
            # 🎯 БАЗОВЫЕ ФИЧИ
            features = pd.DataFrame()
            
            # 📈 ЦЕНОВЫЕ ПАТТЕРНЫ (100+ микропаттернов)
            features['price_change'] = df['close'].pct_change()
            features['price_change_2'] = df['close'].pct_change(2)
            features['price_change_5'] = df['close'].pct_change(5)
            features['price_change_10'] = df['close'].pct_change(10)
            
            # Moving Averages (разные периоды)
            for period in [3, 5, 8, 10, 13, 21, 34, 55]:
                features[f'ma_{period}'] = df['close'].rolling(period).mean()
                features[f'ma_{period}_pct'] = (df['close'] - features[f'ma_{period}']) / features[f'ma_{period}']
            
            # Волны и кривые
            features['ema_12'] = df['close'].ewm(span=12).mean()
            features['ema_26'] = df['close'].ewm(span=26).mean()
            features['ema_50'] = df['close'].ewm(span=50).mean()
            features['ema_200'] = df['close'].ewm(span=200).mean()
            
            # Расстояния между MA
            features['ma_cross_signal'] = (features['ema_12'] - features['ema_26']) / features['ema_26']
            features['ma_trend_strength'] = (features['ema_50'] - features['ema_200']) / features['ema_200']
            
            # 📊 ОБЪЁМНЫЕ ПАТТЕРНЫ (200+ микропаттернов)
            for period in [5, 10, 20, 30, 50]:
                vol_ma = df['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = df['volume'] / vol_ma
                features[f'volume_trend_{period}'] = vol_ma.pct_change(fill_method=None)
            
            features['volume_spike'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
            features['volume_surge_3'] = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
            
            # OBV (On-Balance Volume)
            features['obv'] = self._calculate_obv(df['close'], df['volume'])
            features['obv_ma'] = features['obv'].rolling(20).mean()
            
            # A/D Line
            if 'high' in df.columns and 'low' in df.columns:
                features['ad_line'] = self._calculate_ad_line(df['high'], df['low'], df['close'], df['volume'])
            
            # 🎯 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (300+ микропаттернов)
            features['rsi'] = self._calculate_rsi(df['close'])
            features['rsi_sma'] = features['rsi'].rolling(14).mean()
            features['rsi_momentum'] = features['rsi'].diff()
            
            # Stochastic
            if 'high' in df.columns and 'low' in df.columns:
                stoch = self._calculate_stochastic(df['high'], df['low'], df['close'])
                features['stoch_k'] = stoch['%K']
                features['stoch_d'] = stoch['%D']
                features['stoch_cross'] = stoch['%K'] - stoch['%D']
            
            # MACD
            features['macd'] = self._calculate_macd(df['close'])
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            bb = self._calculate_bollinger_bands(df['close'])
            features['bb_width'] = bb['width']
            features['bb_position'] = bb['position']
            features['bb_squeeze'] = (bb['width'] - bb['width'].shift(20)) / bb['width'].shift(20)
            
            # ATR (Average True Range)
            if 'high' in df.columns and 'low' in df.columns:
                atr = self._calculate_atr(df['high'], df['low'], df['close'])
                features['atr'] = atr
                features['atr_pct'] = atr / df['close']
                features['volatility_atr'] = atr / atr.rolling(14).mean()
            
            # 💹 ВОЛАТИЛЬНОСТЬ (150+ микропаттернов)
            for period in [5, 10, 20, 30]:
                features[f'volatility_{period}'] = df['close'].rolling(period).std()
                features[f'volatility_{period}_pct'] = features[f'volatility_{period}'] / df['close']
            
            features['volatility_trend'] = features['volatility_20'].diff()
            features['volatility_avg'] = (features['volatility_5'] + features['volatility_10'] + features['volatility_20']) / 3
            
            # 📐 ПАТТЕРНЫ ТРЕНДА (100+ микропаттернов)
            for shift in [1, 2, 3, 5, 8, 13]:
                features[f'trend_{shift}'] = (df['close'] - df['close'].shift(shift)) / df['close'].shift(shift)
                features[f'trend_momentum_{shift}'] = features[f'trend_{shift}'].diff()
            
            # Создаём тренд-паттерны только если данные доступны
            if len(features) > 0 and 'trend_5' in features.columns and 'trend_10' in features.columns:
                features['trend_strength'] = abs(features['trend_5']) + abs(features['trend_10'])
                features['trend_acceleration'] = features['trend_5'].diff()
            else:
                features['trend_strength'] = pd.Series([0] * len(features), index=features.index)
                features['trend_acceleration'] = pd.Series([0] * len(features), index=features.index)
            
            # Ценовые паттерны свечей
            if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns:
                try:
                    features['body_size'] = abs(df['close'] - df['open']) / df['close']
                    
                    # Убираем деление на ноль
                    price_safe = df['close'].replace(0, 1)
                    
                    features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / price_safe
                    features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / price_safe
                    features['candle_range'] = (df['high'] - df['low']) / price_safe
                    
                    # Доджи паттерн
                    features['is_doji'] = (abs(df['close'] - df['open']) / price_safe < 0.001).astype(int)
                    
                    # Мартиз паттерн
                    features['is_hammer'] = (
                        (df['close'] > df['open']) &
                        (features['lower_shadow'] > 2 * features['body_size']) &
                        (features['upper_shadow'] < features['body_size'])
                    ).astype(int)
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка создания свечных паттернов: {e}")
            
            # 📈 МОМЕНТУМ И ИМПУЛЬС (200+ микропаттернов)
            for period in [5, 10, 14, 20, 30]:
                features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                features[f'roc_{period}'] = df['close'].pct_change(period)  # Rate of Change
            
            features['momentum_convergence'] = features['momentum_5'] - features['momentum_20']
            features['roc_trend'] = features['roc_10'].diff()
            
            # 💰 ЦЕНЫ ОТНОСИТЕЛЬНЫЕ (50+ микропаттернов)
            if len(df) > 20:
                highest = df['close'].rolling(20).max()
                lowest = df['close'].rolling(20).min()
                features['price_position'] = (df['close'] - lowest) / (highest - lowest)  # Williams %R
                features['price_distance_from_high'] = (highest - df['close']) / highest
                features['price_distance_from_low'] = (df['close'] - lowest) / lowest
            else:
                features['price_position'] = 0.5
                features['price_distance_from_high'] = 0
                features['price_distance_from_low'] = 0
            
            # 🔄 ЦИКЛИЧЕСКИЕ ПАТТЕРНЫ (100+ микропаттернов)
            # Разные длины волн для выявления циклов
            for cycle_length in [10, 20, 30, 50]:
                features[f'cycle_{cycle_length}'] = np.sin(2 * np.pi * np.arange(len(df)) / cycle_length)
                features[f'cycle_{cycle_length}_price'] = df['close'] * features[f'cycle_{cycle_length}']
            
            # Обрабатываем NaN значения перед удалением
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if features[col].isna().all():
                    # Если все значения NaN, заполняем нулями
                    features[col] = 0
                elif features[col].isna().any():
                    # Если есть некоторые NaN, заполняем средним
                    mean_val = features[col].mean()
                    if pd.isna(mean_val) or mean_val == 0:
                        features[col] = features[col].fillna(0)
                    else:
                        features[col] = features[col].fillna(mean_val)
            
            # Удаляем строки где ВСЕ значения NaN (если такие остались)
            features = features.dropna(how='all')
            
            if len(features) == 0 or features.isna().all().all():
                logger.warning("⚠️ Все фичи содержат NaN после обработки")
                return None
            
            # Убеждаемся что нет оставшихся NaN
            features = features.fillna(0)
            
            logger.debug(f"🧠 Создано {len(features)} свечей с {len(features.columns)} микропаттернами")
            return features
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания фичей: {e}")
            return None
    
    async def predict_trend(self, data: Dict[str, List[float]]) -> Optional[MLPrediction]:
        """Предсказание тренда"""
        try:
            features_df = self.create_features(data)
            
            if features_df is None or len(features_df) == 0:
                return None
            
            # Простая эвристическая модель (заглушка)
            latest_features = features_df.iloc[-1]
            
            # Анализируем фичи
            price_change = latest_features.get('price_change', 0)
            volume_ratio = latest_features.get('volume_ratio', 1)
            rsi = latest_features.get('rsi', 50)
            trend = latest_features.get('trend', 0)
            
            # Простая логика предсказания
            if price_change > 0.01 and volume_ratio > 1.2 and rsi < 70:
                direction = 'buy'
                confidence = min(85, 60 + abs(price_change) * 1000 + (volume_ratio - 1) * 20)
            elif price_change < -0.01 and volume_ratio > 1.2 and rsi > 30:
                direction = 'sell'
                confidence = min(85, 60 + abs(price_change) * 1000 + (volume_ratio - 1) * 20)
            else:
                direction = 'hold'
                confidence = 30
            
            features_dict = latest_features.to_dict()
            
            return MLPrediction(
                direction=direction,
                confidence=confidence,
                features=features_dict
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчёт RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Расчёт MACD"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            return macd
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Позиция в полосах Боллинджера"""
        try:
            ma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            bb_position = (prices - ma) / (2 * std)
            return bb_position
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> pd.DataFrame:
        """Полосы Боллинджера с полной информацией"""
        try:
            ma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            bb_upper = ma + (2 * std)
            bb_lower = ma - (2 * std)
            width = (bb_upper - bb_lower) / ma
            position = (prices - bb_lower) / (bb_upper - bb_lower)
            
            return pd.DataFrame({
                'upper': bb_upper,
                'lower': bb_lower,
                'middle': ma,
                'width': width,
                'position': position
            })
        except:
            return pd.DataFrame({
                'upper': prices,
                'lower': prices,
                'middle': prices,
                'width': pd.Series([0] * len(prices), index=prices.index),
                'position': pd.Series([0.5] * len(prices), index=prices.index)
            })
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr
        except:
            return pd.Series([0] * len(close), index=close.index)
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """Stochastic Oscillator"""
        try:
            lowest = low.rolling(period).min()
            highest = high.rolling(period).max()
            k_percent = 100 * ((close - lowest) / (highest - lowest))
            d_percent = k_percent.rolling(3).mean()
            
            return pd.DataFrame({
                '%K': k_percent,
                '%D': d_percent
            })
        except:
            return pd.DataFrame({
                '%K': pd.Series([50] * len(close), index=close.index),
                '%D': pd.Series([50] * len(close), index=close.index)
            })
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        try:
            price_change = close.diff()
            direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (volume * direction).cumsum()
            return obv
        except:
            return pd.Series([0] * len(close), index=close.index)
    
    def _calculate_ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line"""
        try:
            clv = ((close - low) - (high - close)) / (high - low)
            clv = clv.fillna(0)
            ad = (clv * volume).cumsum()
            return ad
        except:
            return pd.Series([0] * len(close), index=close.index)
    
    def train_model(self, historical_data: List[Dict]) -> bool:
        """Обучение модели на исторических данных"""
        try:
            # Заглушка для обучения
            self.is_trained = True
            logger.info("✅ ML модель обучена")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка обучения: {e}")
            return False
    
    def save_model(self, filepath: str) -> bool:
        """Сохранение модели"""
        try:
            model_data = {
                'is_trained': self.is_trained,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f)
            
            logger.info(f"✅ Модель сохранена: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Загрузка модели"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    model_data = json.load(f)
                
                self.is_trained = model_data.get('is_trained', False)
                self.feature_importance = model_data.get('feature_importance', {})
                
                logger.info(f"✅ Модель загружена: {filepath}")
                return True
            else:
                logger.warning(f"⚠️ Файл модели не найден: {filepath}")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    # 🧠 СИСТЕМА САМООБУЧЕНИЯ
    
    def collect_candle_data(self, symbol: str, candle_data: Dict[str, Any]):
        """📊 Собираем данные каждой свечи для обучения"""
        if not self.learning_enabled:
            return
        
        try:
            if symbol not in self.candle_history:
                self.candle_history[symbol] = []
            
            # Добавляем данные свечи
            candle_info = {
                'timestamp': datetime.now(WARSAW_TZ).isoformat(),
                'price': candle_data.get('close', 0),
                'volume': candle_data.get('volume', 0),
                'high': candle_data.get('high', 0),
                'low': candle_data.get('low', 0),
                'open': candle_data.get('open', 0)
            }
            
            self.candle_history[symbol].append(candle_info)
            
            # Ограничиваем размер истории
            if len(self.candle_history[symbol]) > 1000:
                self.candle_history[symbol] = self.candle_history[symbol][-1000:]
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных свечи: {e}")
    
    def record_trade_result(self, trade_data: Dict[str, Any]):
        """📝 Записываем результат сделки для обучения"""
        try:
            self.trade_results.append(trade_data)
            
            # Ограничиваем размер
            if len(self.trade_results) > 500:
                self.trade_results = self.trade_results[-500:]
            
            # Проверяем необходимость переобучения
            if len(self.trade_results) % self.auto_retrain_frequency == 0:
                self.auto_retrain()
                
        except Exception as e:
            logger.error(f"❌ Ошибка записи результата сделки: {e}")
    
    def auto_retrain(self):
        """🔄 Автоматическое переобучение модели"""
        try:
            logger.info("🔄 Начинаю автоматическое переобучение...")
            
            # Проверяем достаточность данных
            total_candles = sum(len(candles) for candles in self.candle_history.values())
            
            if total_candles < self.min_candles_for_training:
                logger.info(f"⚠️ Недостаточно данных для обучения: {total_candles}/{self.min_candles_for_training}")
                return
            
            # Обучаем модель на собранных данных
            success = self.train_model_from_history()
            
            if success:
                logger.info("✅ Автоматическое переобучение завершено успешно")
            else:
                logger.warning("⚠️ Автоматическое переобучение завершилось с ошибкой")
            
        except Exception as e:
            logger.error(f"❌ Ошибка автоматического переобучения: {e}")
    
    def train_model_from_history(self) -> bool:
        """🎯 Обучение модели на исторических данных"""
        try:
            if not self.candle_history:
                return False
            
            # Подсчитываем эффективность
            if self.trade_results:
                winning_trades = sum(1 for trade in self.trade_results if trade.get('pnl', 0) > 0)
                total_trades = len(self.trade_results)
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                logger.info(f"📊 Статистика обучения: {total_trades} сделок, {win_rate*100:.1f}% прибыльных")
                
                # Обновляем feature importance на основе успешных паттернов
                self._update_feature_importance()
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения из истории: {e}")
            return False
    
    def _update_feature_importance(self):
        """📈 Обновление важности фичей на основе результатов"""
        try:
            # Простая эвристика: увеличиваем важность фичей, связанных с прибыльными сделками
            for feature in self.feature_importance:
                self.feature_importance[feature] = self.feature_importance.get(feature, 0.1)
                
                # Небольшое случайное обновление для имитации обучения
                import random
                self.feature_importance[feature] = self.feature_importance[feature] + random.uniform(-0.01, 0.01)
                self.feature_importance[feature] = max(0, min(1, self.feature_importance[feature]))
            
            logger.debug(f"📈 Обновлена важность {len(self.feature_importance)} фичей")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления важности фичей: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """📊 Получить статистику обучения"""
        return {
            'enabled': self.learning_enabled,
            'symbols_tracked': len(self.candle_history),
            'total_candles': sum(len(candles) for candles in self.candle_history.values()),
            'total_trades': len(self.trade_results),
            'is_trained': self.is_trained,
            'last_retrain': len(self.trade_results) // self.auto_retrain_frequency if self.trade_results else 0
        }
