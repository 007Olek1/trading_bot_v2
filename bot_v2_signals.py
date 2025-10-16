"""
📊 АНАЛИЗАТОР ТОРГОВЫХ СИГНАЛОВ V2.0
Простые, но эффективные индикаторы
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from bot_v2_config import Config

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """Анализ сигналов на основе технических индикаторов"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_middle'] = sma_20
            
            # EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
            
            # Volume MA
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Расчет силы тренда (-1 до +1)"""
        try:
            if len(df) < 50:
                return 0
            
            # 1. EMA тренд
            ema_20 = df['close'].ewm(span=20).mean()
            ema_50 = df['close'].ewm(span=50).mean()
            
            ema_trend = 1 if ema_20.iloc[-1] > ema_50.iloc[-1] else -1
            
            # 2. MACD тренд
            macd_line = ema_20 - ema_50
            macd_signal = macd_line.ewm(span=9).mean()
            macd_trend = 1 if macd_line.iloc[-1] > macd_signal.iloc[-1] else -1
            
            # 3. Momentum (скорость изменения)
            momentum_short = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            momentum_long = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            momentum_trend = 1 if momentum_short > 0 and momentum_long > 0 else -1 if momentum_short < 0 and momentum_long < 0 else 0
            
            # 4. Объемный тренд
            volume_trend = 0
            if len(df) >= 20:
                recent_volume = df['volume'].iloc[-5:].mean()
                avg_volume = df['volume'].iloc[-20:].mean()
                if recent_volume > avg_volume * 1.2:
                    volume_trend = 1 if momentum_short > 0 else -1
            
            # Комбинированный скор
            trend_strength = (ema_trend * 0.3 + macd_trend * 0.3 + momentum_trend * 0.3 + volume_trend * 0.1)
            
            return max(-1, min(1, trend_strength))
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета силы тренда: {e}")
            return 0
    
    def _analyze_volatility_breakout(self, df: pd.DataFrame) -> Optional[tuple]:
        """Анализ волатильного пробоя"""
        try:
            if len(df) < 20:
                return None
            
            # Bollinger Bands для определения пробоя
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            
            # Пробой вверх с объемом
            if current_price > bb_upper.iloc[-1] and prev_price <= bb_upper.iloc[-2]:
                volume_spike = df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean() * 1.5
                if volume_spike:
                    return ("buy", 0.9)
            
            # Пробой вниз с объемом
            elif current_price < bb_lower.iloc[-1] and prev_price >= bb_lower.iloc[-2]:
                volume_spike = df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean() * 1.5
                if volume_spike:
                    return ("sell", 0.9)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа пробоя: {e}")
            return None
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ сигнала
        
        Returns:
            {
                "signal": "buy" | "sell" | None,
                "confidence": 0-100,
                "reason": str,
                "indicators": dict
            }
        """
        try:
            if len(df) < 50:
                return {"signal": None, "confidence": 0, "reason": "Недостаточно данных"}
            
            # Рассчитываем индикаторы
            df = self.calculate_indicators(df)
            
            # Последние значения
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Собираем сигналы
            signals = []
            reasons = []
            
            # СИГНАЛ 1: RSI
            rsi = current['rsi']
            if rsi < 30:  # Перепроданность
                signals.append(("buy", 0.8))
                reasons.append(f"RSI={rsi:.1f} перепроданность")
            elif rsi > 70:  # Перекупленность
                signals.append(("sell", 0.8))
                reasons.append(f"RSI={rsi:.1f} перекупленность")
            
            # СИГНАЛ 2: MACD кроссовер
            if prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
                signals.append(("buy", 0.9))
                reasons.append("MACD бычий кроссовер")
            elif prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']:
                signals.append(("sell", 0.9))
                reasons.append("MACD медвежий кроссовер")
            
            # СИГНАЛ 3: Bollinger Bands
            if current['close'] < current['bb_lower']:
                signals.append(("buy", 0.75))
                reasons.append("Цена ниже нижней BB")
            elif current['close'] > current['bb_upper']:
                signals.append(("sell", 0.75))
                reasons.append("Цена выше верхней BB")
            
            # СИГНАЛ 4: EMA тренд
            if current['ema_20'] > current['ema_50'] and current['close'] > current['ema_20']:
                signals.append(("buy", 0.7))
                reasons.append("Бычий тренд EMA")
            elif current['ema_20'] < current['ema_50'] and current['close'] < current['ema_20']:
                signals.append(("sell", 0.7))
                reasons.append("Медвежий тренд EMA")
            
            # СИГНАЛ 5: Объем
            if current['volume'] > current['volume_ma'] * 1.5:
                # Высокий объем усиливает сигнал
                if signals and signals[-1][0] == "buy":
                    signals.append(("buy", 0.6))
                    reasons.append("Высокий объем подтверждает")
                elif signals and signals[-1][0] == "sell":
                    signals.append(("sell", 0.6))
                    reasons.append("Высокий объем подтверждает")
            
            # СИГНАЛ 7: УЛУЧШЕННЫЙ АНАЛИЗ ТРЕНДОВ
            trend_strength = self._calculate_trend_strength(df)
            if trend_strength > 0.7:  # Сильный бычий тренд
                signals.append(("buy", 0.8))
                reasons.append(f"Сильный бычий тренд ({trend_strength:.2f})")
            elif trend_strength < -0.7:  # Сильный медвежий тренд
                signals.append(("sell", 0.8))
                reasons.append(f"Сильный медвежий тренд ({trend_strength:.2f})")
            
            # СИГНАЛ 8: ВОЛАТИЛЬНОСТЬ И BREAKOUT
            volatility_signal = self._analyze_volatility_breakout(df)
            if volatility_signal:
                signals.append(volatility_signal)
                reasons.append("Волатильный пробой")
            
            # СИГНАЛ 7: Поддержка/Сопротивление
            recent_highs = df['high'].rolling(window=20).max()
            recent_lows = df['low'].rolling(window=20).min()
            
            resistance = recent_highs.iloc[-1]
            support = recent_lows.iloc[-1]
            
            # Если цена у поддержки - BUY
            if abs(current['close'] - support) / current['close'] < 0.01:  # В пределах 1%
                signals.append(("buy", 0.8))
                reasons.append(f"Цена у поддержки ${support:.4f}")
            # Если цена у сопротивления - SELL
            elif abs(current['close'] - resistance) / current['close'] < 0.01:
                signals.append(("sell", 0.8))
                reasons.append(f"Цена у сопротивления ${resistance:.4f}")
            
            # СИГНАЛ 8: Momentum (ROC - Rate of Change)
            if len(df) >= 10:
                roc_period = 10
                roc = ((df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period) * 100)
                current_roc = roc.iloc[-1]
                
                if current_roc > 3:  # Сильный рост
                    signals.append(("buy", 0.6))
                    reasons.append(f"Momentum рост {current_roc:.1f}%")
                elif current_roc < -3:  # Сильное падение
                    signals.append(("sell", 0.6))
                    reasons.append(f"Momentum падение {current_roc:.1f}%")
            
            # Анализ сигналов
            if not signals:
                return {"signal": None, "confidence": 0, "reason": "Нет четких сигналов"}
            
            # Подсчет
            buy_signals = [conf for sig, conf in signals if sig == "buy"]
            sell_signals = [conf for sig, conf in signals if sig == "sell"]
            
            buy_score = sum(buy_signals)
            sell_score = sum(sell_signals)
            
            # DEBUG: Логируем детали расчета
            logger.debug(f"DEBUG: signals={len(signals)}, buy_score={buy_score:.2f}, sell_score={sell_score:.2f}")
            
            # Определяем направление с НАКОПЛЕНИЕМ СЛАБЫХ СИГНАЛОВ
            if buy_score > sell_score and buy_score >= 1.5:
                signal = "buy"
                confidence = min(100, buy_score * 50)  # Увеличил с 40 до 50
                
                # БОНУС: Если много слабых индикаторов в одну сторону
                if len(buy_signals) >= 3 and buy_score >= 1.8:
                    confidence = min(100, confidence * 1.3)  # +30% бонус
                    
            elif sell_score > buy_score and sell_score >= 1.5:
                signal = "sell" 
                confidence = min(100, sell_score * 50)  # Увеличил с 40 до 50
                
                # БОНУС: Если много слабых индикаторов в одну сторону
                if len(sell_signals) >= 3 and sell_score >= 1.8:
                    confidence = min(100, confidence * 1.2)  # +20% бонус
                    
            else:
                signal = None
                confidence = 0
            
            # Фильтр: минимальная уверенность
            if confidence < Config.MIN_CONFIDENCE_PERCENT:
                return {
                    "signal": None,
                    "confidence": confidence,
                    "reason": f"Уверенность {confidence:.0f}% < {Config.MIN_CONFIDENCE_PERCENT}%"
                }
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reason": "; ".join(reasons),
                "indicators": {
                    "rsi": float(rsi),
                    "macd": float(current['macd']),
                    "macd_signal": float(current['macd_signal']),
                    "bb_position": "above" if current['close'] > current['bb_upper'] else "below" if current['close'] < current['bb_lower'] else "middle",
                    "ema_trend": "bullish" if current['ema_20'] > current['ema_50'] else "bearish",
                    "atr": float(current['atr'])
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return {"signal": None, "confidence": 0, "reason": f"Ошибка: {e}"}


# Глобальный экземпляр
signal_analyzer = SignalAnalyzer()


