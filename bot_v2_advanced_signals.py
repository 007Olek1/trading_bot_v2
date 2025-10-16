"""
📊 ПРОДВИНУТАЯ СИСТЕМА СИГНАЛОВ V3.0
✅ Уровни поддержки/сопротивления
✅ Объемный анализ (Volume Profile, OBV)
✅ Улучшенная детекция SHORT сигналов
✅ Множественные TP уровни
✅ Trailing Stop
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from bot_v2_config import Config

logger = logging.getLogger(__name__)


class AdvancedSignalAnalyzer:
    """Продвинутый анализатор торговых сигналов"""
    
    def __init__(self):
        self.min_touches_for_level = 2  # Минимум касаний для уровня
        self.level_tolerance = 0.002  # 0.2% толерантность для уровней
    
    @staticmethod
    def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Расчет продвинутых технических индикаторов"""
        try:
            # Базовые индикаторы
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
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean() if len(df) >= 200 else df['close'].ewm(span=len(df)//2).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
            
            # Volume MA
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ma_50'] = df['volume'].rolling(window=50).mean()
            
            # OBV (On Balance Volume)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            
            # Stochastic
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # ADX (Average Directional Index) - Сила тренда
            df['tr'] = ranges.max(axis=1)
            
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr14 = df['tr'].rolling(window=14).sum()
            plus_di = 100 * (plus_dm.rolling(window=14).sum() / tr14)
            minus_di = 100 * (minus_dm.rolling(window=14).sum() / tr14)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df
    
    def find_support_resistance_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, List[float]]:
        """
        Поиск уровней поддержки и сопротивления
        
        Возвращает:
            {
                'support': [уровень1, уровень2, ...],
                'resistance': [уровень1, уровень2, ...]
            }
        """
        try:
            if len(df) < lookback:
                lookback = len(df)
            
            df_recent = df.tail(lookback)
            
            # Находим локальные максимумы и минимумы
            highs = df_recent['high'].values
            lows = df_recent['low'].values
            closes = df_recent['close'].values
            
            # Локальные максимумы (сопротивление)
            resistance_levels = []
            for i in range(2, len(highs) - 2):
                if highs[i] == max(highs[i-2:i+3]):
                    resistance_levels.append(highs[i])
            
            # Локальные минимумы (поддержка)
            support_levels = []
            for i in range(2, len(lows) - 2):
                if lows[i] == min(lows[i-2:i+3]):
                    support_levels.append(lows[i])
            
            # Кластеризация уровней (объединяем близкие уровни)
            resistance_levels = self._cluster_levels(resistance_levels)
            support_levels = self._cluster_levels(support_levels)
            
            # Сортировка
            resistance_levels.sort(reverse=True)
            support_levels.sort()
            
            return {
                'support': support_levels[:5],  # Топ 5 уровней поддержки
                'resistance': resistance_levels[:5]  # Топ 5 уровней сопротивления
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска уровней: {e}")
            return {'support': [], 'resistance': []}
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Объединение близких уровней"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= self.level_tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered
    
    def calculate_tp_levels(
        self,
        entry_price: float,
        side: str,
        atr: float,
        resistance_levels: List[float],
        support_levels: List[float]
    ) -> List[float]:
        """
        Расчет множественных TP уровней как в примере
        
        Args:
            entry_price: Цена входа
            side: 'buy' или 'sell'
            atr: ATR значение
            resistance_levels: Уровни сопротивления
            support_levels: Уровни поддержки
        
        Returns:
            [TP1, TP2, TP3, TP4, TP5]
        """
        tp_levels = []
        
        if side == 'buy':
            # Для LONG ищем уровни сопротивления выше входа
            nearby_resistances = [r for r in resistance_levels if r > entry_price]
            
            if nearby_resistances:
                # Используем ближайшие уровни сопротивления
                tp_levels = nearby_resistances[:5]
            
            # Дополняем ATR-based уровнями
            while len(tp_levels) < 5:
                multiplier = len(tp_levels) + 1
                tp_levels.append(entry_price + (atr * multiplier * 1.5))
        
        else:  # sell (SHORT)
            # Для SHORT ищем уровни поддержки ниже входа
            nearby_supports = [s for s in support_levels if s < entry_price]
            nearby_supports.sort(reverse=True)
            
            if nearby_supports:
                tp_levels = nearby_supports[:5]
            
            # Дополняем ATR-based уровнями
            while len(tp_levels) < 5:
                multiplier = len(tp_levels) + 1
                tp_levels.append(entry_price - (atr * multiplier * 1.5))
        
        # Сортируем по удалению от входа
        if side == 'buy':
            tp_levels.sort()
        else:
            tp_levels.sort(reverse=True)
        
        return tp_levels[:5]
    
    def analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Объемный анализ"""
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            volume_analysis = {
                'volume_surge': current['volume'] > current['volume_ma_20'] * 1.5,
                'volume_trend': 'increasing' if current['volume_ma_20'] > current['volume_ma_50'] else 'decreasing',
                'obv_bullish': current['obv'] > current['obv_ema'],
                'obv_divergence': None
            }
            
            # OBV дивергенция
            price_trend = (df['close'].iloc[-5:].diff().sum() > 0)
            obv_trend = (df['obv'].iloc[-5:].diff().sum() > 0)
            
            if price_trend and not obv_trend:
                volume_analysis['obv_divergence'] = 'bearish'  # Цена растет, OBV падает
            elif not price_trend and obv_trend:
                volume_analysis['obv_divergence'] = 'bullish'  # Цена падает, OBV растет
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа объема: {e}")
            return {}
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        ГЛАВНЫЙ МЕТОД АНАЛИЗА
        
        Returns:
            {
                "signal": "buy" | "sell" | None,
                "confidence": 0-100,
                "reason": str,
                "entry_price": float,
                "sl_price": float,
                "tp_levels": [TP1, TP2, TP3, TP4, TP5],
                "indicators": dict,
                "levels": dict
            }
        """
        try:
            if len(df) < 50:
                return {"signal": None, "confidence": 0, "reason": "Недостаточно данных"}
            
            # Рассчитываем индикаторы
            df = self.calculate_advanced_indicators(df)
            
            # Находим уровни поддержки/сопротивления
            levels = self.find_support_resistance_levels(df)
            
            # Объемный анализ
            volume_analysis = self.analyze_volume(df)
            
            # Последние значения
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Текущая цена
            current_price = current['close']
            
            # Собираем сигналы
            signals = []
            reasons = []
            
            # === СИГНАЛЫ НА ПОКУПКУ (LONG) ===
            
            # 1. RSI перепроданность
            if current['rsi'] < 35:
                signals.append(("buy", 15))
                reasons.append(f"RSI перепроданность ({current['rsi']:.1f})")
            
            # 2. MACD бычий кроссовер
            if prev['macd'] < prev['macd_signal'] and current['macd'] > current['macd_signal']:
                signals.append(("buy", 20))
                reasons.append("MACD бычий кроссовер")
            
            # 3. Цена у уровня поддержки
            for support in levels['support']:
                if abs(current_price - support) / support <= 0.005:  # В пределах 0.5%
                    signals.append(("buy", 25))
                    reasons.append(f"Цена у поддержки ${support:.4f}")
                    break
            
            # 4. EMA тренд бычий
            if current['ema_9'] > current['ema_20'] > current['ema_50']:
                signals.append(("buy", 15))
                reasons.append("Бычий EMA тренд")
            
            # 5. Сильный объем + рост
            if volume_analysis.get('volume_surge') and current['close'] > prev['close']:
                signals.append(("buy", 15))
                reasons.append("Всплеск объема + рост")
            
            # 6. OBV бычий
            if volume_analysis.get('obv_bullish'):
                signals.append(("buy", 10))
                reasons.append("OBV бычий")
            
            # 7. Stochastic перепроданность
            if current['stoch_k'] < 20:
                signals.append(("buy", 10))
                reasons.append("Stochastic перепроданность")
            
            # 8. ADX сильный тренд
            if current['adx'] > 25 and current['plus_di'] > current['minus_di']:
                signals.append(("buy", 15))
                reasons.append(f"Сильный бычий тренд (ADX {current['adx']:.1f})")
            
            # === СИГНАЛЫ НА ПРОДАЖУ (SHORT) ===
            
            # 1. RSI перекупленность
            if current['rsi'] > 65:
                signals.append(("sell", 15))
                reasons.append(f"RSI перекупленность ({current['rsi']:.1f})")
            
            # 2. MACD медвежий кроссовер
            if prev['macd'] > prev['macd_signal'] and current['macd'] < current['macd_signal']:
                signals.append(("sell", 20))
                reasons.append("MACD медвежий кроссовер")
            
            # 3. Цена у уровня сопротивления
            for resistance in levels['resistance']:
                if abs(current_price - resistance) / resistance <= 0.005:  # В пределах 0.5%
                    signals.append(("sell", 25))
                    reasons.append(f"Цена у сопротивления ${resistance:.4f}")
                    break
            
            # 4. EMA тренд медвежий
            if current['ema_9'] < current['ema_20'] < current['ema_50']:
                signals.append(("sell", 15))
                reasons.append("Медвежий EMA тренд")
            
            # 5. Сильный объем + падение
            if volume_analysis.get('volume_surge') and current['close'] < prev['close']:
                signals.append(("sell", 15))
                reasons.append("Всплеск объема + падение")
            
            # 6. OBV медвежий
            if not volume_analysis.get('obv_bullish'):
                signals.append(("sell", 10))
                reasons.append("OBV медвежий")
            
            # 7. Stochastic перекупленность
            if current['stoch_k'] > 80:
                signals.append(("sell", 10))
                reasons.append("Stochastic перекупленность")
            
            # 8. ADX сильный медвежий тренд
            if current['adx'] > 25 and current['minus_di'] > current['plus_di']:
                signals.append(("sell", 15))
                reasons.append(f"Сильный медвежий тренд (ADX {current['adx']:.1f})")
            
            # 9. Медвежья OBV дивергенция
            if volume_analysis.get('obv_divergence') == 'bearish':
                signals.append(("sell", 20))
                reasons.append("Медвежья OBV дивергенция")
            
            # 10. Бычья OBV дивергенция
            if volume_analysis.get('obv_divergence') == 'bullish':
                signals.append(("buy", 20))
                reasons.append("Бычья OBV дивергенция")
            
            # Анализ сигналов
            if not signals:
                return {"signal": None, "confidence": 0, "reason": "Нет четких сигналов"}
            
            # Подсчет
            buy_signals = [conf for sig, conf in signals if sig == "buy"]
            sell_signals = [conf for sig, conf in signals if sig == "sell"]
            
            buy_score = sum(buy_signals)
            sell_score = sum(sell_signals)
            
            # Определяем направление
            if buy_score > sell_score and buy_score >= 40:
                signal = "buy"
                confidence = min(100, buy_score)
                entry_price = current_price
                sl_price = current_price * (1 - Config.MAX_LOSS_PER_TRADE_PERCENT / 100 / Config.LEVERAGE)
                tp_levels = self.calculate_tp_levels(entry_price, 'buy', current['atr'], levels['resistance'], levels['support'])
                
            elif sell_score > buy_score and sell_score >= 40:
                signal = "sell"
                confidence = min(100, sell_score)
                entry_price = current_price
                sl_price = current_price * (1 + Config.MAX_LOSS_PER_TRADE_PERCENT / 100 / Config.LEVERAGE)
                tp_levels = self.calculate_tp_levels(entry_price, 'sell', current['atr'], levels['resistance'], levels['support'])
                
            else:
                return {
                    "signal": None,
                    "confidence": max(buy_score, sell_score),
                    "reason": f"Недостаточная уверенность (BUY: {buy_score}, SELL: {sell_score})"
                }
            
            # Фильтр: минимальная уверенность
            if confidence < Config.MIN_CONFIDENCE_PERCENT:
                return {
                    "signal": None,
                    "confidence": confidence,
                    "reason": f"Уверенность {confidence:.0f}% < {Config.MIN_CONFIDENCE_PERCENT}%"
                }
            
            # Формируем итоговый отчет
            buy_reasons = [r for s, r in zip([s for s, _ in signals], reasons) if s == "buy"]
            sell_reasons = [r for s, r in zip([s for s, _ in signals], reasons) if s == "sell"]
            
            final_reasons = buy_reasons if signal == "buy" else sell_reasons
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reason": "; ".join(final_reasons),
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_levels": tp_levels,
                "indicators": {
                    "rsi": float(current['rsi']),
                    "macd": float(current['macd']),
                    "macd_signal": float(current['macd_signal']),
                    "adx": float(current['adx']),
                    "stoch_k": float(current['stoch_k']),
                    "ema_9": float(current['ema_9']),
                    "ema_20": float(current['ema_20']),
                    "ema_50": float(current['ema_50']),
                    "atr": float(current['atr']),
                    "volume_surge": volume_analysis.get('volume_surge', False),
                    "obv_divergence": volume_analysis.get('obv_divergence')
                },
                "levels": {
                    "support": [float(s) for s in levels['support']],
                    "resistance": [float(r) for r in levels['resistance']]
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return {"signal": None, "confidence": 0, "reason": f"Ошибка: {e}"}


# Глобальный экземпляр
advanced_signal_analyzer = AdvancedSignalAnalyzer()

