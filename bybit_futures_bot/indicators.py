"""
📊 DISCO57 BOT - ИНДИКАТОРЫ
Технические индикаторы для анализа рынка
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


class MarketIndicators:
    """Класс для расчета всех технических индикаторов"""
    
    def __init__(self, params: Dict[str, any]):
        """
        Args:
            params: Словарь с параметрами индикаторов из config
        """
        self.params = params
    
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Расчет всех индикаторов для датафрейма
        
        Args:
            df: DataFrame с колонками: open, high, low, close, volume
        
        Returns:
            Словарь с рассчитанными индикаторами
        """
        if len(df) < 200:
            return None  # Недостаточно данных
        
        indicators = {}
        
        # ═══════════════════════════════════════════════════════════════
        # ТРЕНД
        # ═══════════════════════════════════════════════════════════════
        ema20 = EMAIndicator(df['close'], window=self.params['ema_short']).ema_indicator()
        ema50 = EMAIndicator(df['close'], window=self.params['ema_medium']).ema_indicator()
        ema200 = EMAIndicator(df['close'], window=self.params['ema_long']).ema_indicator()
        
        indicators['ema20'] = ema20.iloc[-1]
        indicators['ema50'] = ema50.iloc[-1]
        indicators['ema200'] = ema200.iloc[-1]
        indicators['ema20_slope'] = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5] * 100
        indicators['ema50_slope'] = (ema50.iloc[-1] - ema50.iloc[-5]) / ema50.iloc[-5] * 100
        
        # Определение тренда
        if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
            indicators['trend'] = "BULLISH"
        elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
            indicators['trend'] = "BEARISH"
        else:
            indicators['trend'] = "NEUTRAL"
        
        # ADX - сила тренда
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=self.params['adx_period'])
        indicators['adx'] = adx.adx().iloc[-1]
        indicators['adx_plus'] = adx.adx_pos().iloc[-1]
        indicators['adx_minus'] = adx.adx_neg().iloc[-1]
        indicators['adx_strong'] = indicators['adx'] > self.params['adx_min_strength']
        
        # ═══════════════════════════════════════════════════════════════
        # ИМПУЛЬС
        # ═══════════════════════════════════════════════════════════════
        rsi = RSIIndicator(df['close'], window=self.params['rsi_period'])
        indicators['rsi'] = rsi.rsi().iloc[-1]
        indicators['rsi_oversold'] = indicators['rsi'] < self.params['rsi_oversold']
        indicators['rsi_overbought'] = indicators['rsi'] > self.params['rsi_overbought']
        
        # Stochastic RSI
        stoch_rsi = StochRSIIndicator(df['close'], window=self.params['stoch_rsi_period'])
        indicators['stoch_rsi'] = stoch_rsi.stochrsi().iloc[-1]
        indicators['stoch_rsi_k'] = stoch_rsi.stochrsi_k().iloc[-1]
        indicators['stoch_rsi_d'] = stoch_rsi.stochrsi_d().iloc[-1]
        
        # MACD
        macd = MACD(
            df['close'],
            window_fast=self.params['macd_fast'],
            window_slow=self.params['macd_slow'],
            window_sign=self.params['macd_signal']
        )
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_hist'] = macd.macd_diff().iloc[-1]
        indicators['macd_bullish'] = indicators['macd'] > indicators['macd_signal']
        
        # ═══════════════════════════════════════════════════════════════
        # ВОЛАТИЛЬНОСТЬ
        # ═══════════════════════════════════════════════════════════════
        bb = BollingerBands(df['close'], window=self.params['bb_period'], window_dev=self.params['bb_std'])
        indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
        indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
        indicators['bb_width'] = bb.bollinger_wband().iloc[-1]
        
        # Позиция цены относительно BB
        price = df['close'].iloc[-1]
        bb_range = indicators['bb_upper'] - indicators['bb_lower']
        if bb_range > 0:
            indicators['bb_position'] = (price - indicators['bb_lower']) / bb_range
        else:
            indicators['bb_position'] = 0.5
        
        # ATR
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=self.params['atr_period'])
        indicators['atr'] = atr.average_true_range().iloc[-1]
        indicators['atr_percent'] = (indicators['atr'] / price) * 100
        
        # ═══════════════════════════════════════════════════════════════
        # ОБЪЁМ
        # ═══════════════════════════════════════════════════════════════
        obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
        indicators['obv'] = obv.on_balance_volume().iloc[-1]
        indicators['obv_slope'] = (indicators['obv'] - obv.on_balance_volume().iloc[-5]) / abs(obv.on_balance_volume().iloc[-5]) if obv.on_balance_volume().iloc[-5] != 0 else 0
        
        # Средний объем
        volume_sma = df['volume'].rolling(window=self.params['volume_sma_period']).mean()
        indicators['volume_sma'] = volume_sma.iloc[-1]
        indicators['volume_ratio'] = df['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
        
        # ═══════════════════════════════════════════════════════════════
        # ДОПОЛНИТЕЛЬНО
        # ═══════════════════════════════════════════════════════════════
        indicators['price'] = price
        indicators['volume'] = df['volume'].iloc[-1]
        
        return indicators
    
    def generate_signal(self, indicators_multi_tf: Dict[str, Dict]) -> Tuple[str, float, int]:
        """
        Генерация торгового сигнала на основе мультифреймового анализа
        
        Args:
            indicators_multi_tf: Словарь {timeframe: indicators}
        
        Returns:
            (signal, confidence, aligned_timeframes)
            signal: "BUY", "SELL" или "HOLD"
            confidence: уверенность в сигнале (0-100)
            aligned_timeframes: количество подтверждающих таймфреймов
        """
        signals = []
        scores = []
        
        for tf, ind in indicators_multi_tf.items():
            if not ind:
                continue
            
            score = 0.0
            
            # ═══════════════════════════════════════════════════════════
            # АНАЛИЗ ТРЕНДА (вес 40%)
            # ═══════════════════════════════════════════════════════════
            if ind['trend'] == "BULLISH":
                score += 20
            elif ind['trend'] == "BEARISH":
                score -= 20
            
            # ADX - сила тренда
            if ind['adx_strong']:
                if ind['adx_plus'] > ind['adx_minus']:
                    score += 10
                else:
                    score -= 10
            
            # EMA slopes
            if ind['ema20_slope'] > 0 and ind['ema50_slope'] > 0:
                score += 10
            elif ind['ema20_slope'] < 0 and ind['ema50_slope'] < 0:
                score -= 10
            
            # ═══════════════════════════════════════════════════════════
            # АНАЛИЗ ИМПУЛЬСА (вес 30%)
            # ═══════════════════════════════════════════════════════════
            if ind['rsi_oversold']:
                score += 15
            elif ind['rsi_overbought']:
                score -= 15
            
            if ind['macd_bullish']:
                score += 10
            else:
                score -= 10
            
            # Stochastic RSI
            if ind['stoch_rsi_k'] > ind['stoch_rsi_d']:
                score += 5
            else:
                score -= 5
            
            # ═══════════════════════════════════════════════════════════
            # АНАЛИЗ ОБЪЁМА (вес 15%)
            # ═══════════════════════════════════════════════════════════
            if ind['volume_ratio'] > 1.2:  # Объем выше среднего
                if ind['obv_slope'] > 0:
                    score += 10
                else:
                    score -= 5
            
            # ═══════════════════════════════════════════════════════════
            # АНАЛИЗ ВОЛАТИЛЬНОСТИ (вес 15%)
            # ═══════════════════════════════════════════════════════════
            # Bollinger Bands
            if ind['bb_position'] < 0.2:  # Цена у нижней границы
                score += 10
            elif ind['bb_position'] > 0.8:  # Цена у верхней границы
                score -= 10
            
            # Нормализуем score в диапазон -1 .. +1
            normalized_score = max(min(score / 100, 1.0), -1.0)
            
            # Определяем сигнал для таймфрейма
            if normalized_score > 0.3:
                signals.append("BUY")
                scores.append(normalized_score)
            elif normalized_score < -0.3:
                signals.append("SELL")
                scores.append(abs(normalized_score))
            else:
                signals.append("HOLD")
                scores.append(0)
        
        # ═══════════════════════════════════════════════════════════════
        # ИТОГОВЫЙ СИГНАЛ
        # ═══════════════════════════════════════════════════════════════
        if not signals:
            return "HOLD", 0.0, 0
        
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        
        # Проверка согласованности таймфреймов
        aligned_timeframes = max(buy_count, sell_count)
        
        # Средняя уверенность
        avg_confidence = (sum(scores) / len(scores)) * 100 if scores else 0
        
        # Генерация финального сигнала
        if buy_count >= 2 and buy_count > sell_count:
            return "BUY", avg_confidence, aligned_timeframes
        elif sell_count >= 2 and sell_count > buy_count:
            return "SELL", avg_confidence, aligned_timeframes
        else:
            return "HOLD", avg_confidence, aligned_timeframes


def detect_market_mode(indicators: Dict[str, any], params: Dict[str, any]) -> str:
    """
    Определение режима рынка: trending, ranging, volatile
    
    Args:
        indicators: Индикаторы основного таймфрейма
        params: Параметры из config
    
    Returns:
        Режим рынка: "trending", "ranging", "volatile"
    """
    if not indicators:
        return "ranging"
    
    adx = indicators.get('adx', 0)
    atr_percent = indicators.get('atr_percent', 0)
    
    # Флэт
    if adx < 20:
        return "ranging"
    
    # Высокая волатильность
    if atr_percent > 3.0:  # ATR > 3% от цены
        return "volatile"
    
    # Тренд
    if adx > 25:
        return "trending"
    
    return "ranging"


print("✅ Индикаторы Disco57 загружены")

