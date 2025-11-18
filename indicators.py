"""
📊 ИНДИКАТОРЫ - Расчёт технических индикаторов
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class MarketIndicators:
    """Класс для расчёта технических индикаторов"""
    
    def __init__(self, params: Dict):
        """Инициализация с параметрами"""
        self.params = params
    
    def calculate_all(self, df: pd.DataFrame) -> Dict:
        """Расчёт всех индикаторов"""
        if df is None or len(df) < 50:
            return {}
        
        indicators = {}
        
        # EMA
        indicators['ema_short'] = self.calculate_ema(df['close'], self.params['ema_short'])
        indicators['ema_medium'] = self.calculate_ema(df['close'], self.params['ema_medium'])
        indicators['ema_long'] = self.calculate_ema(df['close'], self.params['ema_long'])
        
        # RSI
        indicators['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_period'])
        
        # Stochastic RSI
        stoch_rsi = self.calculate_stoch_rsi(df['close'], self.params['stoch_rsi_period'])
        indicators['stoch_rsi_k'] = stoch_rsi['k']
        indicators['stoch_rsi_d'] = stoch_rsi['d']
        
        # MACD
        macd = self.calculate_macd(
            df['close'],
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        indicators['macd'] = macd['macd']
        indicators['macd_signal'] = macd['signal']
        indicators['macd_histogram'] = macd['histogram']
        
        # ADX
        adx = self.calculate_adx(df, self.params['adx_period'])
        indicators['adx'] = adx['adx']
        indicators['di_plus'] = adx['di_plus']
        indicators['di_minus'] = adx['di_minus']
        
        # Bollinger Bands
        bb = self.calculate_bollinger_bands(df['close'], self.params['bb_period'], self.params['bb_std'])
        indicators['bb_upper'] = bb['upper']
        indicators['bb_middle'] = bb['middle']
        indicators['bb_lower'] = bb['lower']
        indicators['bb_width'] = bb['width']
        
        # ATR
        indicators['atr'] = self.calculate_atr(df, self.params['atr_period'])
        
        # Volume
        indicators['volume_sma'] = self.calculate_sma(df['volume'], self.params['volume_sma_period'])
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        return indicators
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Простая скользящая средняя"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Индекс относительной силы (RSI)"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_stoch_rsi(series: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Stochastic RSI"""
        # Сначала рассчитываем RSI
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Затем применяем стохастик к RSI
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        stoch_rsi_k = stoch_rsi.rolling(window=3).mean()  # %K
        stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean()  # %D
        
        return {
            'k': stoch_rsi_k,
            'd': stoch_rsi_d
        }
    
    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD индикатор"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        }
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index (ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=df.index).rolling(window=period).mean()
        minus_dm = pd.Series(minus_dm, index=df.index).rolling(window=period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm / atr)
        minus_di = 100 * (minus_dm / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'di_plus': plus_di,
            'di_minus': minus_di
        }
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle * 100
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width
        }
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
