"""
🎯 АДАПТИВНЫЕ ПАРАМЕТРЫ ПОД ВОЛАТИЛЬНОСТЬ
Автоматическая настройка параметров в зависимости от рыночных условий
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging


class AdaptiveParameters:
    """Адаптивная настройка параметров торговли"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Пороги волатильности (ATR в %)
        self.volatility_thresholds = {
            'very_low': 0.01,   # < 1%
            'low': 0.02,        # 1-2%
            'medium': 0.04,     # 2-4%
            'high': 0.06,       # 4-6%
            'very_high': 0.10,  # > 6%
        }
        
        # Параметры для разных уровней волатильности
        self.params_by_volatility = {
            'very_low': {
                'trailing_sl_callback': 0.8,   # Меньше откат
                'min_confidence': 0.70,         # Выше порог
                'position_multiplier': 1.2,     # Больше размер
            },
            'low': {
                'trailing_sl_callback': 1.0,
                'min_confidence': 0.68,
                'position_multiplier': 1.1,
            },
            'medium': {
                'trailing_sl_callback': 1.5,   # Базовые значения
                'min_confidence': 0.65,
                'position_multiplier': 1.0,
            },
            'high': {
                'trailing_sl_callback': 2.0,   # Больше откат
                'min_confidence': 0.62,         # Ниже порог
                'position_multiplier': 0.9,     # Меньше размер
            },
            'very_high': {
                'trailing_sl_callback': 2.5,
                'min_confidence': 0.60,
                'position_multiplier': 0.8,
            },
        }
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Рассчитывает волатильность (ATR в процентах)
        
        Args:
            df: DataFrame с колонками high, low, close
            
        Returns:
            Волатильность в процентах
        """
        if len(df) < 14:
            return 0.02  # Средняя по умолчанию
        
        # ATR (Average True Range)
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # ATR в процентах от цены
        current_price = close.iloc[-1]
        atr_percent = atr / current_price
        
        return atr_percent
    
    def classify_volatility(self, volatility: float) -> str:
        """
        Классифицирует уровень волатильности
        
        Args:
            volatility: Волатильность в процентах
            
        Returns:
            Уровень: very_low, low, medium, high, very_high
        """
        if volatility < self.volatility_thresholds['very_low']:
            return 'very_low'
        elif volatility < self.volatility_thresholds['low']:
            return 'low'
        elif volatility < self.volatility_thresholds['medium']:
            return 'medium'
        elif volatility < self.volatility_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def get_adaptive_params(self, df: pd.DataFrame) -> Dict:
        """
        Получает адаптивные параметры на основе волатильности
        
        Args:
            df: DataFrame с рыночными данными
            
        Returns:
            Словарь с адаптивными параметрами
        """
        # Рассчитываем волатильность
        volatility = self.calculate_volatility(df)
        
        # Классифицируем
        vol_level = self.classify_volatility(volatility)
        
        # Получаем параметры
        params = self.params_by_volatility[vol_level].copy()
        params['volatility'] = volatility
        params['volatility_level'] = vol_level
        
        self.logger.info(
            f"📊 Волатильность: {volatility*100:.2f}% ({vol_level}) | "
            f"Trailing: {params['trailing_sl_callback']:.1f}% | "
            f"Confidence: {params['min_confidence']:.0%} | "
            f"Size: {params['position_multiplier']:.1f}x"
        )
        
        return params
    
    def adjust_trailing_sl(self, base_callback: float, volatility: float) -> float:
        """
        Корректирует trailing SL callback под волатильность
        
        Args:
            base_callback: Базовый откат (например, 1.5%)
            volatility: Текущая волатильность
            
        Returns:
            Скорректированный откат
        """
        vol_level = self.classify_volatility(volatility)
        multiplier = self.params_by_volatility[vol_level]['trailing_sl_callback'] / 1.5
        
        return base_callback * multiplier
    
    def adjust_confidence_threshold(self, base_confidence: float, volatility: float) -> float:
        """
        Корректирует порог уверенности под волатильность
        
        Args:
            base_confidence: Базовый порог (например, 0.65)
            volatility: Текущая волатильность
            
        Returns:
            Скорректированный порог
        """
        vol_level = self.classify_volatility(volatility)
        
        return self.params_by_volatility[vol_level]['min_confidence']
    
    def adjust_position_size(self, base_size: float, volatility: float) -> float:
        """
        Корректирует размер позиции под волатильность
        
        Args:
            base_size: Базовый размер (например, 1.0 USD)
            volatility: Текущая волатильность
            
        Returns:
            Скорректированный размер
        """
        vol_level = self.classify_volatility(volatility)
        multiplier = self.params_by_volatility[vol_level]['position_multiplier']
        
        return base_size * multiplier
    
    def get_market_regime(self, df: pd.DataFrame) -> Dict:
        """
        Определяет рыночный режим
        
        Args:
            df: DataFrame с рыночными данными
            
        Returns:
            Словарь с информацией о режиме
        """
        if len(df) < 50:
            return {'regime': 'unknown', 'trend_strength': 0}
        
        close = df['close']
        
        # Тренд (EMA 20 vs EMA 50)
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        
        if ema20 > ema50 * 1.02:
            trend = 'uptrend'
            trend_strength = (ema20 - ema50) / ema50
        elif ema20 < ema50 * 0.98:
            trend = 'downtrend'
            trend_strength = (ema50 - ema20) / ema50
        else:
            trend = 'sideways'
            trend_strength = 0
        
        # Волатильность
        volatility = self.calculate_volatility(df)
        vol_level = self.classify_volatility(volatility)
        
        return {
            'regime': trend,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'volatility_level': vol_level,
        }


if __name__ == "__main__":
    # Тест
    print("🧪 Тестирование адаптивных параметров\n")
    
    # Создаём тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='1h')
    
    # Низкая волатильность
    df_low = pd.DataFrame({
        'high': 100 + np.random.randn(100) * 0.5,
        'low': 99 + np.random.randn(100) * 0.5,
        'close': 99.5 + np.random.randn(100) * 0.5,
    }, index=dates)
    
    # Высокая волатильность
    df_high = pd.DataFrame({
        'high': 100 + np.random.randn(100) * 5,
        'low': 95 + np.random.randn(100) * 5,
        'close': 97.5 + np.random.randn(100) * 5,
    }, index=dates)
    
    adapter = AdaptiveParameters()
    
    print("📊 Низкая волатильность:")
    params_low = adapter.get_adaptive_params(df_low)
    print(f"   Trailing SL: {params_low['trailing_sl_callback']}%")
    print(f"   Min Confidence: {params_low['min_confidence']:.0%}")
    print(f"   Position Size: {params_low['position_multiplier']}x")
    print()
    
    print("📊 Высокая волатильность:")
    params_high = adapter.get_adaptive_params(df_high)
    print(f"   Trailing SL: {params_high['trailing_sl_callback']}%")
    print(f"   Min Confidence: {params_high['min_confidence']:.0%}")
    print(f"   Position Size: {params_high['position_multiplier']}x")
    print()
    
    print("✅ Тест завершён!")
