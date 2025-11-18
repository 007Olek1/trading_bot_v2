"""
1️⃣ COMPREHENSIVE BOT TESTING WORKFLOW
Полная валидация всех компонентов: индикаторы, API, стратегии
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from indicators import MarketIndicators
from strategies import TrendVolumeStrategy, ManipulationDetector, GlobalTrendAnalyzer
import config


class IndicatorValidator:
    """Валидация индикаторов"""
    
    def __init__(self):
        self.indicators = MarketIndicators(config.INDICATOR_PARAMS)
        self.results = []
    
    def generate_test_data(self, length: int = 200) -> pd.DataFrame:
        """Генерация тестовых данных"""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1H')
        
        # Генерируем реалистичные OHLCV данные
        base_price = 100.0
        data = []
        
        for i in range(length):
            # Добавляем тренд и случайность
            trend = i * 0.05
            noise = np.random.randn() * 2
            
            close = base_price + trend + noise
            high = close + abs(np.random.randn() * 1)
            low = close - abs(np.random.randn() * 1)
            open_price = close + np.random.randn() * 0.5
            volume = 1000000 + np.random.randint(-100000, 100000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_ema_calculation(self) -> Dict:
        """Тест расчёта EMA"""
        print("🔍 Тестирование EMA...")
        
        df = self.generate_test_data()
        
        try:
            ema_short = self.indicators.calculate_ema(df['close'], config.INDICATOR_PARAMS['ema_short'])
            ema_medium = self.indicators.calculate_ema(df['close'], config.INDICATOR_PARAMS['ema_medium'])
            ema_long = self.indicators.calculate_ema(df['close'], config.INDICATOR_PARAMS['ema_long'])
            
            # Проверки
            assert len(ema_short) == len(df), "EMA short: неверная длина"
            assert len(ema_medium) == len(df), "EMA medium: неверная длина"
            assert len(ema_long) == len(df), "EMA long: неверная длина"
            assert not ema_short.isna().all(), "EMA short: все значения NaN"
            assert not ema_medium.isna().all(), "EMA medium: все значения NaN"
            assert not ema_long.isna().all(), "EMA long: все значения NaN"
            
            print("  ✅ EMA расчёт корректен")
            return {'status': 'PASS', 'indicator': 'EMA', 'message': 'Все проверки пройдены'}
        
        except Exception as e:
            print(f"  ❌ EMA ошибка: {e}")
            return {'status': 'FAIL', 'indicator': 'EMA', 'message': str(e)}
    
    def test_rsi_calculation(self) -> Dict:
        """Тест расчёта RSI"""
        print("🔍 Тестирование RSI...")
        
        df = self.generate_test_data()
        
        try:
            rsi = self.indicators.calculate_rsi(df['close'], config.INDICATOR_PARAMS['rsi_period'])
            
            # Проверки
            assert len(rsi) == len(df), "RSI: неверная длина"
            assert not rsi.isna().all(), "RSI: все значения NaN"
            
            # RSI должен быть в диапазоне 0-100
            valid_rsi = rsi.dropna()
            assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI: значения вне диапазона 0-100"
            
            print("  ✅ RSI расчёт корректен")
            return {'status': 'PASS', 'indicator': 'RSI', 'message': 'Все проверки пройдены'}
        
        except Exception as e:
            print(f"  ❌ RSI ошибка: {e}")
            return {'status': 'FAIL', 'indicator': 'RSI', 'message': str(e)}
    
    def test_macd_calculation(self) -> Dict:
        """Тест расчёта MACD"""
        print("🔍 Тестирование MACD...")
        
        df = self.generate_test_data()
        
        try:
            macd = self.indicators.calculate_macd(
                df['close'],
                config.INDICATOR_PARAMS['macd_fast'],
                config.INDICATOR_PARAMS['macd_slow'],
                config.INDICATOR_PARAMS['macd_signal']
            )
            
            # Проверки
            assert 'macd' in macd, "MACD: отсутствует macd"
            assert 'signal' in macd, "MACD: отсутствует signal"
            assert 'histogram' in macd, "MACD: отсутствует histogram"
            assert len(macd['macd']) == len(df), "MACD: неверная длина"
            
            print("  ✅ MACD расчёт корректен")
            return {'status': 'PASS', 'indicator': 'MACD', 'message': 'Все проверки пройдены'}
        
        except Exception as e:
            print(f"  ❌ MACD ошибка: {e}")
            return {'status': 'FAIL', 'indicator': 'MACD', 'message': str(e)}
    
    def test_bollinger_bands(self) -> Dict:
        """Тест расчёта Bollinger Bands"""
        print("🔍 Тестирование Bollinger Bands...")
        
        df = self.generate_test_data()
        
        try:
            bb = self.indicators.calculate_bollinger_bands(
                df['close'],
                config.INDICATOR_PARAMS['bb_period'],
                config.INDICATOR_PARAMS['bb_std']
            )
            
            # Проверки
            assert 'upper' in bb, "BB: отсутствует upper"
            assert 'middle' in bb, "BB: отсутствует middle"
            assert 'lower' in bb, "BB: отсутствует lower"
            
            # Upper должен быть выше Lower
            valid_data = ~(bb['upper'].isna() | bb['lower'].isna())
            assert (bb['upper'][valid_data] >= bb['lower'][valid_data]).all(), "BB: upper < lower"
            
            print("  ✅ Bollinger Bands расчёт корректен")
            return {'status': 'PASS', 'indicator': 'Bollinger Bands', 'message': 'Все проверки пройдены'}
        
        except Exception as e:
            print(f"  ❌ Bollinger Bands ошибка: {e}")
            return {'status': 'FAIL', 'indicator': 'Bollinger Bands', 'message': str(e)}
    
    def test_adx_calculation(self) -> Dict:
        """Тест расчёта ADX"""
        print("🔍 Тестирование ADX...")
        
        df = self.generate_test_data()
        
        try:
            adx = self.indicators.calculate_adx(df, config.INDICATOR_PARAMS['adx_period'])
            
            # Проверки
            assert 'adx' in adx, "ADX: отсутствует adx"
            assert 'di_plus' in adx, "ADX: отсутствует di_plus"
            assert 'di_minus' in adx, "ADX: отсутствует di_minus"
            
            # ADX должен быть >= 0
            valid_adx = adx['adx'].dropna()
            assert (valid_adx >= 0).all(), "ADX: отрицательные значения"
            
            print("  ✅ ADX расчёт корректен")
            return {'status': 'PASS', 'indicator': 'ADX', 'message': 'Все проверки пройдены'}
        
        except Exception as e:
            print(f"  ❌ ADX ошибка: {e}")
            return {'status': 'FAIL', 'indicator': 'ADX', 'message': str(e)}
    
    def test_all_indicators(self) -> Dict:
        """Тест всех индикаторов вместе"""
        print("🔍 Тестирование всех индикаторов...")
        
        df = self.generate_test_data()
        
        try:
            indicators = self.indicators.calculate_all(df)
            
            # Проверяем наличие всех ключевых индикаторов
            required_indicators = [
                'ema_short', 'ema_medium', 'ema_long',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower',
                'adx', 'di_plus', 'di_minus',
                'atr', 'volume_sma', 'volume_ratio'
            ]
            
            for ind in required_indicators:
                assert ind in indicators, f"Отсутствует индикатор: {ind}"
            
            print("  ✅ Все индикаторы рассчитаны корректно")
            return {'status': 'PASS', 'indicator': 'ALL', 'message': 'Все индикаторы работают'}
        
        except Exception as e:
            print(f"  ❌ Ошибка расчёта всех индикаторов: {e}")
            return {'status': 'FAIL', 'indicator': 'ALL', 'message': str(e)}
    
    def run_all_tests(self) -> List[Dict]:
        """Запуск всех тестов"""
        print("\n" + "="*70)
        print("1️⃣ COMPREHENSIVE BOT TESTING WORKFLOW")
        print("="*70 + "\n")
        
        tests = [
            self.test_ema_calculation,
            self.test_rsi_calculation,
            self.test_macd_calculation,
            self.test_bollinger_bands,
            self.test_adx_calculation,
            self.test_all_indicators
        ]
        
        results = []
        for test in tests:
            result = test()
            results.append(result)
        
        # Итоговый отчёт
        print("\n" + "="*70)
        print("📊 ИТОГОВЫЙ ОТЧЁТ")
        print("="*70)
        
        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')
        
        print(f"✅ Пройдено: {passed}/{len(results)}")
        print(f"❌ Провалено: {failed}/{len(results)}")
        
        if failed > 0:
            print("\n⚠️ ОШИБКИ:")
            for r in results:
                if r['status'] == 'FAIL':
                    print(f"  ❌ {r['indicator']}: {r['message']}")
        else:
            print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        
        print("="*70 + "\n")
        
        return results


class StrategyValidator:
    """Валидация стратегий"""
    
    def __init__(self):
        self.trend_volume = TrendVolumeStrategy(config.INDICATOR_PARAMS)
        self.manipulation = ManipulationDetector()
        self.global_trend = GlobalTrendAnalyzer()
        self.indicators = MarketIndicators(config.INDICATOR_PARAMS)
    
    def generate_test_data(self, trend: str = 'bullish', length: int = 200) -> pd.DataFrame:
        """Генерация тестовых данных с заданным трендом"""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1H')
        
        base_price = 100.0
        data = []
        
        for i in range(length):
            if trend == 'bullish':
                trend_component = i * 0.1  # Восходящий тренд
            elif trend == 'bearish':
                trend_component = -i * 0.1  # Нисходящий тренд
            else:
                trend_component = 0  # Флэт
            
            noise = np.random.randn() * 1
            
            close = base_price + trend_component + noise
            high = close + abs(np.random.randn() * 0.5)
            low = close - abs(np.random.randn() * 0.5)
            open_price = close + np.random.randn() * 0.3
            volume = 1000000 + np.random.randint(-100000, 100000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_trend_volume_strategy(self) -> Dict:
        """Тест стратегии Тренд + Объём + Bollinger"""
        print("🔍 Тестирование Trend + Volume + Bollinger стратегии...")
        
        try:
            # Тест на бычьем тренде
            df_bull = self.generate_test_data(trend='bullish')
            indicators_bull = self.indicators.calculate_all(df_bull)
            signal_bull = self.trend_volume.analyze(df_bull, indicators_bull)
            
            assert signal_bull is not None, "Стратегия не вернула сигнал"
            assert 'direction' in signal_bull, "Отсутствует direction"
            assert 'confidence' in signal_bull, "Отсутствует confidence"
            
            # Тест на медвежьем тренде
            df_bear = self.generate_test_data(trend='bearish')
            indicators_bear = self.indicators.calculate_all(df_bear)
            signal_bear = self.trend_volume.analyze(df_bear, indicators_bear)
            
            print("  ✅ Trend + Volume + Bollinger стратегия работает")
            return {'status': 'PASS', 'strategy': 'Trend + Volume + Bollinger', 'message': 'Корректно'}
        
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            return {'status': 'FAIL', 'strategy': 'Trend + Volume + Bollinger', 'message': str(e)}
    
    def test_manipulation_detector(self) -> Dict:
        """Тест детектора манипуляций"""
        print("🔍 Тестирование Manipulation Detector...")
        
        try:
            df = self.generate_test_data()
            indicators = self.indicators.calculate_all(df)
            signal = self.manipulation.analyze(df, indicators)
            
            assert signal is not None, "Детектор не вернул сигнал"
            assert 'direction' in signal, "Отсутствует direction"
            assert 'confidence' in signal, "Отсутствует confidence"
            
            print("  ✅ Manipulation Detector работает")
            return {'status': 'PASS', 'strategy': 'Manipulation Detector', 'message': 'Корректно'}
        
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            return {'status': 'FAIL', 'strategy': 'Manipulation Detector', 'message': str(e)}
    
    def test_global_trend_analyzer(self) -> Dict:
        """Тест анализатора глобального тренда"""
        print("🔍 Тестирование Global Trend Analyzer...")
        
        try:
            df = self.generate_test_data(trend='bullish', length=300)
            indicators = self.indicators.calculate_all(df)
            signal = self.global_trend.analyze(df, indicators)
            
            assert signal is not None, "Анализатор не вернул сигнал"
            assert 'direction' in signal, "Отсутствует direction"
            assert 'confidence' in signal, "Отсутствует confidence"
            
            print("  ✅ Global Trend Analyzer работает")
            return {'status': 'PASS', 'strategy': 'Global Trend Analyzer', 'message': 'Корректно'}
        
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            return {'status': 'FAIL', 'strategy': 'Global Trend Analyzer', 'message': str(e)}
    
    def run_all_tests(self) -> List[Dict]:
        """Запуск всех тестов стратегий"""
        print("\n" + "="*70)
        print("🎯 ТЕСТИРОВАНИЕ СТРАТЕГИЙ")
        print("="*70 + "\n")
        
        tests = [
            self.test_trend_volume_strategy,
            self.test_manipulation_detector,
            self.test_global_trend_analyzer
        ]
        
        results = []
        for test in tests:
            result = test()
            results.append(result)
        
        # Итоговый отчёт
        print("\n" + "="*70)
        print("📊 ИТОГОВЫЙ ОТЧЁТ СТРАТЕГИЙ")
        print("="*70)
        
        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')
        
        print(f"✅ Пройдено: {passed}/{len(results)}")
        print(f"❌ Провалено: {failed}/{len(results)}")
        
        if failed > 0:
            print("\n⚠️ ОШИБКИ:")
            for r in results:
                if r['status'] == 'FAIL':
                    print(f"  ❌ {r['strategy']}: {r['message']}")
        else:
            print("\n🎉 ВСЕ СТРАТЕГИИ РАБОТАЮТ!")
        
        print("="*70 + "\n")
        
        return results


if __name__ == "__main__":
    # Тестирование индикаторов
    indicator_validator = IndicatorValidator()
    indicator_results = indicator_validator.run_all_tests()
    
    # Тестирование стратегий
    strategy_validator = StrategyValidator()
    strategy_results = strategy_validator.run_all_tests()
    
    # Общий итог
    total_passed = sum(1 for r in indicator_results + strategy_results if r['status'] == 'PASS')
    total_tests = len(indicator_results) + len(strategy_results)
    
    print("\n" + "="*70)
    print("🏆 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
    print("="*70)
    print(f"✅ Всего пройдено: {total_passed}/{total_tests}")
    print(f"📊 Процент успеха: {total_passed/total_tests*100:.1f}%")
    print("="*70 + "\n")
