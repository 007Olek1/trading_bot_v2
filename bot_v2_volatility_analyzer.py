"""
🚀 УЛУЧШЕННЫЙ ПОИСК ВОЛАТИЛЬНЫХ МОНЕТ И АНАЛИЗ ТРЕНДОВ V2.1
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """Анализатор волатильности и трендов"""
    
    def __init__(self):
        self.cache = {}  # Кэш для быстрого доступа
        self.cache_timeout = 300  # 5 минут
    
    async def analyze_symbol_volatility(self, symbol: str, exchange_manager) -> Dict[str, Any]:
        """Анализ волатильности конкретного символа"""
        try:
            # ФИЛЬТР 2: Проверяем минимальный объем торгов (снижено для большего охвата)
            ticker = await exchange_manager.exchange.fetch_ticker(symbol)
            if not ticker or ticker.get('quoteVolume', 0) < 500000:  # Минимум $500K объем (было $1M)
                logger.debug(f"🚫 Исключен: {symbol} (низкий объем: ${ticker.get('quoteVolume', 0):,.0f})")
                return {"volatility_score": 0, "trend_score": 0, "volume_score": 0}
            
            # ФИЛЬТР 3: Проверяем минимальную цену (снижено для большего охвата)
            current_price = ticker.get('last', 0)
            if current_price < 0.001:  # Минимум $0.001 (было $0.01)
                logger.debug(f"🚫 Исключен: {symbol} (слишком дешевый: ${current_price:.6f})")
                return {"volatility_score": 0, "trend_score": 0, "volume_score": 0}
            
            # ФИЛЬТР 4: Проверяем спред (более мягкий фильтр)
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            if bid > 0 and ask > 0:
                spread_pct = ((ask - bid) / bid) * 100
                if spread_pct > 5.0:  # Спред больше 5% (было 2%)
                    logger.debug(f"🚫 Исключен: {symbol} (высокий спред: {spread_pct:.2f}%)")
                    return {"volatility_score": 0, "trend_score": 0, "volume_score": 0}
            
            # Получаем данные за разные периоды
            ohlcv_1h = await exchange_manager.fetch_ohlcv(symbol, "1h", 24)  # 24 часа
            ohlcv_5m = await exchange_manager.fetch_ohlcv(symbol, "5m", 100)  # 8+ часов
            
            if not ohlcv_1h or not ohlcv_5m:
                return {"volatility_score": 0, "trend_score": 0, "volume_score": 0}
            
            # Конвертируем в DataFrame
            df_1h = self._ohlcv_to_dataframe(ohlcv_1h)
            df_5m = self._ohlcv_to_dataframe(ohlcv_5m)
            
            # Анализируем волатильность
            volatility_score = self._calculate_volatility_score(df_5m)
            
            # Анализируем тренд
            trend_score = self._calculate_trend_score(df_1h, df_5m)
            
            # Анализируем объемы
            volume_score = self._calculate_volume_score(df_5m)
            
            # Общий скор
            total_score = (volatility_score * 0.4 + trend_score * 0.4 + volume_score * 0.2)
            
            return {
                "symbol": symbol,
                "volatility_score": volatility_score,
                "trend_score": trend_score,
                "volume_score": volume_score,
                "total_score": total_score,
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа волатильности {symbol}: {e}")
            return {"volatility_score": 0, "trend_score": 0, "volume_score": 0}
    
    def _ohlcv_to_dataframe(self, ohlcv: List) -> pd.DataFrame:
        """Конвертация OHLCV в DataFrame"""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Расчет скор волатильности (0-100)"""
        try:
            if len(df) < 20:
                return 0
            
            # 1. ATR (Average True Range) - базовая волатильность
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # 2. Процентные изменения цены
            price_changes = df['close'].pct_change().abs()
            avg_change = price_changes.rolling(window=20).mean()
            
            # 3. Максимальные движения за период
            max_moves = []
            for i in range(5, len(df)):
                period_high = df['high'].iloc[i-5:i].max()
                period_low = df['low'].iloc[i-5:i].min()
                move_pct = (period_high - period_low) / period_low * 100
                max_moves.append(move_pct)
            
            avg_max_move = np.mean(max_moves) if max_moves else 0
            
            # 4. Волатильность относительно цены
            current_price = df['close'].iloc[-1]
            atr_pct = (atr.iloc[-1] / current_price) * 100
            
            # Комбинированный скор
            volatility_score = min(100, (atr_pct * 10 + avg_change.iloc[-1] * 1000 + avg_max_move * 2))
            
            return max(0, volatility_score)
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета волатильности: {e}")
            return 0
    
    def _calculate_trend_score(self, df_1h: pd.DataFrame, df_5m: pd.DataFrame) -> float:
        """Расчет скор тренда (0-100)"""
        try:
            if len(df_1h) < 10 or len(df_5m) < 20:
                return 0
            
            trend_signals = []
            
            # 1. EMA тренд (1 час)
            ema_12_1h = df_1h['close'].ewm(span=12).mean()
            ema_26_1h = df_1h['close'].ewm(span=26).mean()
            
            if ema_12_1h.iloc[-1] > ema_26_1h.iloc[-1]:
                trend_signals.append(("bullish", 0.8))
            else:
                trend_signals.append(("bearish", 0.8))
            
            # 2. MACD тренд (1 час)
            macd_1h = ema_12_1h - ema_26_1h
            macd_signal_1h = macd_1h.ewm(span=9).mean()
            
            if macd_1h.iloc[-1] > macd_signal_1h.iloc[-1]:
                trend_signals.append(("bullish", 0.7))
            else:
                trend_signals.append(("bearish", 0.7))
            
            # 3. Краткосрочный тренд (5 минут)
            ema_20_5m = df_5m['close'].ewm(span=20).mean()
            ema_50_5m = df_5m['close'].ewm(span=50).mean()
            
            if ema_20_5m.iloc[-1] > ema_50_5m.iloc[-1]:
                trend_signals.append(("bullish", 0.6))
            else:
                trend_signals.append(("bearish", 0.6))
            
            # 4. Momentum (скорость изменения)
            momentum_1h = (df_1h['close'].iloc[-1] - df_1h['close'].iloc[-6]) / df_1h['close'].iloc[-6] * 100
            momentum_5m = (df_5m['close'].iloc[-1] - df_5m['close'].iloc[-12]) / df_5m['close'].iloc[-12] * 100
            
            if momentum_1h > 2:  # Сильный рост за час
                trend_signals.append(("bullish", 0.9))
            elif momentum_1h < -2:  # Сильное падение за час
                trend_signals.append(("bearish", 0.9))
            
            if momentum_5m > 1:  # Рост за час (5-минутные свечи)
                trend_signals.append(("bullish", 0.5))
            elif momentum_5m < -1:
                trend_signals.append(("bearish", 0.5))
            
            # Подсчитываем скор
            bullish_score = sum([score for signal, score in trend_signals if signal == "bullish"])
            bearish_score = sum([score for signal, score in trend_signals if signal == "bearish"])
            
            # Возвращаем скор в зависимости от направления
            if bullish_score > bearish_score:
                return min(100, bullish_score * 20)  # Конвертируем в 0-100
            else:
                return min(100, bearish_score * 20)
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета тренда: {e}")
            return 0
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Расчет скор объемов (0-100)"""
        try:
            if len(df) < 20:
                return 0
            
            # Средний объем за 20 периодов
            avg_volume = df['volume'].rolling(window=20).mean()
            current_volume = df['volume'].iloc[-1]
            
            # Отношение текущего объема к среднему
            volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # Объемный скор
            volume_score = min(100, (volume_ratio - 1) * 50 + 50)  # 1x = 50, 2x = 100, 0.5x = 25
            
            return max(0, volume_score)
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета объемов: {e}")
            return 0


class EnhancedSymbolSelector:
    """Улучшенный селектор символов с анализом волатильности"""
    
    def __init__(self):
        self.volatility_analyzer = VolatilityAnalyzer()
        self.last_analysis_time = None
        self.cached_symbols = []
    
    async def get_volatile_symbols(self, exchange_manager, top_n: int = 200) -> List[Dict[str, Any]]:
        """Получить волатильные символы с анализом - ВСЕ доступные монеты"""
        try:
            # Получаем ВСЕ доступные символы с биржи (не только топ 200)
            logger.info("🔍 Получаем ВСЕ доступные символы с биржи...")
            all_tickers = await exchange_manager.exchange.fetch_tickers()
            
            # Фильтруем только USDT perpetual с минимальным объемом
            usdt_perp_symbols = []
            for symbol, ticker in all_tickers.items():
                if (":USDT" in symbol and 
                    ticker.get("quoteVolume", 0) > 500000 and  # Минимум $500K объем
                    ticker.get("last", 0) > 0.001):  # Минимум $0.001 цена
                    usdt_perp_symbols.append(symbol)
            
            logger.info(f"📊 Найдено {len(usdt_perp_symbols)} USDT perpetual символов с достаточной ликвидностью")
            
            if not usdt_perp_symbols:
                logger.warning("⚠️ Не удалось получить символы с биржи")
                return []
            
            # ФИЛЬТР 1: Исключаем малоликвидные и проблемные монеты
            filtered_symbols = self._filter_problematic_symbols(usdt_perp_symbols)
            logger.info(f"🔍 После фильтрации: {len(filtered_symbols)} символов (было {len(usdt_perp_symbols)})")
            
            logger.info(f"🔍 Анализирую волатильность {len(filtered_symbols)} символов...")
            
            # Анализируем волатильность для каждого символа (все доступные)
            analysis_tasks = []
            for symbol in filtered_symbols:  # Анализируем ВСЕ символы
                task = self.volatility_analyzer.analyze_symbol_volatility(symbol, exchange_manager)
                analysis_tasks.append(task)
            
            # Выполняем анализ параллельно
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Фильтруем результаты
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result.get('total_score', 0) > 0:
                    valid_results.append(result)
            
            # Сортируем по общему скору
            valid_results.sort(key=lambda x: x['total_score'], reverse=True)
            
            # Берем топ N
            top_volatile = valid_results[:top_n]
            
            logger.info(f"📊 Найдено {len(top_volatile)} волатильных символов")
            
            # Логируем топ 10
            for i, symbol_data in enumerate(top_volatile[:10]):
                logger.info(
                    f"🔥 #{i+1} {symbol_data['symbol']}: "
                    f"Vol={symbol_data['volatility_score']:.1f}, "
                    f"Trend={symbol_data['trend_score']:.1f}, "
                    f"Vol={symbol_data['volume_score']:.1f}, "
                    f"Total={symbol_data['total_score']:.1f}"
                )
            
            self.cached_symbols = top_volatile
            self.last_analysis_time = datetime.now()
            
            return top_volatile
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения волатильных символов: {e}")
            return []
    
    def get_cached_symbols(self) -> List[str]:
        """Получить кэшированные символы"""
        return [symbol_data['symbol'] for symbol_data in self.cached_symbols]
    
    def is_cache_valid(self) -> bool:
        """Проверить валидность кэша"""
        if not self.last_analysis_time:
            return False
        
        time_diff = (datetime.now() - self.last_analysis_time).total_seconds()
        return time_diff < 3600  # 1 час (увеличено для большего количества монет)
    
    def _filter_problematic_symbols(self, symbols: List[str]) -> List[str]:
        """Фильтрация только действительно малоликвидных и проблемных символов"""
        try:
            # Минимальный список исключений - только действительно проблемные монеты
            excluded_symbols = {
                # Только самые малоликвидные токены (объем < $100K)
                'XPIN/USDT:USDT', 'KGEN/USDT:USDT', 'TWT/USDT:USDT', 'IN/USDT:USDT',
                
                # Только самые проблемные мемкоины с экстремальной волатильностью
                '1000PEPE/USDT:USDT', '1000SHIB/USDT:USDT', '1000FLOKI/USDT:USDT',
                '1000BONK/USDT:USDT', '1000WIF/USDT:USDT', '1000MEME/USDT:USDT',
                'BABYDOGE/USDT:USDT',
                
                # Только самые низколиквидные DeFi токены
                'REN/USDT:USDT', 'OMG/USDT:USDT'
            }
            
            # Фильтруем символы
            filtered = []
            excluded_count = 0
            
            for symbol in symbols:
                if symbol in excluded_symbols:
                    excluded_count += 1
                    logger.debug(f"🚫 Исключен: {symbol} (малоликвидный)")
                else:
                    filtered.append(symbol)
            
            logger.info(f"🚫 Исключено малоликвидных: {excluded_count}")
            logger.info(f"✅ Прошло фильтрацию: {len(filtered)} (включая популярные монеты)")
            
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Ошибка фильтрации символов: {e}")
            return symbols  # Возвращаем исходный список при ошибке


# Глобальные экземпляры
volatility_analyzer = VolatilityAnalyzer()
enhanced_symbol_selector = EnhancedSymbolSelector()
