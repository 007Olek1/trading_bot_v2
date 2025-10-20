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
            # ФИЛЬТР 2: Проверяем минимальный объем торгов
            ticker = await exchange_manager.exchange.fetch_ticker(symbol)
            if not ticker or ticker.get('quoteVolume', 0) < 1000000:  # Минимум $1M объем
                logger.debug(f"🚫 Исключен: {symbol} (низкий объем: ${ticker.get('quoteVolume', 0):,.0f})")
                return {"volatility_score": 0, "trend_score": 0, "volume_score": 0}
            
            # ФИЛЬТР 3: Проверяем минимальную цену (исключаем слишком дешевые монеты)
            current_price = ticker.get('last', 0)
            if current_price < 0.01:  # Минимум $0.01
                logger.debug(f"🚫 Исключен: {symbol} (слишком дешевый: ${current_price:.6f})")
                return {"volatility_score": 0, "trend_score": 0, "volume_score": 0}
            
            # ФИЛЬТР 4: Проверяем спред (разница между bid/ask)
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            if bid > 0 and ask > 0:
                spread_pct = ((ask - bid) / bid) * 100
                if spread_pct > 2.0:  # Спред больше 2%
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
    
    async def get_volatile_symbols(self, exchange_manager, top_n: int = 50) -> List[Dict[str, Any]]:
        """Получить волатильные символы с анализом - теперь анализирует ВСЕ доступные монеты"""
        try:
            # Получаем РАСШИРЕННЫЙ список символов для максимального охвата
            base_symbols = await exchange_manager.get_top_volume_symbols(top_n=300)  # Увеличено до 300
            
            if not base_symbols:
                logger.warning("⚠️ Не удалось получить базовые символы")
                return []
            
            # ФИЛЬТР 1: Исключаем малоликвидные и проблемные монеты
            filtered_symbols = self._filter_problematic_symbols(base_symbols)
            logger.info(f"🔍 После фильтрации: {len(filtered_symbols)} символов (было {len(base_symbols)})")
            
            logger.info(f"🔍 Анализирую волатильность ВСЕХ {len(filtered_symbols)} доступных символов...")
            
            # Анализируем волатильность для ВСЕХ отфильтрованных символов (не ограничиваем 150)
            analysis_tasks = []
            for symbol in filtered_symbols:  # Убрали ограничение [:150]
                task = self.volatility_analyzer.analyze_symbol_volatility(symbol, exchange_manager)
                analysis_tasks.append(task)
            
            # Выполняем анализ батчами для контроля нагрузки
            batch_size = 50
            valid_results = []
            
            for i in range(0, len(analysis_tasks), batch_size):
                batch_tasks = analysis_tasks[i:i + batch_size]
                logger.info(f"📊 Анализирую батч {i//batch_size + 1}/{(len(analysis_tasks) + batch_size - 1)//batch_size} ({len(batch_tasks)} символов)...")
                
                # Выполняем батч параллельно
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Фильтруем результаты батча
                for result in batch_results:
                    if isinstance(result, dict) and result.get('total_score', 0) > 0:
                        valid_results.append(result)
                
                # Небольшая пауза между батчами
                await asyncio.sleep(0.2)
            
            # Сортируем по общему скору
            valid_results.sort(key=lambda x: x['total_score'], reverse=True)
            
            # Берем топ N (но теперь из ВСЕХ доступных монет)
            top_volatile = valid_results[:top_n]
            
            logger.info(f"📊 Найдено {len(top_volatile)} лучших волатильных символов из {len(valid_results)} проанализированных")
            
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
    
    def has_cached_symbols(self) -> bool:
        """Проверить наличие кэшированных символов"""
        return len(self.cached_symbols) > 0 and self.is_cache_valid()
    
    def is_cache_valid(self) -> bool:
        """Проверить валидность кэша"""
        if not self.last_analysis_time:
            return False
        
        time_diff = (datetime.now() - self.last_analysis_time).total_seconds()
        return time_diff < 1800  # 30 минут
    
    def _filter_problematic_symbols(self, symbols: List[str]) -> List[str]:
        """Фильтрация малоликвидных и проблемных символов"""
        try:
            # Расширенный список исключений - малоликвидные и проблемные монеты
            excluded_symbols = {
                # Малоликвидные токены
                'XPIN/USDT:USDT', 'KGEN/USDT:USDT', 'TWT/USDT:USDT', 'IN/USDT:USDT',
                '1000PEPE/USDT:USDT', '1000SHIB/USDT:USDT', '1000FLOKI/USDT:USDT',
                '1000BONK/USDT:USDT', '1000WIF/USDT:USDT', '1000MEME/USDT:USDT',
                
                # Проблемные символы с низкой ликвидностью
                'GALA/USDT:USDT', 'SAND/USDT:USDT', 'MANA/USDT:USDT', 'AXS/USDT:USDT',
                'IMX/USDT:USDT', 'APE/USDT:USDT', 'GMT/USDT:USDT', 'GAL/USDT:USDT',
                
                # Слишком волатильные мемкоины (оставляем только самые проблемные)
                'PEPE/USDT:USDT', 'FLOKI/USDT:USDT', 'BONK/USDT:USDT', 'WIF/USDT:USDT', 
                'MEME/USDT:USDT', 'BABYDOGE/USDT:USDT', 'ELON/USDT:USDT', 'AKITA/USDT:USDT',
                
                # Проблемные DeFi токены (убираем популярные вроде AAVE, UNI)
                'CRV/USDT:USDT', 'SNX/USDT:USDT', 'COMP/USDT:USDT', 'MKR/USDT:USDT',
                'SUSHI/USDT:USDT', '1INCH/USDT:USDT', 'YFI/USDT:USDT', 'BAL/USDT:USDT',
                
                # Низколиквидные альткоины
                'STORJ/USDT:USDT', 'ANKR/USDT:USDT', 'BAT/USDT:USDT', 'ZRX/USDT:USDT',
                'KNC/USDT:USDT', 'REN/USDT:USDT', 'LRC/USDT:USDT', 'OMG/USDT:USDT',
                
                # Дополнительные малоликвидные
                'DENT/USDT:USDT', 'RSR/USDT:USDT', 'SKL/USDT:USDT', 'CELR/USDT:USDT',
                'CTK/USDT:USDT', 'ALPHA/USDT:USDT', 'BETA/USDT:USDT', 'TLM/USDT:USDT',
                'SLP/USDT:USDT', 'PYR/USDT:USDT', 'GHST/USDT:USDT', 'SUPER/USDT:USDT',
                
                # Стейблкоины и wrapped токены (не нужны для торговли)
                'USDC/USDT:USDT', 'BUSD/USDT:USDT', 'DAI/USDT:USDT', 'TUSD/USDT:USDT',
                'WBTC/USDT:USDT', 'WETH/USDT:USDT', 'stETH/USDT:USDT'
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
            logger.info(f"✅ Прошло фильтрацию: {len(filtered)}")
            
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Ошибка фильтрации символов: {e}")
            return symbols  # Возвращаем исходный список при ошибке


# Глобальные экземпляры
volatility_analyzer = VolatilityAnalyzer()
enhanced_symbol_selector = EnhancedSymbolSelector()
