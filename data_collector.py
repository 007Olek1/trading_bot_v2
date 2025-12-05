#!/usr/bin/env python3
"""
Data Collector for Disco57 Model
Collects and prepares candle data for training
"""

import logging
import os
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import ccxt.async_support as ccxt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DataCollector:
    """Collector for historical market data"""
    
    def __init__(self, symbols: List[str], timeframe: str = '1m', days: int = 30):
        self.symbols = symbols
        self.timeframe = timeframe
        self.days = days
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.testnet = os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'
        
        if not self.api_key or not self.api_secret:
            raise ValueError("BYBIT_API_KEY и BYBIT_API_SECRET должны быть установлены в .env")
        
        # Initialize CCXT
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 10000,
            }
        })
        
        if self.testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("DataCollector: TESTNET режим")
        else:
            logger.info("DataCollector: LIVE режим")
        
        self.retry_count = 3
        self.retry_delay = 1
    
    async def _retry_request(self, func, *args, **kwargs):
        """Retry request on failure"""
        for attempt in range(self.retry_count):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt < self.retry_count - 1:
                    logger.warning(f"Ошибка API (попытка {attempt + 1}/{self.retry_count}): {e}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Ошибка API после {self.retry_count} попыток: {e}")
                    raise
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                          since: Optional[int] = None, limit: int = 1000) -> List:
        """Fetch OHLCV candles"""
        return await self._retry_request(
            self.exchange.fetch_ohlcv,
            symbol,
            timeframe,
            since=since,
            limit=limit
        )
    
    async def collect_data(self, output_file: str = 'training_data.npz'):
        """
        Collect historical data for all symbols
        
        Args:
            output_file: File to save the processed data
        """
        from datetime import datetime, timedelta
        
        all_data = []
        since = int((datetime.now() - timedelta(days=self.days)).timestamp() * 1000)
        
        logger.info(f"Сбор данных для {len(self.symbols)} символов за последние {self.days} дней")
        
        for i, symbol in enumerate(self.symbols):
            try:
                logger.info(f"Сбор данных для {symbol} ({i+1}/{len(self.symbols)})")
                candles = await self.fetch_ohlcv(symbol, self.timeframe, since=since, limit=10000)
                
                if len(candles) == 0:
                    logger.warning(f"Нет данных для {symbol}")
                    continue
                
                # Process candles into features
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                features = self.process_candles(df)
                
                # Add symbol info for potential filtering
                features['symbol'] = symbol
                all_data.append(features)
                
                logger.info(f"Получено {len(candles)} свечей для {symbol}")
                
                # Avoid rate limiting
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Ошибка сбора данных для {symbol}: {e}")
                continue
        
        # Combine all data
        if not all_data:
            logger.error("Нет данных для сохранения")
            return
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Всего собрано {len(combined_data)} записей")
        
        # Prepare numpy arrays for training
        feature_columns = [
            'close_zscore', 'volume_zscore', 'rsi_14_normalized',
            'atr_14_normalized', 'momentum_14_normalized',
            'ema_fast_distance', 'ema_slow_distance', 'price_position'
        ]
        
        # Create observation array
        observations = combined_data[feature_columns].to_numpy()
        
        # Save data
        np.savez(output_file, observations=observations, symbols=combined_data['symbol'].to_numpy())
        logger.info(f"Данные сохранены в {output_file}")
    
    def process_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw candles into normalized features for Disco57
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with processed features
        """
        # Calculate features
        df = df.copy()
        
        # 1. Close z-score (over last 50 candles)
        rolling_mean = df['close'].rolling(window=50).mean()
        rolling_std = df['close'].rolling(window=50).std()
        # Защита от деления на 0
        df['close_zscore'] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)
        
        # 2. Volume z-score (over last 50 candles)
        vol_mean = df['volume'].rolling(window=50).mean()
        vol_std = df['volume'].rolling(window=50).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-10)
        
        # 3. RSI 14 normalized (0-1)
        def calculate_rsi(series, periods=14):
            deltas = np.diff(series)
            seed = deltas[:periods+1]
            up = seed[seed >= 0].sum()/periods
            down = -seed[seed < 0].sum()/periods
            rs = up/down if down != 0 else 0
            rsi = np.zeros_like(series)
            rsi[:periods] = 100. - 100./(1. + rs)
            
            for i in range(periods, len(series)):
                delta = deltas[i-1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta
                
                up = (up * (periods - 1) + upval) / periods
                down = (down * (periods - 1) + downval) / periods
                rs = up/down if down != 0 else 0
                rsi[i] = 100. - 100./(1. + rs)
            return rsi
        
        df['rsi_14'] = calculate_rsi(df['close'].values, 14)
        df['rsi_14_normalized'] = df['rsi_14'] / 100.0
        
        # 4. ATR 14 normalized
        def calculate_atr(high, low, close, periods=14):
            trs = np.zeros(len(close))
            for i in range(1, len(close)):
                trs[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            atr = np.zeros(len(close))
            atr[periods-1] = trs[1:periods].mean()
            for i in range(periods, len(close)):
                atr[i] = (atr[i-1] * (periods-1) + trs[i]) / periods
            return atr
        
        df['atr_14'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        mean_price = df['close'].rolling(window=50).mean()
        df['atr_14_normalized'] = df['atr_14'] / (mean_price + 1e-10)
        
        # 5. Momentum 14 normalized
        df['momentum_14'] = df['close'] - df['close'].shift(14)
        rolling_std_momentum = df['close'].rolling(window=50).std()
        df['momentum_14_normalized'] = df['momentum_14'] / (rolling_std_momentum + 1e-10)
        
        # 6. EMA fast distance (% from price to EMA9)
        ema9 = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_fast_distance'] = (df['close'] - ema9) / (df['close'] + 1e-10) * 100
        
        # 7. EMA slow distance (% from price to EMA21)
        ema21 = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_slow_distance'] = (df['close'] - ema21) / (df['close'] + 1e-10) * 100
        
        # 8. Price position in range (0=low, 1=high over last 10 bars)
        rolling_high = df['high'].rolling(window=10).max()
        rolling_low = df['low'].rolling(window=10).min()
        df['price_position'] = (df['close'] - rolling_low) / ((rolling_high - rolling_low) + 1e-10)
        
        # Fill NaN values
        df.fillna(0, inplace=True)
        
        return df
    
    async def close(self):
        """Close connection"""
        await self.exchange.close()


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Limited set of symbols for testing
    test_symbols = [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT',
        'XRP/USDT:USDT',
        'SOL/USDT:USDT',
        'DOT/USDT:USDT'
    ]
    
    collector = DataCollector(test_symbols, timeframe='1m', days=7)
    
    asyncio.run(collector.collect_data('test_data.npz'))
    asyncio.run(collector.close())
