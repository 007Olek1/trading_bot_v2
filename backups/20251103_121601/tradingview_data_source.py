#!/usr/bin/env python3
"""
Интеграция с TradingView для получения данных 45m таймфрейма
Использует неофициальный API TradingView через WebSocket
"""
import asyncio
import json
import logging
import time
from datetime import datetime
import pytz
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WARSAW_TZ = pytz.timezone('Europe/Warsaw')

class TradingViewDataSource:
    """Источник данных TradingView для 45m таймфрейма"""
    
    def __init__(self):
        self.base_url = "https://symbol-search.tradingview.com"
        self.socket_url = "wss://data.tradingview.com/socket.io/websocket"
        
    def normalize_symbol(self, symbol: str) -> str:
        """Нормализация символа для TradingView (BTCUSDT -> BINANCE:BTCUSDT)"""
        symbol = symbol.upper().replace('/', '').replace('-', '')
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            # TradingView формат: BINANCE:BTCUSDT
            return f"BINANCE:{symbol}"
        return symbol
    
    def get_45m_data_tv_scraper(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Получение 45m данных через неофициальный метод (scraping TradingView)
        ВНИМАНИЕ: Это экспериментальный метод, может не работать стабильно
        """
        try:
            # Попытка использовать tvDatafeed если установлен
            try:
                from tvDatafeed import TvDatafeed
                tv = TvDatafeed()
                
                # Пробуем разные форматы символов
                symbol_variants = [
                    symbol,  # BTCUSDT
                    symbol.replace('USDT', '/USDT'),  # BTC/USDT
                    f"BINANCE:{symbol}",  # BINANCE:BTCUSDT
                    f"BINANCE:{symbol.replace('USDT', '/USDT')}",  # BINANCE:BTC/USDT
                ]
                
                for tv_symbol in symbol_variants:
                    try:
                        # TradingView поддерживает '45' как интервал
                        df = tv.get_hist(
                            symbol=tv_symbol,
                            exchange='BINANCE',
                            interval=45,  # 45 минут (число, не строка)
                            n_bars=limit
                        )
                
                if df is not None and not df.empty:
                    # Переименовываем колонки для совместимости
                    df = df.rename(columns={
                        'datetime': 'timestamp',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume'
                    })
                    
                    # Конвертируем timestamp если нужно
                    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    logger.info(f"✅ TradingView: Получено {len(df)} свечей 45m для {symbol}")
                    return df
            except ImportError:
                logger.debug("⚠️ tvDatafeed не установлен, пробуем альтернативные методы")
            except Exception as e:
                logger.debug(f"⚠️ Ошибка получения данных через tvDatafeed: {e}")
            
            return pd.DataFrame()
        except Exception as e:
            logger.debug(f"⚠️ Ошибка получения 45m данных из TradingView для {symbol}: {e}")
            return pd.DataFrame()
    
    def get_45m_data_binance_fallback(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Альтернативный источник: Binance API (поддерживает 45m)
        """
        try:
            import ccxt
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}  # Для фьючерсов
            })
            
            normalized_symbol = symbol.replace('USDT', '/USDT:USDT') if 'USDT' in symbol else symbol
            
            ohlcv = await exchange.fetch_ohlcv(
                normalized_symbol,
                '45m',
                limit=limit
            )
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logger.info(f"✅ Binance: Получено {len(df)} свечей 45m для {symbol}")
                return df
        except Exception as e:
            logger.debug(f"⚠️ Ошибка получения данных из Binance для {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_45m_data_okx_fallback(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Альтернативный источник: OKX API (поддерживает 45m)
        """
        try:
            import ccxt
            exchange = ccxt.okx({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            
            normalized_symbol = symbol.replace('USDT', '/USDT:USDT') if 'USDT' in symbol else symbol
            
            ohlcv = await exchange.fetch_ohlcv(
                normalized_symbol,
                '45m',
                limit=limit
            )
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logger.info(f"✅ OKX: Получено {len(df)} свечей 45m для {symbol}")
                return df
        except Exception as e:
            logger.debug(f"⚠️ Ошибка получения данных из OKX для {symbol}: {e}")
        
        return pd.DataFrame()

def integrate_tradingview_45m_into_bot():
    """
    Интеграция TradingView данных 45m в бот
    Модифицирует функцию _fetch_ohlcv для использования TV как источника для 45m
    """
    logger.info("🔌 Интеграция TradingView 45m данных в бот...")
    
    tv_source = TradingViewDataSource()
    
    # Тест получения данных
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for symbol in test_symbols:
        logger.info(f"\n📊 Тест получения 45m данных для {symbol}:")
        
        # Пробуем TradingView
        df_tv = tv_source.get_45m_data_tv_scraper(symbol, 50)
        if not df_tv.empty:
            logger.info(f"   ✅ TradingView: {len(df_tv)} свечей")
            logger.info(f"   Последние цены: {df_tv['close'].tail(3).tolist()}")
        else:
            logger.info(f"   ❌ TradingView: данные не получены")
        
        # Пробуем Binance как fallback (синхронный метод)
        df_binance = tv_source.get_45m_data_binance_fallback(symbol, 50)
        if not df_binance.empty:
            logger.info(f"   ✅ Binance: {len(df_binance)} свечей")
        else:
            logger.info(f"   ❌ Binance: данные не получены")
    
    logger.info("\n✅ Тест завершен")

if __name__ == "__main__":
    integrate_tradingview_45m_into_bot()

