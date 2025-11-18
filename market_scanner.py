"""
🔍 MARKET SCANNER - Динамическое сканирование ликвидных монет
"""

from typing import List, Dict, Optional
import logging
from pybit.unified_trading import HTTP


class MarketScanner:
    """Сканер рынка для поиска ликвидных монет"""
    
    def __init__(self, client: HTTP, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.cached_symbols = []
        self.last_update = None
    
    def get_liquid_symbols(self, min_volume_24h: float = 1000000, limit: int = 60) -> List[str]:
        """
        Получение списка ликвидных монет
        
        Args:
            min_volume_24h: Минимальный объём за 24ч в USD
            limit: Максимальное количество монет
        
        Returns:
            Список символов отсортированных по ликвидности
        """
        try:
            # Получаем все доступные инструменты
            response = self.client.get_tickers(category="linear")
            
            if response['retCode'] != 0:
                self.logger.error(f"Ошибка получения тикеров: {response.get('retMsg')}")
                return self.cached_symbols if self.cached_symbols else []
            
            tickers = response['result']['list']
            
            # Фильтруем только USDT пары
            usdt_pairs = []
            for ticker in tickers:
                symbol = ticker['symbol']
                
                # Только USDT пары
                if not symbol.endswith('USDT'):
                    continue
                
                # Проверяем объём
                volume_24h = float(ticker.get('turnover24h', 0))
                if volume_24h < min_volume_24h:
                    continue
                
                # Проверяем что торговля активна
                last_price = float(ticker.get('lastPrice', 0))
                if last_price <= 0:
                    continue
                
                usdt_pairs.append({
                    'symbol': symbol,
                    'volume_24h': volume_24h,
                    'last_price': last_price,
                    'price_change_percent': float(ticker.get('price24hPcnt', 0)) * 100
                })
            
            # Сортируем по объёму
            usdt_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
            
            # Берём топ N
            top_symbols = [pair['symbol'] for pair in usdt_pairs[:limit]]
            
            self.logger.info(f"📊 Найдено {len(top_symbols)} ликвидных монет (мин. объём: ${min_volume_24h:,.0f})")
            
            # Кешируем результат
            self.cached_symbols = top_symbols
            
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"Ошибка сканирования рынка: {e}")
            return self.cached_symbols if self.cached_symbols else []
    
    def validate_symbol(self, symbol: str) -> bool:
        """Проверка что символ доступен для торговли"""
        try:
            response = self.client.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            
            if response['retCode'] != 0:
                return False
            
            if not response['result']['list']:
                return False
            
            info = response['result']['list'][0]
            status = info.get('status', '')
            
            return status == 'Trading'
            
        except Exception:
            return False
