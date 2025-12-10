#!/usr/bin/env python3
"""
Bybit API Wrapper для GoldTrigger Bot
Работа с Bybit Futures (USDT Perpetual)
"""

import logging
import os
import asyncio
import time
from typing import Dict, List, Optional
import ccxt.async_support as ccxt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BybitAPI:
    """Обертка для работы с Bybit API"""
    
    def __init__(self, paper_trading: bool | None = None):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        if paper_trading is None:
            self.testnet = os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'
        else:
            self.testnet = paper_trading
        
        if not self.api_key or not self.api_secret:
            raise ValueError("BYBIT_API_KEY и BYBIT_API_SECRET должны быть установлены в .env")
        
        # Инициализация CCXT
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
            logger.info("Bybit API: TESTNET режим")
        else:
            logger.info("Bybit API: LIVE режим")
        
        self.retry_count = 3
        self.retry_delay = 1
        self.leverage_cache: Dict[str, int] = {}
    
    async def _retry_request(self, func, *args, **kwargs):
        """Повторить запрос при ошибке"""
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
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Получить текущую цену"""
        return await self._retry_request(self.exchange.fetch_ticker, symbol)
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '5m', 
                          limit: int = 100) -> List:
        """Получить свечи OHLCV"""
        return await self._retry_request(
            self.exchange.fetch_ohlcv, 
            symbol, 
            timeframe, 
            limit=limit
        )
    
    async def fetch_positions(self) -> List[Dict]:
        """Получить открытые позиции"""
        try:
            positions = await self._retry_request(self.exchange.fetch_positions)
            # Фильтруем только открытые позиции
            return [p for p in positions if float(p.get('contracts', 0)) > 0]
        except Exception as e:
            logger.error(f"Ошибка получения позиций: {e}")
            return []

    async def fetch_order_book(self, symbol: str, limit: int = 5) -> Dict:
        """Получить стакан по инструменту через родной Bybit endpoint"""
        try:
            result = await self._retry_request(
                self.exchange.public_get_v5_market_orderbook,
                {
                    "category": "linear",
                    "symbol": symbol.replace('/', '').replace(':USDT', ''),
                    "limit": limit
                }
            )
            book_list = (result.get("result", {}) or {}).get("list", [])
            if not book_list:
                return {"bids": [], "asks": []}
            book = book_list[0]
            bids = [(float(p[0]), float(p[1])) for p in (book.get("b") or []) if len(p) >= 2]
            asks = [(float(p[0]), float(p[1])) for p in (book.get("a") or []) if len(p) >= 2]
            if not bids and not asks:
                raise ValueError("empty orderbook from bybit endpoint")
            return {"bids": bids, "asks": asks}
        except Exception as e:
            logger.debug(f"Ошибка получения стакана {symbol}: {e}, fallback ccxt")
            try:
                orderbook = await self._retry_request(
                    self.exchange.fetch_order_book,
                    symbol,
                    limit
                )
                return {
                    "bids": [(float(price), float(amount)) for price, amount in orderbook.get('bids', [])],
                    "asks": [(float(price), float(amount)) for price, amount in orderbook.get('asks', [])]
                }
            except Exception as fallback_err:
                logger.debug(f"Fallback ccxt orderbook провалился: {fallback_err}")
                return {"bids": [], "asks": []}
    
    async def fetch_closed_pnl(self, start_time_ms: Optional[int] = None, limit: int = 200) -> List[Dict]:
        """Получить список закрытых сделок через Bybit API"""
        try:
            params = {
                "category": "linear",
                "limit": limit
            }
            if start_time_ms:
                params["startTime"] = start_time_ms
                params["endTime"] = int(time.time() * 1000)
            result = await self._retry_request(
                self.exchange.private_get_v5_position_closed_pnl,
                params
            )
            return (result.get("result", {}) or {}).get("list", []) or []
        except Exception as e:
            logger.error(f"Ошибка получения закрытых сделок: {e}")
            return []
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Установить плечо, избегая повторных запросов"""
        symbol_key = symbol.upper()
        if self.leverage_cache.get(symbol_key) == leverage:
            return True
        try:
            await self._retry_request(
                self.exchange.set_leverage,
                leverage,
                symbol
            )
            self.leverage_cache[symbol_key] = leverage
            logger.info(f"Плечо установлено: {symbol} x{leverage}")
            return True
        except Exception as e:
            message = str(e)
            if '110043' in message or 'leverage not modified' in message:
                logger.info(f"Плечо уже установлено для {symbol}, пропускаем")
                self.leverage_cache[symbol_key] = leverage
                return True
            logger.error(f"Ошибка установки плеча {symbol}: {e}")
            return False
    
    async def create_order(self, symbol: str, side: str, amount: float, 
                          price: Optional[float] = None, 
                          leverage: int = 20,
                          reduce_only: bool = False) -> Optional[Dict]:
        """
        Создать ордер
        
        Args:
            symbol: Торговая пара
            side: 'buy' или 'sell'
            amount: Количество контрактов
            price: Цена (None для market order)
            leverage: Плечо
            reduce_only: Только закрытие позиции (не открывать новую)
        
        Returns:
            Информация об ордере или None при ошибке
        """
        try:
            # Устанавливаем плечо только если не reduce_only
            if not reduce_only:
                await self.set_leverage(symbol, leverage)
            
            # Создаем ордер
            order_type = 'market' if price is None else 'limit'
            
            params = {
                'position_idx': 0,  # One-way mode
            }
            
            # КРИТИЧНО: reduceOnly для закрытия позиции
            if reduce_only:
                params['reduceOnly'] = True
            
            order = await self._retry_request(
                self.exchange.create_order,
                symbol,
                order_type,
                side,
                amount,
                price,
                params
            )
            
            logger.info(f"Ордер создан: {symbol} {side} {amount} @ {price or 'market'}")
            return order
            
        except Exception as e:
            logger.error(f"Ошибка создания ордера {symbol}: {e}")
            return None
    
    async def set_stop_loss(self, symbol: str, side: str, sl_price: float) -> bool:
        """
        Установить Stop Loss
        
        Args:
            symbol: Торговая пара
            side: 'long' или 'short'
            sl_price: Цена SL
        
        Returns:
            True если успешно
        """
        try:
            # Для long позиции SL - это sell stop
            # Для short позиции SL - это buy stop
            order_side = 'sell' if side == 'long' else 'buy'
            
            params = {
                'stopLoss': sl_price,
                'position_idx': 0,
            }
            
            # Используем set_trading_stop
            await self._retry_request(
                self.exchange.private_post_v5_position_trading_stop,
                {
                    'category': 'linear',
                    'symbol': symbol.replace('/', '').replace(':USDT', ''),
                    'stopLoss': str(sl_price),
                    'positionIdx': 0,
                }
            )
            
            logger.info(f"SL установлен: {symbol} @ {sl_price}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка установки SL {symbol}: {e}")
            return False
    
    async def set_take_profit(self, symbol: str, side: str, tp_price: float) -> bool:
        """
        Установить Take Profit
        
        Args:
            symbol: Торговая пара
            side: 'long' или 'short'
            tp_price: Цена TP
        
        Returns:
            True если успешно
        """
        try:
            params = {
                'takeProfit': tp_price,
                'position_idx': 0,
            }
            
            await self._retry_request(
                self.exchange.private_post_v5_position_trading_stop,
                {
                    'category': 'linear',
                    'symbol': symbol.replace('/', '').replace(':USDT', ''),
                    'takeProfit': str(tp_price),
                    'positionIdx': 0,
                }
            )
            
            logger.info(f"TP установлен: {symbol} @ {tp_price}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка установки TP {symbol}: {e}")
            return False
    
    async def close_position(self, symbol: str) -> bool:
        """
        Закрыть позицию
        
        Args:
            symbol: Торговая пара
        
        Returns:
            True если успешно
        """
        try:
            # Получаем текущую позицию
            positions = await self.fetch_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position:
                logger.warning(f"Позиция {symbol} не найдена")
                return False
            
            # Определяем сторону закрытия (противоположную открытию)
            # ccxt возвращает 'long'/'short', а не 'Buy'/'Sell'
            pos_side = position['side'].lower()
            if pos_side in ['long', 'buy']:
                side = 'sell'  # Закрываем LONG через SELL
            else:
                side = 'buy'   # Закрываем SHORT через BUY
            amount = abs(float(position['contracts']))
            
            # Закрываем market ордером с reduceOnly=True
            order = await self.create_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=None,  # Market
                reduce_only=True  # КРИТИЧНО: только закрытие!
            )
            
            if order:
                logger.info(f"Позиция закрыта: {symbol}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
            return False
    
    async def get_account_balance(self) -> Dict:
        """Получить баланс аккаунта"""
        try:
            balance = await self._retry_request(self.exchange.fetch_balance)
            return {
                'total': balance.get('total', {}),
                'free': balance.get('free', {}),
                'used': balance.get('used', {})
            }
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return {}
    
    async def get_positions_with_sl_tp(self) -> List[Dict]:
        """
        Получить позиции напрямую через Bybit API с SL/TP
        Возвращает сырые данные от Bybit (не ccxt)
        """
        try:
            result = await self._retry_request(
                self.exchange.private_get_v5_position_list,
                {
                    "category": "linear",
                    "settleCoin": "USDT"
                }
            )
            return result.get("result", {}).get("list", [])
        except Exception as e:
            logger.error(f"Ошибка получения позиций с SL/TP: {e}")
            return []
    
    async def close(self):
        """Закрыть соединение"""
        await self.exchange.close()


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

async def test_api():
    """Тест API"""
    logging.basicConfig(level=logging.INFO)
    
    api = BybitAPI()
    
    try:
        # Тест получения цены
        print("\n=== Тест получения цены ===")
        ticker = await api.fetch_ticker('BTC/USDT:USDT')
        print(f"BTC цена: ${ticker['last']}")
        
        # Тест получения свечей
        print("\n=== Тест получения свечей ===")
        candles = await api.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=5)
        print(f"Получено {len(candles)} свечей")
        for c in candles[-3:]:
            print(f"  {c}")
        
        # Тест получения позиций
        print("\n=== Тест получения позиций ===")
        positions = await api.fetch_positions()
        print(f"Открытых позиций: {len(positions)}")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['side']} {pos['contracts']}")
        
        # Тест получения баланса
        print("\n=== Тест получения баланса ===")
        balance = await api.get_account_balance()
        usdt_balance = balance.get('total', {}).get('USDT', 0)
        print(f"USDT баланс: ${usdt_balance}")
        
    finally:
        await api.close()


if __name__ == '__main__':
    asyncio.run(test_api())
