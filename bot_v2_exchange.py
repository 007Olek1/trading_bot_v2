"""
🏦 МЕНЕДЖЕР РАБОТЫ С БИРЖЕЙ BYBIT
Безопасное взаимодействие с биржей
"""

import ccxt.async_support as ccxt
import logging
import random
import math
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from bot_v2_config import Config

logger = logging.getLogger(__name__)


class ExchangeManager:
    """Управление подключением к бирже и операциями"""
    
    def __init__(self):
        self.exchange = None
        self.connected = False
        self._using_mock = False
    
    # ======================
    # ВСТРОЕННАЯ MOCK-БИРЖА
    # ======================
    class _MockBybit:
        """Простая имитация биржи для офлайн-тестов"""
        def __init__(self, symbols: List[str]):
            self._symbols = symbols[:]
            if not self._symbols:
                # Базовый набор если не передали
                self._symbols = [
                    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT'
                ]
            self._balance_usdt = 100.0
            self._positions: Dict[str, Dict[str, Any]] = {}
            self._open_orders: List[Dict[str, Any]] = []
            self._leverage_by_symbol: Dict[str, int] = {}
            self._start_ts_ms = int(time.time() * 1000) - (60 * 60 * 1000)
            # Имитация рынков с минимальным размером 0.01
            self.markets = {sym: {'precision': {'price': 2}, 'limits': {'amount': {'min': 0.01}}} for sym in self._symbols}
        
        async def close(self):
            return True
        
        async def set_leverage(self, leverage: int, symbol: str):
            self._leverage_by_symbol[symbol] = leverage
            return True
        
        async def fetch_balance(self):
            return {"USDT": {"free": self._balance_usdt}}
        
        async def fetch_tickers(self):
            # Генерируем тикеры с псевдо-объемом
            tickers = {}
            for i, sym in enumerate(self._symbols):
                base_price = 10 + i * 5
                last = base_price * (1 + (math.sin(time.time() / 300 + i) * 0.05))
                tickers[sym] = {
                    'symbol': sym,
                    'last': last,
                    'bid': last * 0.999,
                    'ask': last * 1.001,
                    'quoteVolume': 1_000_000 + (len(self._symbols) - i) * 10_000,
                }
            return tickers
        
        def _gen_price_series(self, start_price: float, steps: int, volatility: float) -> List[float]:
            price = start_price
            series = []
            for _ in range(steps):
                drift = 0.0002
                shock = random.uniform(-volatility, volatility)
                price = max(0.0001, price * (1 + drift + shock))
                series.append(price)
            return series
        
        async def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100):
            # Псевдо OHLCV на основе случайного блуждания
            seed = abs(hash(symbol + timeframe)) % 1_000_000
            random.seed(seed)
            base = 10 + (abs(hash(symbol)) % 100) / 3
            vol = 0.003 if timeframe in ("1m", "5m") else 0.001
            closes = self._gen_price_series(base, limit, vol)
            ohlcv = []
            tf_minutes = 1
            if timeframe.endswith('m'):
                tf_minutes = int(timeframe[:-1])
            elif timeframe.endswith('h'):
                tf_minutes = int(timeframe[:-1]) * 60
            elif timeframe.endswith('d'):
                tf_minutes = int(timeframe[:-1]) * 60 * 24
            ts = self._start_ts_ms
            for c in closes:
                high = c * (1 + random.uniform(0, 0.002))
                low = c * (1 - random.uniform(0, 0.002))
                open_ = (high + low) / 2
                volume = random.uniform(1000, 10000)
                ohlcv.append([ts, open_, high, low, c, volume])
                ts += tf_minutes * 60 * 1000
            return ohlcv
        
        async def create_market_order(self, symbol: str, side: str, amount: float):
            # Цена по последнему close
            candles = await self.fetch_ohlcv(symbol, limit=1)
            price = float(candles[-1][4]) if candles else 1.0
            position = self._positions.get(symbol)
            if not position:
                position = {
                    'symbol': symbol,
                    'side': 'long' if side.lower() == 'buy' else 'short',
                    'contracts': float(amount),
                    'entryPrice': price,
                    'markPrice': price,
                    'leverage': self._leverage_by_symbol.get(symbol, 5),
                    'info': {'stopLoss': None},
                }
            else:
                # Простая логика: заменяем позицию
                position['side'] = 'long' if side.lower() == 'buy' else 'short'
                position['contracts'] = float(amount)
                position['entryPrice'] = price
                position['markPrice'] = price
            self._positions[symbol] = position
            return {'id': f'MOCK_MKT_{symbol}_{int(time.time()*1000)}', 'symbol': symbol, 'price': price}
        
        async def private_post_v5_position_trading_stop(self, payload: Dict[str, Any]):
            symbol = payload.get('symbol')  # ожид. TAOUSDT
            # Преобразуем обратно к формату с ":USDT"
            # Поиск по сохраненным символам, у которых без "/" совпадает
            for full_sym in self._positions.keys():
                if full_sym.split(':')[0].replace('/', '') == symbol:
                    mock_symbol = full_sym
                    break
            else:
                # Если позиции нет, просто ок
                return {'retCode': 0, 'retMsg': 'OK'}
            pos = self._positions.get(mock_symbol)
            if pos:
                pos['info']['stopLoss'] = str(payload.get('stopLoss'))
            return {'retCode': 0, 'retMsg': 'OK'}
        
        async def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: Dict[str, Any] = None):
            order = {
                'id': f'MOCK_LIM_{symbol}_{int(price*100)}_{int(time.time()*1000)}',
                'symbol': symbol,
                'type': 'limit',
                'side': side,
                'amount': float(amount),
                'price': float(price),
                'reduceOnly': bool(params.get('reduceOnly') if params else False)
            }
            self._open_orders.append(order)
            return order
        
        async def cancel_order(self, order_id: str, symbol: str):
            self._open_orders = [o for o in self._open_orders if o.get('id') != order_id]
            return True
        
        async def fetch_open_orders(self, symbol: Optional[str] = None):
            if symbol:
                return [o for o in self._open_orders if o.get('symbol') == symbol]
            return list(self._open_orders)
        
        async def fetch_positions(self):
            # Возвращаем только позиции с контрактами > 0
            result = []
            for sym, pos in self._positions.items():
                if float(pos.get('contracts', 0)) > 0:
                    result.append({
                        'symbol': sym,
                        'side': pos['side'],
                        'contracts': pos['contracts'],
                        'entryPrice': pos['entryPrice'],
                        'markPrice': pos['markPrice'],
                        'leverage': pos.get('leverage', 5),
                        'stopLoss': pos['info'].get('stopLoss'),
                        'info': pos['info']
                    })
            return result
        
        async def load_markets(self):
            # Возвращаем словарь рынков
            return self.markets
    
    async def connect(self):
        """Подключение к Bybit"""
        try:
            if getattr(Config, 'OFFLINE_MODE', False):
                logger.warning("🌐 OFFLINE MODE: используем имитацию биржи для тестов")
                self.exchange = self._MockBybit(getattr(Config, 'TOP_100_SYMBOLS', []))
                self._using_mock = True
                self.connected = True
                return True
            
            self.exchange = ccxt.bybit({
                "apiKey": Config.BYBIT_API_KEY,
                "secret": Config.BYBIT_API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"}
            })
            
            # Проверка подключения
            await self.exchange.fetch_balance()
            self.connected = True
            
            logger.info("✅ Подключение к Bybit успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Bybit: {e}")
            # Фолбэк: при тестовом режиме включаем MOCK, чтобы можно было тестировать
            if getattr(Config, 'TEST_MODE', False):
                logger.warning("🧪 Тестовый режим: переключаюсь на OFFLINE MOCK биржу")
                self.exchange = self._MockBybit(getattr(Config, 'TOP_100_SYMBOLS', []))
                self._using_mock = True
                self.connected = True
                return True
            self.connected = False
            return False
    
    async def disconnect(self):
        """Отключение от биржи"""
        if self.exchange:
            await self.exchange.close()
            self.connected = False
            logger.info("🔌 Отключение от Bybit")
    
    async def get_balance(self) -> Optional[float]:
        """Получить баланс USDT"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_balance = balance.get("USDT", {}).get("free", 0)
            return float(usdt_balance)
        except Exception as e:
            logger.error(f"❌ Ошибка получения баланса: {e}")
            return None
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Установить леверидж"""
        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info(f"✅ Леверидж {leverage}x установлен для {symbol}")
            return True
        except Exception as e:
            # Игнорируем ошибку "leverage not modified"
            if "110043" in str(e) or "leverage not modified" in str(e).lower():
                logger.debug(f"⚠️ Леверидж уже установлен: {symbol}")
                return True
            logger.error(f"❌ Ошибка установки leverage: {e}")
            return False
    
    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float
    ) -> Optional[Dict[str, Any]]:
        """
        Создать рыночный ордер
        
        Args:
            symbol: Пара (например "BTC/USDT:USDT")
            side: "buy" или "sell"
            amount: Количество контрактов
        """
        try:
            logger.info(f"🚀 Создаю market ордер: {symbol} {side} {amount:.6f}")
            
            order = await self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount
            )
            
            logger.info(f"✅ Market ордер создан: {order.get('id')}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания market ордера: {e}")
            return None
    
    async def create_stop_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        КРИТИЧЕСКИ ВАЖНО: Создать Stop Loss для СУЩЕСТВУЮЩЕЙ позиции
        Использует endpoint set-trading-stop для Bybit v5
        
        Args:
            symbol: Пара
            side: "buy" или "sell" (НЕ используется, определяем из позиции)
            amount: Количество (НЕ используется, SL для всей позиции)
            stop_price: Цена срабатывания Stop Loss
        """
        try:
            logger.info(f"🛡️ Устанавливаю STOP LOSS для {symbol} @ ${stop_price:.4f}")
            
            # Используем set-trading-stop endpoint Bybit v5
            # Этот метод устанавливает SL для СУЩЕСТВУЮЩЕЙ позиции
            
            # Получаем информацию о паре для правильного форматирования цены
            markets = getattr(self.exchange, 'markets', {})
            if symbol in markets:
                # precision может быть float (0.01) или int (2)
                price_prec = markets.get(symbol, {}).get('precision', {}).get('price', 2)
                # Преобразуем в кол-во знаков после запятой
                if isinstance(price_prec, float):
                    # 0.01 -> 2, 0.1 -> 1, 0.001 -> 3
                    decimal_places = int(abs(math.log10(price_prec)))
                else:
                    decimal_places = int(price_prec)
                formatted_sl = round(stop_price, decimal_places)
            else:
                # По умолчанию 2 знака
                formatted_sl = round(stop_price, 2)
            
            # КРИТИЧНО: Правильный формат символа для Bybit
            # Из "TAO/USDT:USDT" нужно получить "TAOUSDT"
            clean_symbol = symbol.split(':')[0].replace('/', '')  # TAO/USDT:USDT -> TAOUSDT
            
            logger.debug(f"🔧 Символ для API: {symbol} -> {clean_symbol}")
            
            response = await self.exchange.private_post_v5_position_trading_stop({
                'category': 'linear',
                'symbol': clean_symbol,
                'stopLoss': str(formatted_sl),
                'slTriggerBy': 'LastPrice',
                'positionIdx': 0  # 0 для One-Way режима
            })
            
            # Полный debug ответа
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg')
            logger.debug(f"🔍 Ответ Bybit: retCode={ret_code} (type:{type(ret_code)}), retMsg={ret_msg}")
            
            # КРИТИЧНО: retCode может быть int(0) или str("0")!
            if ret_code == 0 or ret_code == "0" or str(ret_code) == "0":
                logger.info(f"✅ STOP LOSS установлен для {symbol} @ ${stop_price:.4f}")
                # Возвращаем фиктивный ордер для совместимости
                return {
                    'id': f"SL_{symbol}_{int(stop_price * 10000)}",
                    'symbol': symbol,
                    'type': 'stop_loss',
                    'side': side,
                    'price': stop_price,
                    'status': 'set',
                    'info': response
                }
            # retCode 34040 = "not modified" - SL уже установлен на этот уровень
            elif ret_code == 34040 or ret_code == "34040":
                logger.info(f"ℹ️ STOP LOSS уже установлен для {symbol} @ ${stop_price:.4f}")
                # Возвращаем успешный результат, т.к. SL есть
                return {
                    'id': f"SL_{symbol}_{int(stop_price * 10000)}",
                    'symbol': symbol,
                    'type': 'stop_loss',
                    'side': side,
                    'price': stop_price,
                    'status': 'already_set',
                    'info': response
                }
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                ret_code = response.get('retCode', 'N/A')
                logger.error(f"❌ Ошибка установки SL: retCode={ret_code}, retMsg={error_msg}")
                logger.error(f"🔍 Полный ответ: {response}")
                return None
                
        except Exception as e:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА установки STOP LOSS: {e}")
            logger.exception("Детали ошибки:")
            return None
    
    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Создать лимитный ордер (для Take Profit)
        
        Args:
            symbol: Пара
            side: "buy" или "sell"
            amount: Количество
            price: Цена
        """
        try:
            logger.info(f"🎯 Создаю LIMIT: {symbol} {side} @ ${price:.4f}")
            
            order = await self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                params={
                    "reduceOnly": True,
                    "timeInForce": "GTC"
                }
            )
            
            if order and order.get("id"):
                logger.info(f"✅ LIMIT создан: {order['id']}")
                return order
            else:
                logger.warning("⚠️ LIMIT не получил ID")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания LIMIT: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Отменить ордер"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ Ордер {order_id} отменен")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Ошибка отмены ордера {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Отменить все ордера"""
        try:
            if symbol:
                orders = await self.exchange.fetch_open_orders(symbol)
            else:
                orders = await self.exchange.fetch_open_orders()
            
            count = 0
            for order in orders:
                try:
                    await self.exchange.cancel_order(order["id"], order["symbol"])
                    count += 1
                except:
                    pass
            
            logger.info(f"✅ Отменено {count} ордеров")
            return count
            
        except Exception as e:
            logger.error(f"❌ Ошибка отмены всех ордеров: {e}")
            return 0
    
    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """Получить открытые позиции"""
        try:
            positions = await self.exchange.fetch_positions()
            # Только открытые
            open_positions = [
                p for p in positions
                if float(p.get("contracts", 0) or 0) > 0
            ]
            return open_positions
        except Exception as e:
            logger.error(f"❌ Ошибка получения позиций: {e}")
            return []
    
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получить открытые ордера"""
        try:
            if symbol:
                orders = await self.exchange.fetch_open_orders(symbol)
            else:
                orders = await self.exchange.fetch_open_orders()
            return orders
        except Exception as e:
            logger.error(f"❌ Ошибка получения ордеров: {e}")
            return []
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 100
    ) -> Optional[List]:
        """Получить свечи"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            return ohlcv
        except Exception as e:
            logger.error(f"❌ Ошибка получения OHLCV {symbol}: {e}")
            return None
    
    async def get_top_volume_symbols(self, top_n: int = 50) -> List[str]:
        """Получить топ символы по объему"""
        try:
            tickers = await self.exchange.fetch_tickers()
            
            # Фильтр: только USDT perpetual
            usdt_perp = {
                symbol: ticker for symbol, ticker in tickers.items()
                if ":USDT" in symbol and ticker.get("quoteVolume", 0) > 0
            }
            
            # Сортировка по объему
            sorted_symbols = sorted(
                usdt_perp.items(),
                key=lambda x: x[1].get("quoteVolume", 0),
                reverse=True
            )
            
            # Топ N
            top_symbols = [symbol for symbol, _ in sorted_symbols[:top_n]]
            
            logger.info(f"📊 Топ {len(top_symbols)} символов по объему")
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения топ символов: {e}")
            return []


# Глобальный экземпляр
exchange_manager = ExchangeManager()


