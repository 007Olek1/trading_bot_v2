"""
🏦 МЕНЕДЖЕР РАБОТЫ С БИРЖЕЙ BYBIT
Безопасное взаимодействие с биржей
"""

import ccxt.async_support as ccxt
import logging
from typing import Optional, Dict, Any, List
from bot_v2_config import Config

logger = logging.getLogger(__name__)


class ExchangeManager:
    """Управление подключением к бирже и операциями"""
    
    def __init__(self):
        self.exchange = None
        self.connected = False
    
    async def connect(self):
        """Подключение к Bybit"""
        try:
            self.exchange = ccxt.bybit({
                "apiKey": Config.BYBIT_API_KEY,
                "secret": Config.BYBIT_API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"}
            })
            
            # Проверка подключения
            balance = await self.exchange.fetch_balance()
            self.connected = True
            
            logger.info("✅ Подключение к Bybit успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Bybit: {e}")
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
            markets = self.exchange.markets
            if symbol in markets:
                # precision может быть float (0.01) или int (2)
                price_prec = markets[symbol].get('precision', {}).get('price', 2)
                # Преобразуем в кол-во знаков после запятой
                if isinstance(price_prec, float):
                    # 0.01 -> 2, 0.1 -> 1, 0.001 -> 3
                    import math
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
        """Получить топ символы по объему с фильтрацией ликвидности"""
        try:
            tickers = await self.exchange.fetch_tickers()
            
            # Фильтр: только USDT perpetual с минимальной ликвидностью
            usdt_perp = {}
            for symbol, ticker in tickers.items():
                if ":USDT" in symbol:
                    volume = ticker.get("quoteVolume", 0)
                    price = ticker.get("last", 0)
                    
                    # Фильтры ликвидности:
                    # 1. Минимальный объем торгов $500,000
                    # 2. Минимальная цена $0.001 (исключаем слишком дешевые)
                    # 3. Проверяем спред если доступен
                    if volume >= 500000 and price >= 0.001:
                        # Проверяем спред если доступны bid/ask
                        bid = ticker.get('bid', 0)
                        ask = ticker.get('ask', 0)
                        if bid > 0 and ask > 0:
                            spread_pct = ((ask - bid) / bid) * 100
                            if spread_pct <= 3.0:  # Спред не больше 3%
                                usdt_perp[symbol] = ticker
                        else:
                            # Если нет данных о спреде, включаем по объему и цене
                            usdt_perp[symbol] = ticker
            
            # Сортировка по объему
            sorted_symbols = sorted(
                usdt_perp.items(),
                key=lambda x: x[1].get("quoteVolume", 0),
                reverse=True
            )
            
            # Топ N
            top_symbols = [symbol for symbol, _ in sorted_symbols[:top_n]]
            
            logger.info(f"📊 Топ {len(top_symbols)} ликвидных символов по объему (из {len(tickers)} всего)")
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения топ символов: {e}")
            return []


# Глобальный экземпляр
exchange_manager = ExchangeManager()


