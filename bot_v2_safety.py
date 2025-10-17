"""
🛡️ СИСТЕМА БЕЗОПАСНОСТИ БОТА V2.0
ПРИОРИТЕТ #1 - Защита капитала!
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
from bot_v2_config import Config

logger = logging.getLogger(__name__)


class RiskManager:
    """Управление рисками - ЖЕСТКИЕ лимиты"""
    
    def __init__(self):
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.last_trade_time = None
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        self.weekly_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        # Счетчик тестовых сделок
        self.test_trades_count = 0
    
    def reset_daily_stats(self):
        """Сброс дневной статистики"""
        now = datetime.now()
        if now >= self.daily_reset_time + timedelta(days=1):
            logger.info("🔄 Сброс дневной статистики")
            self.daily_loss = 0.0
            self.trades_today = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
    
    def reset_weekly_stats(self):
        """Сброс недельной статистики"""
        now = datetime.now()
        if now >= self.weekly_reset_time + timedelta(weeks=1):
            logger.info("🔄 Сброс недельной статистики")
            self.weekly_loss = 0.0
            self.weekly_reset_time = now.replace(hour=0, minute=0, second=0)
    
    def can_open_trade(self, balance: float) -> Tuple[bool, str]:
        """
        КРИТИЧЕСКАЯ ПРОВЕРКА: Можно ли открывать сделку?
        
        Returns:
            (can_open, reason)
        """
        self.reset_daily_stats()
        self.reset_weekly_stats()
        
        # CHECK 1: Тестовый режим
        if Config.TEST_MODE and self.test_trades_count >= Config.TEST_MAX_TRADES:
            return False, f"🧪 Лимит тестовых сделок ({Config.TEST_MAX_TRADES}) достигнут! Остановка для анализа."
        
        # CHECK 2: Дневной лимит убытка
        if self.daily_loss >= Config.MAX_DAILY_LOSS_USD:
            return False, f"💔 Дневной лимит убытка (${Config.MAX_DAILY_LOSS_USD}) достигнут!"
        
        # CHECK 3: Недельный лимит убытка
        if self.weekly_loss >= Config.MAX_WEEKLY_LOSS_USD:
            return False, f"💔 Недельный лимит убытка (${Config.MAX_WEEKLY_LOSS_USD}) достигнут!"
        
        # CHECK 4: Серия убытков
        if self.consecutive_losses >= Config.CONSECUTIVE_LOSSES_LIMIT:
            return False, f"🚫 {Config.CONSECUTIVE_LOSSES_LIMIT} убытка подряд - ОСТАНОВКА!"
        
        # CHECK 5: Лимит сделок в день
        if self.trades_today >= Config.MAX_TRADES_PER_DAY:
            return False, f"📊 Лимит сделок ({Config.MAX_TRADES_PER_DAY}/день) достигнут"
        
        # CHECK 6: Cooldown после последней сделки
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < Config.COOLDOWN_AFTER_TRADE_SECONDS:
                remaining = int(Config.COOLDOWN_AFTER_TRADE_SECONDS - time_since_last)
                return False, f"⏰ Cooldown: осталось {remaining // 60} мин"
        
        # CHECK 7: Баланс
        min_balance = Config.get_position_size() * 5  # Минимум на 5 сделок
        if balance < min_balance:
            return False, f"💰 Баланс ${balance:.2f} < ${min_balance:.2f} (минимум)"
        
        # CHECK 8: Достаточно ли баланса на позицию
        position_size = Config.get_position_size()
        if balance < position_size * 2:
            return False, f"💰 Недостаточно баланса для позиции ${position_size}"
        
        return True, "✅ Все проверки пройдены"
    
    def record_trade_result(self, profit: float):
        """Записать результат сделки"""
        self.trades_today += 1
        self.last_trade_time = datetime.now()
        
        if Config.TEST_MODE:
            self.test_trades_count += 1
            logger.info(f"🧪 Тестовая сделка {self.test_trades_count}/{Config.TEST_MAX_TRADES}")
        
        if profit < 0:
            # Убыток
            self.daily_loss += abs(profit)
            self.weekly_loss += abs(profit)
            self.consecutive_losses += 1
            logger.warning(f"❌ Убыток: ${profit:.2f}, серия: {self.consecutive_losses}")
        else:
            # Прибыль
            self.consecutive_losses = 0  # Сброс серии
            logger.info(f"✅ Прибыль: ${profit:.2f}")
        
        logger.info(f"📊 Дневной убыток: ${self.daily_loss:.2f}/{Config.MAX_DAILY_LOSS_USD}")
        logger.info(f"📊 Недельный убыток: ${self.weekly_loss:.2f}/{Config.MAX_WEEKLY_LOSS_USD}")
    
    def calculate_position_size(self, balance: float, symbol: str = None) -> float:
        """Расчет размера позиции (КОНСЕРВАТИВНО)"""
        base_size = Config.get_position_size()
        
        # ИСПРАВЛЕНИЕ: Адаптивный размер для максимальной прибыли!
        if symbol:
            # ТОП-5: x10 размер ($20-30) - МАКСИМАЛЬНАЯ ПРИБЫЛЬ!
            top_5 = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT']
            if symbol in top_5:
                return base_size * 10.0  # $20 для ТОП-5
            
            # ТОП-15: x7 размер ($15-20)
            top_15 = ['ADA/USDT:USDT', 'DOGE/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'AVAX/USDT:USDT',
                      'LINK/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'LTC/USDT:USDT', 'TRX/USDT:USDT']
            if symbol in top_15:
                return base_size * 7.0  # $14 для ТОП-15
            
            # Остальные ТОП-50: x5 размер ($10)
            return base_size * 5.0  # $10 для остальных
        
        return base_size
    
    def calculate_sl_tp_prices(
        self,
        entry_price: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Расчет цен Stop Loss и Take Profit
        
        Args:
            entry_price: Цена входа
            side: "buy" или "sell"
            
        Returns:
            (stop_loss_price, take_profit_price)
        """
        # Расчет процентов изменения цены
        # SL: Макс убыток 10% от инвестиций при 5X leverage = 2% от цены
        # TP: рассчитывается отдельно для каждого уровня
        sl_price_pct = Config.MAX_LOSS_PER_TRADE_PERCENT / Config.LEVERAGE / 100  # 10/5/100 = 0.02 (2% цены)
        tp_price_pct = Config.TAKE_PROFIT_MIN_PERCENT / Config.LEVERAGE / 100     # 25/5/100 = 0.05 (5% цены)
        
        if side == "buy":
            stop_loss = entry_price * (1 - sl_price_pct)   # -3.33% цены
            take_profit = entry_price * (1 + tp_price_pct)  # +8.33% цены
        else:  # sell
            stop_loss = entry_price * (1 + sl_price_pct)   # +3.33% цены
            take_profit = entry_price * (1 - tp_price_pct)  # -8.33% цены
        
        logger.info(
            f"🎯 SL/TP: вход=${entry_price:.4f}, "
            f"SL=${stop_loss:.4f} ({'+' if side == 'sell' else '-'}{Config.MAX_LOSS_PER_TRADE_PERCENT}%), "
            f"TP=${take_profit:.4f} ({'+' if side == 'buy' else '-'}{Config.TAKE_PROFIT_MIN_PERCENT}%)"
        )
        
        return stop_loss, take_profit


class EmergencyStop:
    """Аварийная остановка - последняя линия защиты"""
    
    def __init__(self):
        self.emergency_stopped = False
        self.stop_reason = None
        self.last_check_time = datetime.now()
    
    async def check_emergency_conditions(
        self,
        risk_manager: RiskManager,
        open_positions: list,
        bot_errors_count: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Проверка аварийных условий
        
        Returns:
            (should_stop, reason)
        """
        # Проверка только раз в минуту
        if (datetime.now() - self.last_check_time).total_seconds() < 60:
            return False, None
        
        self.last_check_time = datetime.now()
        
        # УСЛОВИЕ 1: Дневной лимит превышен
        if risk_manager.daily_loss >= Config.MAX_DAILY_LOSS_USD:
            return True, f"Дневной убыток ${risk_manager.daily_loss:.2f} >= ${Config.MAX_DAILY_LOSS_USD}"
        
        # УСЛОВИЕ 2: Серия убытков
        if risk_manager.consecutive_losses >= Config.CONSECUTIVE_LOSSES_LIMIT:
            return True, f"{risk_manager.consecutive_losses} убытка подряд"
        
        # УСЛОВИЕ 3: Много ошибок бота
        if bot_errors_count >= 5:
            return True, f"Критические ошибки бота: {bot_errors_count}"
        
        # УСЛОВИЕ 4: Позиция без SL ордера - ОТКЛЮЧЕНО
        # Причина: SL устанавливается через trading-stop API и проверяется при sync_positions
        # Если sl_order_id есть - значит SL установлен на бирже
        for position in open_positions:
            if not position.get('sl_order_id'):
                logger.warning(f"⚠️ Позиция {position['symbol']} без sl_order_id, но это не критично")
                # НЕ ВЫЗЫВАЕМ Emergency Stop! SL может быть на позиции через trading-stop API
        
        return False, None
    
    def activate(self, reason: str):
        """Активировать аварийную остановку"""
        self.emergency_stopped = True
        self.stop_reason = reason
        logger.critical(f"🚨🚨🚨 EMERGENCY STOP ACTIVATED: {reason}")


class PositionGuard:
    """Защита позиций - мониторинг и контроль"""
    
    @staticmethod
    async def create_stop_loss_order(
        exchange,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        КРИТИЧЕСКИ ВАЖНО: Создать Stop Loss ордер НА БИРЖЕ
        
        Returns:
            Order dict или None если ошибка
        """
        try:
            # Сторона для закрытия позиции
            close_side = "sell" if side == "buy" else "buy"
            
            logger.info(f"🛡️ Создаю SL ордер: {symbol} {close_side} @ ${stop_price:.4f}")
            
            # Создаем Stop Market ордер
            sl_order = await exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=close_side,
                amount=amount,
                params={
                    "stopPrice": stop_price,
                    "reduceOnly": True  # Только закрытие позиции
                }
            )
            
            if sl_order and sl_order.get("id"):
                logger.info(f"✅ SL ордер создан: {sl_order['id']}")
                return sl_order
            else:
                logger.error("❌ SL ордер не получил ID!")
                return None
                
        except Exception as e:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА создания SL: {e}")
            return None
    
    @staticmethod
    async def create_take_profit_order(
        exchange,
        symbol: str,
        side: str,
        amount: float,
        tp_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Создать Take Profit ордер НА БИРЖЕ
        
        Returns:
            Order dict или None если ошибка
        """
        try:
            # Сторона для закрытия позиции
            close_side = "sell" if side == "buy" else "buy"
            
            logger.info(f"🎯 Создаю TP ордер: {symbol} {close_side} @ ${tp_price:.4f}")
            
            # Создаем Limit ордер
            tp_order = await exchange.create_order(
                symbol=symbol,
                type="LIMIT",
                side=close_side,
                amount=amount,
                price=tp_price,
                params={
                    "reduceOnly": True,
                    "timeInForce": "GTC"  # Good Till Cancelled
                }
            )
            
            if tp_order and tp_order.get("id"):
                logger.info(f"✅ TP ордер создан: {tp_order['id']}")
                return tp_order
            else:
                logger.warning("⚠️ TP ордер не получил ID")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания TP: {e}")
            return None
    
    @staticmethod
    async def verify_sl_order_exists(
        exchange,
        symbol: str,
        sl_order_id: str
    ) -> bool:
        """Проверить что SL ордер существует на бирже"""
        try:
            orders = await exchange.fetch_open_orders(symbol)
            return any(order["id"] == sl_order_id for order in orders)
        except Exception as e:
            logger.error(f"❌ Ошибка проверки SL ордера: {e}")
            return False


# Глобальные экземпляры
risk_manager = RiskManager()
emergency_stop = EmergencyStop()
position_guard = PositionGuard()


