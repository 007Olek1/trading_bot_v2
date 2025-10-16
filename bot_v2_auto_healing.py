"""
🔧 AUTO-HEALING СИСТЕМА - САМОИСПРАВЛЕНИЕ
Автоматическое обнаружение и исправление проблем
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from bot_v2_config import Config

logger = logging.getLogger(__name__)


class AutoHealingSystem:
    """Система автоматического самоисправления"""
    
    def __init__(self):
        self.healing_attempts = 0
        self.max_healing_attempts = 3
        self.last_healing_time = None
        self.healing_cooldown = 300  # 5 минут между попытками
        self.known_issues = {}
        
        logger.info("🔧 Auto-Healing система инициализирована")
    
    async def diagnose_and_heal(
        self,
        exchange_manager,
        health_monitor,
        open_positions: list
    ) -> Tuple[bool, str]:
        """
        Диагностика и автоматическое исправление проблем
        
        Returns:
            (healed, action_taken)
        """
        try:
            # Проверка cooldown
            if self.last_healing_time:
                time_since_last = (datetime.now() - self.last_healing_time).total_seconds()
                if time_since_last < self.healing_cooldown:
                    return False, f"Healing в cooldown ({int(self.healing_cooldown - time_since_last)}с)"
            
            # Проверка лимита попыток
            if self.healing_attempts >= self.max_healing_attempts:
                return False, "Достигнут лимит попыток самоисправления"
            
            # ДИАГНОСТИКА 1: Проблемы с подключением к бирже
            if not exchange_manager.connected:
                logger.warning("🔧 Healing: Обнаружена проблема с подключением к бирже")
                success = await self.heal_exchange_connection(exchange_manager)
                if success:
                    return True, "✅ Восстановлено подключение к бирже"
                return False, "❌ Не удалось восстановить подключение"
            
            # ДИАГНОСТИКА 2: Позиции без SL ордеров
            positions_without_sl = [p for p in open_positions if not p.get('sl_order_id')]
            if positions_without_sl:
                logger.critical(f"🔧 Healing: {len(positions_without_sl)} позиций без SL!")
                success = await self.heal_missing_sl_orders(
                    exchange_manager,
                    positions_without_sl
                )
                if success:
                    return True, f"✅ Созданы SL ордера для {len(positions_without_sl)} позиций"
                return False, "❌ Не удалось создать SL ордера"
            
            # ДИАГНОСТИКА 3: Слишком много ошибок
            if health_monitor.errors_count >= 5:
                logger.warning(f"🔧 Healing: Обнаружено {health_monitor.errors_count} ошибок")
                success = await self.heal_errors(health_monitor)
                if success:
                    return True, "✅ Счетчики ошибок сброшены"
            
            # ДИАГНОСТИКА 4: "Зависшие" позиции (открыты очень долго)
            stuck_positions = await self.find_stuck_positions(open_positions)
            if stuck_positions:
                logger.warning(f"🔧 Healing: {len(stuck_positions)} зависших позиций")
                success = await self.heal_stuck_positions(
                    exchange_manager,
                    stuck_positions
                )
                if success:
                    return True, f"✅ Закрыты {len(stuck_positions)} зависших позиций"
            
            return False, "Проблем не обнаружено"
            
        except Exception as e:
            logger.error(f"❌ Ошибка Auto-Healing: {e}")
            return False, f"Ошибка: {e}"
    
    async def heal_exchange_connection(self, exchange_manager) -> bool:
        """Восстановление подключения к бирже"""
        try:
            logger.info("🔧 Healing: Попытка восстановить подключение к бирже...")
            
            # Закрываем старое подключение
            try:
                await exchange_manager.disconnect()
            except:
                pass
            
            await asyncio.sleep(2)
            
            # Открываем новое
            connected = await exchange_manager.connect()
            
            if connected:
                self.healing_attempts += 1
                self.last_healing_time = datetime.now()
                logger.info("✅ Healing: Подключение восстановлено!")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Healing: Ошибка восстановления подключения: {e}")
            return False
    
    async def heal_missing_sl_orders(
        self,
        exchange_manager,
        positions: list
    ) -> bool:
        """Создание отсутствующих SL ордеров"""
        try:
            logger.critical("🔧 Healing: КРИТИЧНО! Создаю отсутствующие SL ордера...")
            
            success_count = 0
            
            for position in positions:
                try:
                    symbol = position['symbol']
                    side = position['side']
                    amount = position['amount']
                    
                    # Рассчитываем SL (консервативно -10%)
                    entry_price = position['entry_price']
                    sl_price_pct = Config.MAX_LOSS_PER_TRADE_PERCENT / Config.LEVERAGE / 100
                    
                    if side == "buy":
                        sl_price = entry_price * (1 - sl_price_pct)
                    else:
                        sl_price = entry_price * (1 + sl_price_pct)
                    
                    # Создаем SL ордер
                    close_side = "sell" if side == "buy" else "buy"
                    sl_order = await exchange_manager.create_stop_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=amount,
                        stop_price=sl_price
                    )
                    
                    if sl_order:
                        position['sl_order_id'] = sl_order['id']
                        success_count += 1
                        logger.info(f"✅ Healing: SL ордер создан для {symbol}")
                    
                except Exception as e:
                    logger.error(f"❌ Healing: Не удалось создать SL для {position.get('symbol')}: {e}")
            
            if success_count > 0:
                self.healing_attempts += 1
                self.last_healing_time = datetime.now()
                logger.info(f"✅ Healing: Создано {success_count} SL ордеров")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Healing: Ошибка создания SL ордеров: {e}")
            return False
    
    async def heal_errors(self, health_monitor) -> bool:
        """Исправление накопленных ошибок"""
        try:
            logger.info("🔧 Healing: Сброс счетчиков ошибок...")
            
            # Сохраняем информацию об ошибках
            error_report = {
                "total_errors": health_monitor.errors_count,
                "api_errors": health_monitor.api_errors,
                "network_errors": health_monitor.network_errors,
                "trading_errors": health_monitor.trading_errors,
                "timestamp": datetime.now()
            }
            
            # Сбрасываем счетчики
            health_monitor.reset_errors()
            
            self.healing_attempts += 1
            self.last_healing_time = datetime.now()
            
            logger.info("✅ Healing: Счетчики ошибок сброшены")
            return True
            
        except Exception as e:
            logger.error(f"❌ Healing: Ошибка сброса счетчиков: {e}")
            return False
    
    async def find_stuck_positions(self, positions: list) -> list:
        """Найти зависшие позиции"""
        stuck = []
        max_hold_time = 86400  # 24 часа
        
        for position in positions:
            if 'open_time' in position:
                hold_time = (datetime.now() - position['open_time']).total_seconds()
                if hold_time > max_hold_time:
                    stuck.append(position)
                    logger.warning(f"⚠️ Healing: Позиция {position['symbol']} открыта {hold_time/3600:.1f}ч!")
        
        return stuck
    
    async def heal_stuck_positions(
        self,
        exchange_manager,
        positions: list
    ) -> bool:
        """Закрытие зависших позиций"""
        try:
            logger.warning("🔧 Healing: Закрытие зависших позиций...")
            
            for position in positions:
                try:
                    symbol = position['symbol']
                    side = position['side']
                    amount = position['amount']
                    
                    # Закрываем рыночным ордером
                    close_side = "sell" if side == "buy" else "buy"
                    await exchange_manager.create_market_order(
                        symbol, close_side, amount
                    )
                    
                    logger.info(f"✅ Healing: Зависшая позиция {symbol} закрыта")
                    
                except Exception as e:
                    logger.error(f"❌ Healing: Ошибка закрытия {position.get('symbol')}: {e}")
            
            self.healing_attempts += 1
            self.last_healing_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"❌ Healing: Ошибка закрытия зависших позиций: {e}")
            return False
    
    def reset_healing_attempts(self):
        """Сброс счетчика попыток исправления"""
        self.healing_attempts = 0
        logger.info("🔧 Healing: Счетчик попыток сброшен")


# Глобальный экземпляр
auto_healing = AutoHealingSystem()

