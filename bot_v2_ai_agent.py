"""
🤖 AI АГЕНТ ДЛЯ БОТА V2.0
Мониторинг производительности и здоровья бота
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from bot_v2_config import Config

logger = logging.getLogger(__name__)


class TradingBotAgent:
    """AI агент для мониторинга бота"""
    
    def __init__(self):
        self.trade_history: List[Dict[str, Any]] = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        
        logger.info("🤖 AI агент инициализирован")
    
    def record_trade(self, trade: Dict[str, Any]):
        """Записать результат сделки"""
        self.trade_history.append(trade)
        self.total_trades += 1
        
        profit = trade.get('profit', 0)
        self.total_profit += profit
        
        if profit > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            logger.info(f"✅ Агент: Прибыльная сделка, серия побед: {self.consecutive_wins}")
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            logger.warning(f"❌ Агент: Убыточная сделка, серия убытков: {self.consecutive_losses}")
    
    def should_allow_new_trade(self, signal_confidence: float, balance: float) -> Tuple[bool, str]:
        """
        Решение агента: разрешить ли новую сделку?
        
        Returns:
            (allow, reason)
        """
        # Проверка 1: Серия убытков
        if self.consecutive_losses >= 2:
            return False, f"🚫 Агент заблокировал: {self.consecutive_losses} убытка подряд"
        
        # Проверка 2: Win Rate
        if self.total_trades >= 5:
            win_rate = self.winning_trades / self.total_trades
            if win_rate < 0.60:  # Меньше 60%
                return False, f"📉 Агент заблокировал: Win Rate {win_rate:.0%} < 60%"
        
        # Проверка 3: Уверенность
        if signal_confidence < Config.MIN_CONFIDENCE_PERCENT / 100:
            return False, f"🎯 Агент заблокировал: Уверенность {signal_confidence:.0%} < {Config.MIN_CONFIDENCE_PERCENT}%"
        
        # Проверка 4: Баланс
        if balance < Config.get_position_size() * 3:
            return False, f"💰 Агент заблокировал: Низкий баланс ${balance:.2f}"
        
        # Все проверки пройдены
        logger.info(f"✅ Агент РАЗРЕШИЛ сделку: уверенность {signal_confidence:.0%}, Win Rate {self.get_win_rate():.0%}")
        return True, "OK"
    
    def get_win_rate(self) -> float:
        """Получить Win Rate"""
        if self.total_trades == 0:
            return 1.0
        return self.winning_trades / self.total_trades
    
    def get_profit_factor(self) -> float:
        """Получить Profit Factor"""
        if not self.trade_history:
            return 0.0
        
        gross_profit = sum(t.get('profit', 0) for t in self.trade_history if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in self.trade_history if t.get('profit', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Получить отчет о производительности"""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.get_win_rate(),
            "total_profit": self.total_profit,
            "profit_factor": self.get_profit_factor(),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "avg_profit": self.total_profit / self.total_trades if self.total_trades > 0 else 0
        }


class BotHealthMonitor:
    """Мониторинг здоровья бота"""
    
    def __init__(self):
        self.errors_count = 0
        self.last_error_time = None
        self.api_errors = 0
        self.network_errors = 0
        self.trading_errors = 0
        self.health_checks_failed = 0
        self.last_successful_trade = None
        self.last_analysis_time = datetime.now()
        
        logger.info("🏥 Health Monitor инициализирован")
    
    def record_error(self, error_type: str, error_message: str):
        """Записать ошибку"""
        self.errors_count += 1
        self.last_error_time = datetime.now()
        
        if "api" in error_type.lower() or "bybit" in error_type.lower():
            self.api_errors += 1
        elif "network" in error_type.lower() or "connection" in error_type.lower():
            self.network_errors += 1
        elif "trade" in error_type.lower() or "order" in error_type.lower():
            self.trading_errors += 1
        
        logger.warning(f"🏥 Health Monitor: Ошибка зафиксирована [{error_type}]: {error_message}")
    
    def record_successful_analysis(self):
        """Записать успешный анализ"""
        self.last_analysis_time = datetime.now()
    
    def record_successful_trade(self):
        """Записать успешную сделку"""
        self.last_successful_trade = datetime.now()
    
    def is_healthy(self, open_positions_count: int = 0, max_positions: int = 3) -> Tuple[bool, str]:
        """
        Проверка здоровья бота
        
        Args:
            open_positions_count: Количество открытых позиций
            max_positions: Максимальное количество позиций
        
        Returns:
            (is_healthy, reason)
        """
        # Проверка 1: Слишком много ошибок
        if self.errors_count >= 10:
            return False, f"❌ Критично: {self.errors_count} ошибок!"
        
        # Проверка 2: Много ошибок за короткое время
        if self.last_error_time:
            time_since_error = (datetime.now() - self.last_error_time).total_seconds()
            if time_since_error < 60 and self.errors_count >= 5:
                return False, f"❌ Критично: {self.errors_count} ошибок за минуту!"
        
        # Проверка 3: Давно не было анализа (только если позиции НЕ заполнены)
        time_since_analysis = (datetime.now() - self.last_analysis_time).total_seconds()
        
        # Если позиции заполнены - увеличиваем таймаут до 30 минут
        max_time_without_analysis = 1800 if open_positions_count >= max_positions else 600
        
        if time_since_analysis > max_time_without_analysis:
            minutes = int(time_since_analysis / 60)
            if open_positions_count >= max_positions:
                return True, f"⚠️ Нет анализа {minutes} мин (позиции {open_positions_count}/{max_positions})"
            else:
                return False, f"⚠️ Нет анализа {minutes} минут!"
        
        # Проверка 4: Много API ошибок
        if self.api_errors >= 5:
            return False, f"❌ Критично: {self.api_errors} API ошибок!"
        
        return True, "✅ Бот здоров"
    
    def get_health_report(self, open_positions_count: int = 0, max_positions: int = 3) -> Dict[str, Any]:
        """
        Получить отчет о здоровье
        
        Args:
            open_positions_count: Количество открытых позиций
            max_positions: Максимальное количество позиций
        """
        is_healthy, health_status = self.is_healthy(open_positions_count, max_positions)
        
        return {
            "is_healthy": is_healthy,
            "health_status": health_status,
            "total_errors": self.errors_count,
            "api_errors": self.api_errors,
            "network_errors": self.network_errors,
            "trading_errors": self.trading_errors,
            "last_error_time": self.last_error_time,
            "last_analysis_time": self.last_analysis_time,
            "last_successful_trade": self.last_successful_trade,
            "time_since_last_analysis": (datetime.now() - self.last_analysis_time).total_seconds()
        }
    
    def reset_errors(self):
        """Сброс счетчика ошибок (после исправления)"""
        self.errors_count = 0
        self.api_errors = 0
        self.network_errors = 0
        self.trading_errors = 0
        logger.info("🏥 Health Monitor: Счетчики ошибок сброшены")


# Глобальные экземпляры
trading_bot_agent = TradingBotAgent()
health_monitor = BotHealthMonitor()


