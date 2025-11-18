"""
🛑 CIRCUIT BREAKER - ЗАЩИТА ОТ СЕРИИ УБЫТКОВ
Автоматическая остановка торговли при серии убытков
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging


class CircuitBreaker:
    """Circuit Breaker для защиты от серии убытков"""
    
    def __init__(
        self,
        max_consecutive_losses: int = 5,
        max_daily_loss_usd: float = 5.0,
        max_weekly_loss_usd: float = 15.0,
        cooldown_hours: int = 24,
        logger: logging.Logger = None
    ):
        """
        Args:
            max_consecutive_losses: Максимум подряд убыточных сделок
            max_daily_loss_usd: Максимальный дневной убыток
            max_weekly_loss_usd: Максимальный недельный убыток
            cooldown_hours: Часов до возобновления после срабатывания
            logger: Логгер
        """
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss = max_daily_loss_usd
        self.max_weekly_loss = max_weekly_loss_usd
        self.cooldown_hours = cooldown_hours
        self.logger = logger or logging.getLogger(__name__)
        
        # Состояние
        self.is_active = True
        self.triggered_at = None
        self.trigger_reason = None
        
        # История сделок
        self.trade_history: List[Dict] = []
        
        # Счётчики
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
    
    def add_trade(self, pnl_usd: float, success: bool, symbol: str = None):
        """
        Добавляет сделку и проверяет условия срабатывания
        
        Args:
            pnl_usd: Прибыль/убыток в USD
            success: Успешна ли сделка
            symbol: Символ (опционально)
        """
        trade = {
            'pnl': pnl_usd,
            'success': success,
            'symbol': symbol,
            'timestamp': datetime.now()
        }
        
        self.trade_history.append(trade)
        
        # Обновляем счётчики
        if success:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Обновляем дневной/недельный убыток
        self._update_period_losses()
        
        # Проверяем условия срабатывания
        self._check_triggers()
    
    def _update_period_losses(self):
        """Обновляет дневной и недельный убыток"""
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        # Фильтруем сделки
        recent_trades = [t for t in self.trade_history if t['timestamp'] > week_ago]
        
        # Дневной убыток
        daily_trades = [t for t in recent_trades if t['timestamp'] > day_ago]
        self.daily_loss = sum(t['pnl'] for t in daily_trades if t['pnl'] < 0)
        
        # Недельный убыток
        self.weekly_loss = sum(t['pnl'] for t in recent_trades if t['pnl'] < 0)
    
    def _check_triggers(self):
        """Проверяет условия срабатывания circuit breaker"""
        if not self.is_active:
            return
        
        # 1. Проверка подряд идущих убытков
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._trigger(
                f"Серия убытков: {self.consecutive_losses} подряд "
                f"(макс: {self.max_consecutive_losses})"
            )
            return
        
        # 2. Проверка дневного убытка
        if abs(self.daily_loss) >= self.max_daily_loss:
            self._trigger(
                f"Дневной убыток: ${abs(self.daily_loss):.2f} "
                f"(макс: ${self.max_daily_loss})"
            )
            return
        
        # 3. Проверка недельного убытка
        if abs(self.weekly_loss) >= self.max_weekly_loss:
            self._trigger(
                f"Недельный убыток: ${abs(self.weekly_loss):.2f} "
                f"(макс: ${self.max_weekly_loss})"
            )
            return
    
    def _trigger(self, reason: str):
        """
        Срабатывает circuit breaker
        
        Args:
            reason: Причина срабатывания
        """
        self.is_active = False
        self.triggered_at = datetime.now()
        self.trigger_reason = reason
        
        self.logger.error(
            f"🛑 CIRCUIT BREAKER СРАБОТАЛ!\n"
            f"   Причина: {reason}\n"
            f"   Торговля остановлена на {self.cooldown_hours}ч"
        )
    
    def check_can_trade(self) -> tuple[bool, Optional[str]]:
        """
        Проверяет можно ли торговать
        
        Returns:
            (можно_торговать, причина_если_нельзя)
        """
        # Если активен - можно торговать
        if self.is_active:
            return True, None
        
        # Проверяем cooldown
        if self.triggered_at:
            cooldown_end = self.triggered_at + timedelta(hours=self.cooldown_hours)
            now = datetime.now()
            
            if now >= cooldown_end:
                # Cooldown закончился - возобновляем
                self._reset()
                return True, None
            else:
                # Ещё в cooldown
                remaining = cooldown_end - now
                hours = remaining.total_seconds() / 3600
                return False, f"Circuit breaker активен. Осталось: {hours:.1f}ч"
        
        return False, self.trigger_reason
    
    def _reset(self):
        """Сбрасывает circuit breaker после cooldown"""
        self.is_active = True
        self.triggered_at = None
        self.trigger_reason = None
        self.consecutive_losses = 0
        
        self.logger.info(
            f"✅ Circuit breaker сброшен. Торговля возобновлена."
        )
    
    def force_reset(self):
        """Принудительный сброс (для ручного управления)"""
        self._reset()
        self.logger.warning("⚠️ Circuit breaker сброшен ВРУЧНУЮ")
    
    def get_status(self) -> Dict:
        """
        Получает текущий статус
        
        Returns:
            Словарь со статусом
        """
        can_trade, reason = self.check_can_trade()
        
        status = {
            'can_trade': can_trade,
            'is_active': self.is_active,
            'consecutive_losses': self.consecutive_losses,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
            'total_trades': len(self.trade_history),
        }
        
        if not can_trade:
            status['reason'] = reason
            status['triggered_at'] = self.triggered_at.isoformat() if self.triggered_at else None
            
            if self.triggered_at:
                cooldown_end = self.triggered_at + timedelta(hours=self.cooldown_hours)
                remaining = cooldown_end - datetime.now()
                status['cooldown_remaining_hours'] = remaining.total_seconds() / 3600
        
        return status
    
    def get_statistics(self) -> Dict:
        """
        Получает статистику сделок
        
        Returns:
            Словарь со статистикой
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
            }
        
        df = pd.DataFrame(self.trade_history)
        
        wins = df[df['success'] == True]
        losses = df[df['success'] == False]
        
        return {
            'total_trades': len(df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'max_consecutive_losses': self.consecutive_losses,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
        }
    
    def should_reduce_risk(self) -> bool:
        """
        Проверяет нужно ли снизить риск (предупреждение)
        
        Returns:
            True если приближаемся к лимитам
        """
        # Проверяем приближение к лимитам (80%)
        if self.consecutive_losses >= self.max_consecutive_losses * 0.8:
            return True
        
        if abs(self.daily_loss) >= self.max_daily_loss * 0.8:
            return True
        
        if abs(self.weekly_loss) >= self.max_weekly_loss * 0.8:
            return True
        
        return False


if __name__ == "__main__":
    # Тест
    print("🧪 Тестирование Circuit Breaker\n")
    
    breaker = CircuitBreaker(
        max_consecutive_losses=5,
        max_daily_loss_usd=5.0,
        cooldown_hours=1  # 1 час для теста
    )
    
    print("📊 Тест 1: Серия убытков")
    for i in range(6):
        breaker.add_trade(pnl_usd=-1.0, success=False, symbol=f"TEST{i}")
        status = breaker.get_status()
        print(f"   Сделка {i+1}: Подряд убытков: {status['consecutive_losses']}")
        
        if not status['can_trade']:
            print(f"   🛑 {status['reason']}")
            break
    print()
    
    # Сброс для следующего теста
    breaker.force_reset()
    
    print("📊 Тест 2: Дневной лимит убытков")
    for i in range(6):
        breaker.add_trade(pnl_usd=-1.2, success=False, symbol=f"TEST{i}")
        status = breaker.get_status()
        print(f"   Сделка {i+1}: Дневной убыток: ${abs(status['daily_loss']):.2f}")
        
        if not status['can_trade']:
            print(f"   🛑 {status['reason']}")
            break
    print()
    
    # Проверка предупреждения
    breaker.force_reset()
    print("📊 Тест 3: Предупреждение о снижении риска")
    for i in range(4):
        breaker.add_trade(pnl_usd=-1.0, success=False)
        if breaker.should_reduce_risk():
            print(f"   ⚠️ После {i+1} убытков: Рекомендуется снизить риск!")
    print()
    
    # Статистика
    print("📊 Статистика:")
    stats = breaker.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Тест завершён!")
