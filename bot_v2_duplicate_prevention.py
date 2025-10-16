"""
🛡️ МОДУЛЬ ПРЕДОТВРАЩЕНИЯ ДУБЛИКАТОВ V1.0
Предотвращает открытие нескольких позиций по одной монете
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import asyncio
from threading import Lock

logger = logging.getLogger(__name__)


class DuplicatePreventionManager:
    """Менеджер предотвращения дубликатов позиций"""
    
    def __init__(self, cooldown_hours: int = 6):
        self.cooldown_hours = cooldown_hours
        self.position_locks: Dict[str, asyncio.Lock] = {}  # Локи для предотвращения race conditions
        self.active_positions: Dict[str, datetime] = {}  # Активные позиции
        self.cooldowns: Dict[str, datetime] = {}  # Cooldown после закрытия
        self.last_sides: Dict[str, str] = {}  # Последнее направление сделки
        self._file_lock = Lock()  # Лок для файловых операций
        self.state_file = 'duplicate_prevention_state.json'
        
        # Загружаем сохраненное состояние
        self.load_state()
    
    async def can_open_position(self, symbol: str, current_positions: List[Dict]) -> Tuple[bool, str]:
        """
        Проверяет можно ли открыть позицию по символу
        
        Returns:
            (can_open, reason)
        """
        # Создаем лок для символа если его нет
        if symbol not in self.position_locks:
            self.position_locks[symbol] = asyncio.Lock()
        
        async with self.position_locks[symbol]:
            # 1. Проверка активной позиции
            if any(p['symbol'] == symbol for p in current_positions):
                return False, f"Уже есть открытая позиция по {symbol}"
            
            # 2. Проверка в нашем трекере активных позиций
            if symbol in self.active_positions:
                # Проверяем не устарела ли запись (более 24 часов)
                if datetime.now() - self.active_positions[symbol] < timedelta(hours=24):
                    return False, f"Позиция по {symbol} уже отслеживается как активная"
            
            # 3. Проверка cooldown
            if symbol in self.cooldowns:
                time_passed = datetime.now() - self.cooldowns[symbol]
                hours_passed = time_passed.total_seconds() / 3600
                
                if hours_passed < self.cooldown_hours:
                    hours_remaining = self.cooldown_hours - hours_passed
                    last_side = self.last_sides.get(symbol, 'unknown')
                    return False, f"{symbol} в cooldown (осталось {hours_remaining:.1f}ч, последняя сделка: {last_side})"
            
            return True, "OK"
    
    async def register_position_opening(self, symbol: str, side: str):
        """Регистрирует открытие позиции"""
        async with self.position_locks.get(symbol, asyncio.Lock()):
            self.active_positions[symbol] = datetime.now()
            self.last_sides[symbol] = side.lower()
            
            # Удаляем из cooldown если был там
            if symbol in self.cooldowns:
                del self.cooldowns[symbol]
            
            self.save_state()
            logger.info(f"✅ Позиция {symbol} {side} зарегистрирована как активная")
    
    async def register_position_closing(self, symbol: str, side: str, pnl: float):
        """Регистрирует закрытие позиции"""
        async with self.position_locks.get(symbol, asyncio.Lock()):
            # Удаляем из активных
            if symbol in self.active_positions:
                del self.active_positions[symbol]
            
            # Добавляем в cooldown
            self.cooldowns[symbol] = datetime.now()
            self.last_sides[symbol] = side.lower()
            
            # При убытке увеличиваем cooldown
            if pnl < 0:
                # Увеличиваем cooldown на 50% при убытке
                extended_time = datetime.now() - timedelta(hours=self.cooldown_hours * 0.5)
                self.cooldowns[symbol] = extended_time
                logger.info(f"⏰ {symbol} добавлена в расширенный cooldown после убытка")
            else:
                logger.info(f"⏰ {symbol} добавлена в стандартный cooldown после прибыли")
            
            self.save_state()
    
    def sync_with_exchange_positions(self, exchange_positions: List[Dict]):
        """Синхронизирует состояние с позициями биржи"""
        # Получаем список символов с биржи
        exchange_symbols = {pos['symbol'] for pos in exchange_positions if float(pos.get('contracts', 0)) > 0}
        
        # Добавляем новые позиции
        for pos in exchange_positions:
            symbol = pos['symbol']
            if float(pos.get('contracts', 0)) > 0 and symbol not in self.active_positions:
                self.active_positions[symbol] = datetime.now()
                self.last_sides[symbol] = pos.get('side', 'unknown').lower()
                logger.info(f"📊 Синхронизация: {symbol} добавлена как активная позиция")
        
        # Удаляем закрытые позиции
        closed_symbols = []
        for symbol in list(self.active_positions.keys()):
            if symbol not in exchange_symbols:
                closed_symbols.append(symbol)
                del self.active_positions[symbol]
                # Добавляем в cooldown с уменьшенным временем
                self.cooldowns[symbol] = datetime.now() - timedelta(hours=self.cooldown_hours * 0.7)
                logger.info(f"📊 Синхронизация: {symbol} перемещена в cooldown")
        
        if exchange_positions or closed_symbols:
            self.save_state()
    
    def cleanup_old_entries(self):
        """Очищает устаревшие записи"""
        current_time = datetime.now()
        
        # Очищаем старые cooldowns
        expired_cooldowns = []
        for symbol, cooldown_time in list(self.cooldowns.items()):
            if current_time - cooldown_time > timedelta(hours=self.cooldown_hours):
                expired_cooldowns.append(symbol)
                del self.cooldowns[symbol]
                if symbol in self.last_sides:
                    del self.last_sides[symbol]
        
        # Очищаем очень старые активные позиции (более 48 часов)
        old_positions = []
        for symbol, open_time in list(self.active_positions.items()):
            if current_time - open_time > timedelta(hours=48):
                old_positions.append(symbol)
                del self.active_positions[symbol]
        
        if expired_cooldowns or old_positions:
            logger.info(f"🧹 Очистка: {len(expired_cooldowns)} cooldowns, {len(old_positions)} старых позиций")
            self.save_state()
    
    def save_state(self):
        """Сохраняет состояние в файл"""
        with self._file_lock:
            try:
                state = {
                    'active_positions': {
                        symbol: timestamp.isoformat() 
                        for symbol, timestamp in self.active_positions.items()
                    },
                    'cooldowns': {
                        symbol: timestamp.isoformat() 
                        for symbol, timestamp in self.cooldowns.items()
                    },
                    'last_sides': self.last_sides,
                    'saved_at': datetime.now().isoformat(),
                    'cooldown_hours': self.cooldown_hours
                }
                
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                logger.debug(f"💾 Состояние сохранено: {len(self.active_positions)} активных, {len(self.cooldowns)} в cooldown")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения состояния: {e}")
    
    def load_state(self):
        """Загружает состояние из файла"""
        with self._file_lock:
            try:
                if os.path.exists(self.state_file):
                    with open(self.state_file, 'r') as f:
                        state = json.load(f)
                    
                    # Восстанавливаем активные позиции
                    for symbol, timestamp_str in state.get('active_positions', {}).items():
                        timestamp = datetime.fromisoformat(timestamp_str)
                        # Только если позиция не слишком старая
                        if datetime.now() - timestamp < timedelta(hours=48):
                            self.active_positions[symbol] = timestamp
                    
                    # Восстанавливаем cooldowns
                    for symbol, timestamp_str in state.get('cooldowns', {}).items():
                        timestamp = datetime.fromisoformat(timestamp_str)
                        # Только если cooldown еще действует
                        if datetime.now() - timestamp < timedelta(hours=self.cooldown_hours):
                            self.cooldowns[symbol] = timestamp
                    
                    # Восстанавливаем последние направления
                    self.last_sides = state.get('last_sides', {})
                    
                    logger.info(
                        f"📂 Загружено состояние: {len(self.active_positions)} активных, "
                        f"{len(self.cooldowns)} в cooldown"
                    )
                    
                    # Сразу очищаем устаревшие
                    self.cleanup_old_entries()
                    
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки состояния: {e}")
    
    def get_status_report(self) -> Dict:
        """Возвращает отчет о текущем состоянии"""
        current_time = datetime.now()
        
        cooldowns_info = []
        for symbol, cooldown_time in self.cooldowns.items():
            time_passed = current_time - cooldown_time
            hours_remaining = max(0, self.cooldown_hours - time_passed.total_seconds() / 3600)
            cooldowns_info.append({
                'symbol': symbol,
                'hours_remaining': hours_remaining,
                'last_side': self.last_sides.get(symbol, 'unknown')
            })
        
        return {
            'active_positions': list(self.active_positions.keys()),
            'cooldowns': cooldowns_info,
            'total_active': len(self.active_positions),
            'total_cooldowns': len(self.cooldowns)
        }


# Глобальный экземпляр
duplicate_prevention = DuplicatePreventionManager()