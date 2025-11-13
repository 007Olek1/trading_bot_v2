#!/usr/bin/env python3
"""
📅 МЕНЕДЖЕР ВАЖНЫХ СОБЫТИЙ (ФРС, макро-новости)
===============================================

Автоматическое управление рисками перед важными событиями:
- Снижение плеча
- Повышение MIN_CONFIDENCE
- Режим "ожидания подтверждений"
- Автоматическая адаптация торговых параметров
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pytz

logger = logging.getLogger(__name__)

WARSAW_TZ = pytz.timezone('Europe/Warsaw')


class FedEventManager:
    """Менеджер важных событий (ФРС, макро-новости)"""
    
    def __init__(self):
        self.warsaw_tz = WARSAW_TZ
        
        # Важные события (можно расширить)
        # Формат: (дата, время UTC, название, уровень важности)
        self.important_events = [
            # Можно добавить события вручную или получать через API
            # Пример: (datetime(2025, 10, 29, 18, 0, 0, tzinfo=pytz.UTC), "Fed Meeting", "HIGH")
        ]
        
        # За сколько часов до события начинать осторожность
        self.hours_before_caution = 6  # 6 часов до события
        self.hours_after_caution = 2    # 2 часа после события
    
    def add_fed_event(self, event_date: datetime, event_name: str = "Fed Meeting", 
                     importance: str = "HIGH"):
        """Добавить событие ФРС или другое важное событие"""
        try:
            # Если дата без timezone, предполагаем UTC
            if event_date.tzinfo is None:
                event_date = pytz.UTC.localize(event_date)
            
            self.important_events.append({
                'date': event_date,
                'name': event_name,
                'importance': importance  # HIGH, MEDIUM, LOW
            })
            
            logger.info(f"📅 Добавлено событие: {event_name} на {event_date.strftime('%Y-%m-%d %H:%M')} UTC")
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления события: {e}")
    
    def check_near_event(self, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[Dict], float]:
        """
        Проверяет, близко ли важное событие
        
        Returns:
            (is_near_event, event_info, hours_until_event)
        """
        try:
            if current_time is None:
                current_time = datetime.now(pytz.UTC)
            elif current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)
            
            for event in self.important_events:
                event_date = event['date']
                hours_until = (event_date - current_time).total_seconds() / 3600
                hours_after = (current_time - event_date).total_seconds() / 3600
                
                # Проверяем: за N часов до события или в течение M часов после
                if (0 <= hours_until <= self.hours_before_caution) or \
                   (0 <= hours_after <= self.hours_after_caution):
                    
                    logger.warning(f"⚠️ БЛИЗКО ВАЖНОЕ СОБЫТИЕ: {event['name']} "
                                 f"(до: {hours_until:.1f}ч, после: {hours_after:.1f}ч)")
                    
                    return True, event, hours_until if hours_until >= 0 else -hours_after
            
            return False, None, 0
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки событий: {e}")
            return False, None, 0
    
    def get_risk_adjustments(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Получает корректировки рисков для текущего времени
        
        Returns:
            {
                'leverage_multiplier': float,  # Множитель для плеча (1.0 = без изменений, 0.5 = в 2 раза меньше)
                'confidence_bonus': float,      # Бонус к MIN_CONFIDENCE (в процентах)
                'position_size_multiplier': float,  # Множитель для размера позиции
                'message': str,                 # Сообщение для пользователя
                'mode': str                     # 'NORMAL', 'CAUTION', 'WAIT'
            }
        """
        try:
            is_near, event_info, hours = self.check_near_event(current_time)
            
            if not is_near:
                return {
                    'leverage_multiplier': 1.0,
                    'confidence_bonus': 0,
                    'position_size_multiplier': 1.0,
                    'message': 'Режим нормальной торговли',
                    'mode': 'NORMAL'
                }
            
            event_name = event_info['name']
            importance = event_info.get('importance', 'HIGH')
            is_before = hours >= 0
            
            # Для HIGH важности - более строгие ограничения
            if importance == 'HIGH':
                if is_before:
                    # До события - очень осторожно
                    if hours <= 2:
                        # За 2 часа или меньше - почти не торгуем
                        return {
                            'leverage_multiplier': 0.4,  # Плечо x5 -> x2
                            'confidence_bonus': +15,      # +15% к уверенности
                            'position_size_multiplier': 0.6,  # Размер позиции -60%
                            'message': f'⚠️ ОЖИДАНИЕ СОБЫТИЯ: {event_name} (через {hours:.1f}ч). '
                                     f'Режим минимальных рисков. Плечо снижено, уверенность повышена.',
                            'mode': 'WAIT'
                        }
                    else:
                        # За 2-6 часов - осторожно
                        return {
                            'leverage_multiplier': 0.6,  # Плечо x5 -> x3
                            'confidence_bonus': +10,      # +10% к уверенности
                            'position_size_multiplier': 0.7,  # Размер позиции -30%
                            'message': f'⚠️ ПРИБЛИЖАЕТСЯ СОБЫТИЕ: {event_name} (через {hours:.1f}ч). '
                                     f'Режим осторожной торговли.',
                            'mode': 'CAUTION'
                        }
                else:
                    # После события (первые 2 часа) - осторожно
                    hours_after = abs(hours)
                    return {
                        'leverage_multiplier': 0.6,  # Плечо x5 -> x3
                        'confidence_bonus': +10,
                        'position_size_multiplier': 0.7,
                        'message': f'⚠️ ПРОИЗОШЛО СОБЫТИЕ: {event_name} ({hours_after:.1f}ч назад). '
                                 f'Ожидание реакции рынка.',
                        'mode': 'CAUTION'
                    }
            
            # Для MEDIUM важности - умеренные ограничения
            elif importance == 'MEDIUM':
                return {
                    'leverage_multiplier': 0.8,
                    'confidence_bonus': +5,
                    'position_size_multiplier': 0.8,
                    'message': f'⚠️ Событие: {event_name}. Умеренная осторожность.',
                    'mode': 'CAUTION'
                }
            
            # Для LOW - минимальные изменения
            else:
                return {
                    'leverage_multiplier': 0.9,
                    'confidence_bonus': +3,
                    'position_size_multiplier': 0.9,
                    'message': f'📅 Событие: {event_name}. Небольшая осторожность.',
                    'mode': 'CAUTION'
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка расчета корректировок рисков: {e}")
            return {
                'leverage_multiplier': 1.0,
                'confidence_bonus': 0,
                'position_size_multiplier': 1.0,
                'message': 'Ошибка расчета корректировок',
                'mode': 'NORMAL'
            }
    
    def get_fed_message(self) -> Optional[str]:
        """Получить сообщение о текущем состоянии важных событий"""
        try:
            is_near, event_info, hours = self.check_near_event()
            
            if not is_near:
                return None
            
            adjustments = self.get_risk_adjustments()
            
            return adjustments['message']
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения сообщения о событиях: {e}")
            return None


# Глобальный экземпляр
fed_event_manager = FedEventManager()


