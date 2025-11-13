#!/usr/bin/env python3
"""
📅 Скрипт для добавления события ФРС на сегодня
"""

import sys
import pytz
from datetime import datetime, timedelta

# Добавляем событие ФРС на сегодня
WARSAW_TZ = pytz.timezone('Europe/Warsaw')
UTC = pytz.UTC

def add_fed_event_today():
    """Добавить событие ФРС на сегодня"""
    try:
        # Импортируем модуль бота
        sys.path.insert(0, '/opt/bot')
        from fed_event_manager import FedEventManager
        
        manager = FedEventManager()
        
        # Сегодня в 18:00 UTC (обычное время заседаний ФРС)
        # Это примерно 20:00 по Варшаве
        today = datetime.now(UTC).date()
        event_time = datetime.combine(today, datetime.strptime("18:00", "%H:%M").time())
        event_time = UTC.localize(event_time)
        
        # Добавляем событие
        manager.add_fed_event(
            event_date=event_time,
            event_name="Fed Meeting (Решение по ставке)",
            importance="HIGH"
        )
        
        print(f"✅ Событие ФРС добавлено на {event_time.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"   (Варшава: {event_time.astimezone(WARSAW_TZ).strftime('%Y-%m-%d %H:%M')})")
        
        # Проверяем корректировки
        adjustments = manager.get_risk_adjustments()
        print(f"\n📊 Текущие корректировки рисков:")
        print(f"   Режим: {adjustments['mode']}")
        print(f"   Множитель плеча: {adjustments['leverage_multiplier']}")
        print(f"   Множитель размера: {adjustments['position_size_multiplier']}")
        print(f"   Бонус уверенности: +{adjustments['confidence_bonus']}%")
        print(f"\n   {adjustments['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    add_fed_event_today()


