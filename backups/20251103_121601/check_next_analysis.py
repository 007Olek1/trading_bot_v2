#!/usr/bin/env python3
from datetime import datetime, timedelta
import pytz

warsaw_tz = pytz.timezone("Europe/Warsaw")
now = datetime.now(warsaw_tz)

# Анализ каждые 15 минут (00, 15, 30, 45)
current_minute = now.minute
current_second = now.second

# Следующее время анализа - округление вверх до ближайших 15 минут
next_minute = ((current_minute // 15) + 1) * 15

if next_minute >= 60:
    next_hour = (now.hour + 1) % 24
    if next_hour == 0:
        next_day = now.day + 1
    else:
        next_day = now.day
    next_minute = 0
else:
    next_hour = now.hour
    next_day = now.day

next_analysis = warsaw_tz.localize(
    datetime(now.year, now.month, next_day, next_hour, next_minute, 0)
)
time_until_next = (next_analysis - now).total_seconds()

print(f"\n⏰ ТЕКУЩЕЕ ВРЕМЯ: {now.strftime('%H:%M:%S')}")
print(f"📅 СЛЕДУЮЩИЙ АНАЛИЗ: {next_analysis.strftime('%H:%M:%S')}")
minutes_left = int(time_until_next // 60)
seconds_left = int(time_until_next % 60)
print(f"⏱️ ЧЕРЕЗ: {minutes_left} минут {seconds_left} секунд")
print(f"\n🔄 РАСПИСАНИЕ:")
print(f"   • Анализ рынка: каждые 15 минут")
print(f"   • Времена: :00, :15, :30, :45 каждого часа")
print(f"   • Примеры: 13:00, 13:15, 13:30, 13:45, 14:00...")




