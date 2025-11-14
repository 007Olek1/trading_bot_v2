#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Загружаем .env
env_file = Path("/opt/bot/.env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value.strip().strip("\"\'")

print("🔍 ПРОВЕРКА ДОРАБОТОК:")
print("="*60)

# Проверяем код
with open("/opt/bot/super_bot_v4_mtf.py", "r") as f:
    content = f.read()

checks = [
    ("🎭 Детектор манипуляций", "🎭 ПРИОРИТЕТ #1: ДЕТЕКТОР МАНИПУЛЯЦИЙ" in content or "ManipulationDetector.detect_manipulation" in content),
    ("📊 Стратегия бокового рынка", "БОКОВОЙ РЫНОК" in content or "is_sideways" in content),
    ("⏰ Ограничение 24 часа", "ОГРАНИЧЕНИЕ ВРЕМЕНИ УДЕРЖАНИЯ: 24 часа" in content or "max_hold_time = timedelta(hours=24)" in content),
    ("🎯 TP логика", "TP: старт +1% → трейлинг 0.5% → +4% ($1 гарантированно)" in content or "BASE_TARGET = 1.0" in content),
    ("🤖 OpenAI API ключ", os.getenv("OPENAI_API_KEY") is not None),
    ("🧹 Автоочистка", Path("/opt/bot/auto_cleanup_system.py").exists()),
]

for name, status in checks:
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")

print("="*60)










