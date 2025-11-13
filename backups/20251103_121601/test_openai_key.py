#!/usr/bin/env python3
import os
import sys

# Загружаем .env
env_file = "/opt/bot/.env"
if os.path.exists(env_file):
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip("\"'")

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✅ OPENAI_API_KEY успешно загружен из .env")
    print(f"   Длина: {len(openai_key)} символов")
    print(f"   Префикс: {openai_key[:10]}...")
    
    # Проверяем подключение
    try:
        import requests
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {openai_key}"},
            timeout=5
        )
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Подключение к OpenAI API работает!")
            print(f"📊 Доступно моделей: {len(models.get('data', []))}")
        else:
            print(f"❌ Ошибка подключения: {response.status_code}")
            print(f"   Ответ: {response.text[:200]}")
    except Exception as e:
        print(f"⚠️ Не удалось проверить подключение: {e}")
else:
    print("❌ OPENAI_API_KEY не найден")
    sys.exit(1)




