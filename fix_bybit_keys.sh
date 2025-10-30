#!/bin/bash
# 🔧 Скрипт для обновления Bybit API ключей в .env

ENV_FILE="/Users/aleksandrfilippov/Downloads/.env"

echo "🔑 Обновление Bybit API ключей..."

# Правильные ключи, которые вы давали ранее
NEW_KEY="44SH7IrmIXtkKHgk1i"
NEW_SECRET="xTR5Rq5yj0F6DnynqYldRHq2ZKO6cZFbQQeg"

# Проверяем существование файла
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Файл .env не найден: $ENV_FILE"
    exit 1
fi

# Создаем backup
cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Backup создан"

# Обновляем ключи
sed -i.bak "s/^BYBIT_API_KEY=.*/BYBIT_API_KEY=$NEW_KEY/" "$ENV_FILE"
sed -i.bak "s/^BYBIT_API_SECRET=.*/BYBIT_API_SECRET=$NEW_SECRET/" "$ENV_FILE"

# Удаляем временный .bak файл (если создался)
rm -f "${ENV_FILE}.bak"

echo "✅ API ключи обновлены в $ENV_FILE"
echo ""
echo "Обновленные ключи:"
grep "^BYBIT_API" "$ENV_FILE" | sed 's/=.*/=***/'


