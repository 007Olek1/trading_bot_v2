#!/bin/bash
# 🧹 Очистка и тестирование бота

echo "🧹 ОЧИСТКА И ТЕСТИРОВАНИЕ DISCO57 BOT"
echo "======================================"

# Очистка Python кэша
echo ""
echo "1️⃣ Очистка Python кэша..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "✅ Кэш очищен"

# Очистка backup файлов
echo ""
echo "2️⃣ Очистка backup файлов..."
find . -maxdepth 1 -name "*.backup" -type f -delete 2>/dev/null
find . -name "*.bak" -type f -delete 2>/dev/null
echo "✅ Backup файлы удалены"

# Очистка временных файлов
echo ""
echo "3️⃣ Очистка временных файлов..."
find . -name "*.tmp" -type f -delete 2>/dev/null
find . -name "*.swp" -type f -delete 2>/dev/null
find . -name ".DS_Store" -type f -delete 2>/dev/null
echo "✅ Временные файлы удалены"

echo ""
echo "✅ ОЧИСТКА ЗАВЕРШЕНА"

