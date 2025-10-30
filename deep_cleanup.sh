#!/bin/bash
# 🔧 Глубокая очистка рабочего пространства

echo "🧹 Начинаю глубокую очистку..."

# Удаляем логи
echo "📋 Удаление логов..."
find . -type f -name "*.log" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null
find . -type f -name "*.log.*" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null

# Удаляем Python кеш
echo "🐍 Удаление Python кеша..."
find . -type d -name "__pycache__" -not -path "./venv/*" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null
find . -type f -name "*.pyo" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null

# Удаляем временные файлы
echo "📁 Удаление временных файлов..."
find . -type f -name "*.tmp" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null
find . -type f -name "*~" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null
find . -type f -name ".DS_Store" -not -path "./venv/*" -not -path "./.git/*" -delete 2>/dev/null

# Удаляем старые тестовые файлы (кроме основных)
echo "🗑️ Удаление старых тестовых файлов..."
rm -f test_*.py.backup comprehensive_test_results.log 2>/dev/null

echo "✅ Очистка завершена!"


