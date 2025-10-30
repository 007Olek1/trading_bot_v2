#!/bin/bash
# 🧹 Очистка рабочего пространства

echo "🧹 Начинаем очистку рабочего пространства..."
echo ""

# Удаляем временные файлы и логи
echo "📁 Удаляем логи..."
rm -f *.log
rm -f test_*.log
rm -f expert_*.log
rm -f bot_*.log
echo "✅ Логи удалены"

# Удаляем Python кэш
echo "🐍 Удаляем Python кэш..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "✅ Python кэш удален"

# Удаляем временные тестовые файлы (оставляем экспертное тестирование)
echo "📄 Удаляем временные файлы..."
rm -f test_*.py
rm -f *test*.json
# Оставляем expert_system_test.py и expert_test_report.json для справки

# Удаляем старые текстовые файлы инструкций (некоторые могут быть устаревшими)
echo "📝 Очищаем документацию..."
# Оставляем важные файлы, удаляем только временные

echo ""
echo "✅ Очистка завершена!"
echo ""
echo "📊 Оставлены важные файлы:"
echo "   - Все .py модули"
echo "   - expert_system_test.py"
echo "   - expert_test_report.json"
echo "   - requirements_bot.txt"
echo ""


