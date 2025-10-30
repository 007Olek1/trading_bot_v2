#!/bin/bash
# 🧹 ПОЛНАЯ ОЧИСТКА РАБОЧЕГО ПРОСТРАНСТВА

echo "🧹 ПОЛНАЯ ОЧИСТКА РАБОЧЕГО ПРОСТРАНСТВА"
echo "========================================"
echo ""

# Удаляем старые текстовые файлы с инструкциями (устаревшие)
echo "📝 Удаляем устаревшие инструкции..."
rm -f ВСЕ_ДАННЫЕ_СЕРВЕРА.txt
rm -f ВЧЕРАШНЯЯ_РАБОТА_НА_СЕРВЕРЕ.txt
rm -f ИНСТРУКЦИЯ_НОВЫЙ_СЕРВЕР.txt
rm -f КОМАНДЫ_ДЛЯ_CONSOLE.txt
rm -f КОМАНДЫ_ПОСЛЕ_ПОДКЛЮЧЕНИЯ.txt
rm -f ПРОСТОЙ_ПЛАН.txt
rm -f РАЗВЕРТЫВАНИЕ_ЗАВЕРШЕНО.txt
rm -f ПОЛНАЯ_ИСТОРИЯ_СЕРВЕРОВ.txt
rm -f NOVY_SERVER.txt
rm -f CURRENT_SERVER.txt
rm -f SERVER_STATUS.txt
rm -f QUICK_RESTORE.txt
rm -f SSH_KEYS.txt
echo "✅ Устаревшие инструкции удалены"

# Удаляем старые markdown файлы с отчетами
echo "📄 Удаляем старые отчеты..."
rm -f DEEP_CLEANUP_SUCCESS.md
rm -f WORKSPACE_CLEANUP_SUCCESS.md
rm -f GITHUB_PUSH_SUCCESS.md
rm -f SYSTEM_AUDIT_COMPLETE.md
rm -f RESTORE_ACCESS_5.22.220.105.md
rm -f SERVER_GITHUB_SYNC_REPORT.md
rm -f КРАТКАЯ_СВОДКА.md
rm -f ИТОГОВЫЙ_СТАТУС.md
rm -f ПРОВЕРКА_СООТВЕТСТВИЯ_СПЕЦИФИКАЦИИ.md
echo "✅ Старые отчеты удалены"

# Оставляем только важные файлы
echo ""
echo "📊 Оставляем важные файлы:"
echo "   ✅ Все .py модули"
echo "   ✅ expert_system_test.py"
echo "   ✅ expert_test_report.json"
echo "   ✅ FINAL_EXPERT_REPORT.md"
echo "   ✅ requirements_bot.txt"
echo "   ✅ essential_*.sh скрипты"
echo ""

# Подсчет оставшихся файлов
PYTHON_FILES=$(find . -maxdepth 1 -name "*.py" | wc -l | tr -d ' ')
echo "📁 Итого Python файлов: $PYTHON_FILES"
echo ""
echo "✅ Очистка завершена!"


