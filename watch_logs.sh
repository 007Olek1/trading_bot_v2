#!/bin/bash
# Просмотр логов бота в реальном времени с цветовой фильтрацией

echo "📊 ПРОСМОТР ЛОГОВ БОТА В РЕАЛЬНОМ ВРЕМЕНИ"
echo "=" | head -c 70 && echo ""
echo ""
echo "Файл: super_bot_v4_mtf.log"
echo "Для остановки: Ctrl+C"
echo ""
echo "=" | head -c 70 && echo ""

# Показываем последние 20 строк перед началом
echo "📋 Последние 20 строк:"
tail -20 super_bot_v4_mtf.log
echo ""
echo "🔄 Теперь следите за обновлениями в реальном времени..."
echo "=" | head -c 70 && echo ""
echo ""

# Запускаем tail -f
tail -f super_bot_v4_mtf.log
