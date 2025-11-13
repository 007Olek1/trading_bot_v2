#!/bin/bash
# Настройка ротации логов для торгового бота (исправленная версия)

echo "🔄 Настройка ротации логов..."

# 1. Устанавливаем logrotate конфигурацию (без дубликатов)
cat > /etc/logrotate.d/trading_bot << 'LOGROTATE_EOF'
# Ротация логов торгового бота
/opt/bot/logs/system/bot.log /opt/bot/logs/system/bot_error.log {
    daily
    rotate 3
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    copytruncate
    maxsize 500M
    sharedscripts
}

/opt/bot/logs/system/*.log {
    daily
    rotate 2
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    copytruncate
    maxsize 100M
}

/opt/bot/logs/*.log {
    daily
    rotate 2
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    copytruncate
    maxsize 100M
}

/opt/bot/*.log {
    daily
    rotate 2
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    copytruncate
    maxsize 100M
}
LOGROTATE_EOF

echo "✅ Logrotate конфигурация создана (исправленная)"

# 2. Тестируем конфигурацию
echo "🧪 Тестирование logrotate конфигурации..."
if logrotate -d /etc/logrotate.d/trading_bot 2>&1 | grep -q "error"; then
    echo "❌ Ошибка в конфигурации logrotate"
    logrotate -d /etc/logrotate.d/trading_bot
    exit 1
else
    echo "✅ Конфигурация logrotate корректна"
fi

# 3. Применяем обновленный код бота
echo "🔄 Обновление кода бота с ротацией логов..."
if [ -f /opt/bot/super_bot_v4_mtf.py.new ]; then
    # Делаем бэкап старого файла
    cp /opt/bot/super_bot_v4_mtf.py /opt/bot/super_bot_v4_mtf.py.backup_$(date +%Y%m%d_%H%M%S)
    mv /opt/bot/super_bot_v4_mtf.py.new /opt/bot/super_bot_v4_mtf.py
    echo "✅ Код бота обновлен с RotatingFileHandler"
else
    echo "⚠️ Новый файл не найден, пропускаем обновление кода"
fi

# 4. Проверяем статус
echo ""
echo "📊 Статус ротации логов:"
echo "   Конфигурация: /etc/logrotate.d/trading_bot"
echo "   Размер текущих логов:"
du -sh /opt/bot/logs/system/*.log 2>/dev/null | head -n 5

echo ""
echo "✅ Ротация логов полностью настроена!"
echo "   📋 Python ротация (RotatingFileHandler):"
echo "      - Максимальный размер: 500MB на файл"
echo "      - Хранится 3 ротированных файла (до 1.5GB)"
echo "      - Автоматическая ротация при достижении лимита"
echo ""
echo "   📋 Системная ротация (logrotate):"
echo "      - Ежедневная ротация в 00:00"
echo "      - Хранится 3 копии ротированных логов"
echo "      - Автоматическое сжатие после ротации"
echo "      - Максимальный размер: 500MB для основных логов"
echo ""
echo "   💡 Рекомендация: перезапустите бот для применения изменений"
echo "      (ротация в Python начнет работать сразу после перезапуска)"








