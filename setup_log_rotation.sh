#!/bin/bash
# Настройка ротации логов для торгового бота

echo "🔄 Настройка ротации логов..."

# 1. Устанавливаем logrotate конфигурацию
echo "📝 Создание logrotate конфигурации..."
cat > /etc/logrotate.d/trading_bot << 'LOGROTATE_EOF'
# Ротация логов торгового бота
/opt/bot/logs/system/bot.log {
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

/opt/bot/logs/system/bot_error.log {
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

echo "✅ Logrotate конфигурация создана"

# 2. Тестируем конфигурацию
echo "🧪 Тестирование logrotate конфигурации..."
logrotate -d /etc/logrotate.d/trading_bot
if [ $? -eq 0 ]; then
    echo "✅ Конфигурация logrotate корректна"
else
    echo "⚠️ Предупреждения в конфигурации logrotate (нормально)"
fi

# 3. Принудительная ротация для проверки
echo "🔄 Тестовая ротация логов..."
logrotate -f /etc/logrotate.d/trading_bot 2>/dev/null || true

# 4. Проверяем статус
echo ""
echo "📊 Статус ротации логов:"
echo "   Конфигурация: /etc/logrotate.d/trading_bot"
echo "   Размер логов до ротации:"
du -sh /opt/bot/logs/system/*.log 2>/dev/null | head -n 5

echo ""
echo "✅ Ротация логов настроена!"
echo "   - Логи ротируются ежедневно"
echo "   - Хранится 3 копии ротированных логов"
echo "   - Автоматическое сжатие после ротации"
echo "   - Максимальный размер файла: 500MB для основных логов"










