#!/bin/bash
# Оптимизированная настройка ротации логов
cat > /etc/logrotate.d/trading_bot << 'LOGROTATE_EOF'
# Ротация логов торгового бота (оптимизировано)
/opt/bot/logs/system/bot.log {
    daily
    rotate 2
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    copytruncate
    maxsize 200M
}

/opt/bot/logs/system/bot_error.log {
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

/opt/bot/logs/system/*.log {
    daily
    rotate 1
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    copytruncate
    maxsize 50M
}

/opt/bot/logs/*.log {
    daily
    rotate 1
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    copytruncate
    maxsize 50M
}
LOGROTATE_EOF

# Очистка старых логов
find /opt/bot/logs -name "*.log.*" -type f -mtime +7 -delete 2>/dev/null
find /opt/bot/logs -name "*.log.gz" -type f -mtime +7 -delete 2>/dev/null

echo "✅ Ротация настроена"
