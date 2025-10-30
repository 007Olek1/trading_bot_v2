# 🚀 БЫСТРЫЕ КОМАНДЫ ДЛЯ СЕРВЕРА

## 🔌 ПОДКЛЮЧЕНИЕ

```bash
ssh -i ~/.ssh/upcloud_trading_bot root@213.163.199.116
```

---

## 📊 МОНИТОРИНГ

### Статус бота
```bash
systemctl status trading-bot
```

### Логи в реальном времени
```bash
tail -f /opt/bot/logs/bot.log
```

### Логи ошибок
```bash
tail -f /opt/bot/logs/bot_error.log
```

### Последние 50 строк логов
```bash
tail -50 /opt/bot/logs/bot.log
```

---

## 🔄 УПРАВЛЕНИЕ

### Перезапуск бота
```bash
systemctl restart trading-bot
```

### Остановка бота
```bash
systemctl stop trading-bot
```

### Запуск бота
```bash
systemctl start trading-bot
```

### Включить автозапуск
```bash
systemctl enable trading-bot
```

---

## 📁 РАБОЧИЕ ДИРЕКТОРИИ

```bash
cd /opt/bot              # Основная директория
ls -la /opt/bot/         # Список файлов
cat /opt/bot/api.env     # Просмотр конфигурации
```

---

## 🔍 БЫСТРАЯ ДИАГНОСТИКА

```bash
# Проверка процессов
ps aux | grep python

# Проверка памяти
free -h

# Проверка места на диске
df -h

# Проверка сетевых соединений
netstat -tuln | grep 443
```


