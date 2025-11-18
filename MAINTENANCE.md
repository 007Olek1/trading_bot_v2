# 🔧 ОБСЛУЖИВАНИЕ TRADING BOT V4.0 MTF

## 📊 Мониторинг системы

### Проверка состояния бота
```bash
# Полная проверка системы
/root/trading_bot_v4_mtf/system_check.sh

# Статус сервиса
systemctl status trading-bot.service

# Логи в реальном времени
journalctl -u trading-bot.service -f

# Последние 100 строк логов
tail -100 /root/trading_bot_v4_mtf/logs/trading_bot_v4.log
```

---

## 🧹 Очистка и обслуживание

### Автоматическая очистка
Настроена автоматическая очистка каждый день в 4:00 утра через cron.

### Ручная очистка
```bash
# Запустить очистку вручную
/root/trading_bot_v4_mtf/cleanup.sh

# Проверить размер логов
du -sh /root/trading_bot_v4_mtf/logs/

# Проверить общий размер проекта
du -sh /root/trading_bot_v4_mtf/
```

### Что очищается автоматически:
- ✅ Python cache (`__pycache__`, `*.pyc`)
- ✅ Старые сжатые логи (>30 дней)
- ✅ Временные файлы (`*.tmp`, `*.bak`)
- ✅ Старые тестовые логи (>60 дней)
- ✅ Обрезка основного лога если >10MB

---

## 📝 Ротация логов

### Настроена через logrotate:

**Основные логи** (`trading_bot_v4.log`):
- Ротация: каждый день
- Хранится: 7 дней
- Сжатие: да

**Логи сделок** (`trades.json`):
- Ротация: каждую неделю
- Хранится: 4 недели
- Сжатие: да

**Тестовые логи**:
- Ротация: раз в месяц
- Хранится: 3 месяца
- Сжатие: да

### Проверка logrotate:
```bash
# Проверить конфигурацию
cat /etc/logrotate.d/trading-bot

# Тестовый запуск (без изменений)
logrotate -d /etc/logrotate.d/trading-bot

# Принудительная ротация
logrotate -f /etc/logrotate.d/trading-bot
```

---

## 💾 Управление местом на диске

### Проверка использования:
```bash
# Свободное место
df -h /

# Размер проекта
du -sh /root/trading_bot_v4_mtf/

# Топ-10 самых больших файлов
find /root/trading_bot_v4_mtf -type f -exec du -h {} + | sort -rh | head -10
```

### Рекомендации:
- 📊 Проект должен занимать **~150-200MB**
- 📝 Логи не должны превышать **50MB**
- 💾 Свободное место на диске: минимум **5GB**

---

## 🔄 Обновление бота

### Обновление кода:
```bash
# 1. Остановить бота
systemctl stop trading-bot.service

# 2. Сделать бэкап
cp -r /root/trading_bot_v4_mtf /root/trading_bot_v4_mtf_backup_$(date +%Y%m%d)

# 3. Обновить файлы (загрузить новые через scp)

# 4. Перезапустить
systemctl start trading-bot.service

# 5. Проверить
systemctl status trading-bot.service
```

---

## 📊 Мониторинг производительности

### CPU и Memory:
```bash
# Использование ресурсов ботом
ps aux | grep "[p]ython.*main.py"

# Топ процессов
top -b -n 1 | head -20
```

### Сетевые подключения:
```bash
# Активные подключения бота
lsof -p $(pgrep -f "python.*main.py") -a -i
```

---

## 🚨 Решение проблем

### Бот не запускается:
```bash
# Проверить логи
journalctl -u trading-bot.service -n 50

# Проверить конфигурацию
python3 -c "import config; print('OK')"

# Проверить зависимости
source venv/bin/activate && pip list
```

### Логи растут слишком быстро:
```bash
# Проверить размер
du -sh logs/

# Обрезать вручную
tail -n 5000 logs/trading_bot_v4.log > logs/trading_bot_v4.log.tmp
mv logs/trading_bot_v4.log.tmp logs/trading_bot_v4.log
```

### Нет места на диске:
```bash
# Запустить очистку
/root/trading_bot_v4_mtf/cleanup.sh

# Удалить старые бэкапы
rm -rf /root/trading_bot_v4_mtf_backup_*

# Очистить systemd логи
journalctl --vacuum-time=7d
```

---

## ⏰ Расписание обслуживания

### Автоматические задачи (cron):
```
0 4 * * * - Очистка старых данных (cleanup.sh)
```

### Ручные проверки:
- **Ежедневно**: `/stats` в Telegram
- **Еженедельно**: `system_check.sh`
- **Ежемесячно**: Проверка свободного места

---

## 📞 Команды быстрого доступа

```bash
# Алиасы (добавить в ~/.bashrc)
alias bot-status='systemctl status trading-bot.service'
alias bot-logs='journalctl -u trading-bot.service -f'
alias bot-check='/root/trading_bot_v4_mtf/system_check.sh'
alias bot-clean='/root/trading_bot_v4_mtf/cleanup.sh'
alias bot-restart='systemctl restart trading-bot.service'
```

---

## ✅ Чеклист обслуживания

### Ежедневно:
- [ ] Проверить `/stats` в Telegram
- [ ] Убедиться что бот активен

### Еженедельно:
- [ ] Запустить `system_check.sh`
- [ ] Проверить размер логов
- [ ] Проверить свободное место

### Ежемесячно:
- [ ] Проверить Win Rate
- [ ] Оптимизировать параметры при необходимости
- [ ] Обновить watchlist

---

**Последнее обновление**: 16.11.2025
