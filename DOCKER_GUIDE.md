# 🐳 **DOCKER DEPLOYMENT GUIDE**

## **ЧТО ТАКОЕ DOCKER?**

Docker упаковывает бота и все его зависимости в изолированный контейнер, который работает одинаково на любом сервере.

### **ПРЕИМУЩЕСТВА:**
✅ Изоляция от системных зависимостей  
✅ Автоматический перезапуск при сбоях  
✅ Простое развёртывание на любом сервере  
✅ Ограничение ресурсов (CPU, RAM)  
✅ Логирование и мониторинг  
✅ Лёгкие обновления (git pull + docker-compose restart)  

---

## **УСТАНОВКА DOCKER НА СЕРВЕРЕ**

```bash
# 1. Подключаемся к серверу
ssh -i ~/.ssh/upcloud_trading_bot root@5.22.215.2

# 2. Устанавливаем Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Устанавливаем Docker Compose
apt-get update
apt-get install -y docker-compose-plugin

# 4. Проверяем установку
docker --version
docker compose version
```

---

## **ЗАПУСК БОТА В DOCKER**

### **Первый запуск:**

```bash
# 1. Переходим в директорию проекта
cd /root/trading_bot_v2

# 2. Обновляем код
git pull

# 3. Создаём директории для данных
mkdir -p logs ml_data

# 4. Проверяем что .env файл существует
ls -la .env

# 5. Собираем Docker образ
docker compose build

# 6. Запускаем бота
docker compose up -d

# 7. Проверяем статус
docker compose ps
```

### **Просмотр логов:**

```bash
# Все логи
docker compose logs -f

# Последние 100 строк
docker compose logs --tail=100

# Логи в реальном времени
docker compose logs -f --tail=50
```

### **Управление:**

```bash
# Остановить бота
docker compose stop

# Запустить бота
docker compose start

# Перезапустить бота
docker compose restart

# Остановить и удалить контейнер
docker compose down

# Полная пересборка и перезапуск
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## **ОБНОВЛЕНИЕ БОТА**

```bash
# 1. Остановить текущую версию
docker compose down

# 2. Получить обновления
git pull

# 3. Пересобрать образ
docker compose build

# 4. Запустить новую версию
docker compose up -d

# 5. Проверить логи
docker compose logs -f --tail=50
```

---

## **МОНИТОРИНГ**

### **Статус контейнера:**
```bash
docker compose ps
docker compose top
```

### **Использование ресурсов:**
```bash
docker stats trading_bot_v2
```

### **Healthcheck:**
```bash
docker inspect trading_bot_v2 | grep -A 10 Health
```

---

## **TROUBLESHOOTING**

### **Бот не запускается:**
```bash
# Проверить логи
docker compose logs --tail=100

# Проверить .env файл
cat .env

# Проверить порты
netstat -tulpn | grep LISTEN
```

### **Очистка Docker:**
```bash
# Удалить неиспользуемые образы
docker image prune -a

# Удалить неиспользуемые контейнеры
docker container prune

# Полная очистка
docker system prune -a --volumes
```

### **Пересоздание с нуля:**
```bash
docker compose down -v
docker system prune -a
docker compose build --no-cache
docker compose up -d
```

---

## **BACKUP И ВОССТАНОВЛЕНИЕ**

### **Бэкап данных:**
```bash
# Остановить бота
docker compose stop

# Создать бэкап
tar -czf backup_$(date +%Y%m%d).tar.gz logs/ ml_data/ trade_history.json .env

# Запустить бота
docker compose start
```

### **Восстановление:**
```bash
# Остановить бота
docker compose down

# Распаковать бэкап
tar -xzf backup_20251017.tar.gz

# Запустить бота
docker compose up -d
```

---

## **ЛУЧШИЕ ПРАКТИКИ**

1. **Регулярные бэкапы:**
   - Создавайте бэкап перед каждым обновлением
   - Храните бэкапы минимум 7 дней

2. **Мониторинг:**
   - Проверяйте логи 2 раза в день
   - Следите за использованием ресурсов

3. **Обновления:**
   - Обновляйте бота раз в неделю
   - Тестируйте обновления на тестовом балансе

4. **Безопасность:**
   - Никогда не коммитьте .env в Git
   - Регулярно меняйте API ключи
   - Используйте SSH ключи для доступа

---

## **КОНФИГУРАЦИЯ РЕСУРСОВ**

В `docker-compose.yml` настроены лимиты:
- **CPU:** до 2 ядер (минимум 0.5)
- **RAM:** до 2GB (минимум 512MB)

Для изменения отредактируйте секцию `deploy.resources` в `docker-compose.yml`.

---

## **АВТОЗАПУСК ПРИ ПЕРЕЗАГРУЗКЕ СЕРВЕРА**

Docker Compose уже настроен с `restart: unless-stopped`, поэтому бот автоматически запустится после перезагрузки сервера.

Проверить:
```bash
# Перезагрузить сервер
reboot

# После перезагрузки проверить
docker compose ps
```

---

## **ПОЛЕЗНЫЕ КОМАНДЫ**

```bash
# Войти в контейнер
docker compose exec trading_bot bash

# Выполнить команду в контейнере
docker compose exec trading_bot python3 -c "print('Hello')"

# Скопировать файл из контейнера
docker compose cp trading_bot:/app/logs/bot_v2.log ./

# Скопировать файл в контейнер
docker compose cp ./new_config.py trading_bot:/app/

# Просмотр переменных окружения
docker compose exec trading_bot env
```

