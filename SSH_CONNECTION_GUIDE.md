# 🔐 SSH ПОДКЛЮЧЕНИЕ К СЕРВЕРУ

## 📋 ИНФОРМАЦИЯ О СЕРВЕРЕ

```
IP адрес: 5.22.215.2
Пользователь: root
SSH порт: 22
 трудов
```

---

## 🔑 SSH КЛЮЧ

### Приватный ключ (для подключения):

```bash
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACDb7eTaqutNgILGjY~/K854sJBGNrj0VwZ6jCi0gAWZ/iAAAAJhH+o8pR/qP
KQAAAAtzc2gtZWQyNTUxOQAAACDb7eTaqutNgILGjYwK854sJBGNrj0VwZ6jCi0gAWZ/iA
AAAEDhIL4u4xMtwoYveuekdtFxGo7SwfnDcpfzF7aPREJKy9vt5Nqq602AgsaNjArzniwk
EY2uPRXBnqMKLSABZn+IAAAAE3RyYWRpbmdfYm90X3VwY2xvdWQBAg==
-----END OPENSSH PRIVATE KEY-----
```

**Расположение на вашем Mac:**
```
~/.ssh/upcloud_trading_bot
```

### Публичный ключ (уже на сервере):

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINvt5Nqq602AgsaNjArzniwkEY2uPRXBnqMKLSABZn+I trading_bot_upcloud
```

---

## 💻 КОМАНДА ДЛЯ ПОДКЛЮЧЕНИЯ (macOS)

### Вариант 1: Простое подключение

```bash
ssh -i ~/.ssh/upcloud_trading_bot root@5.22.215.2
```

### Вариант 2: С отключенной проверкой ключа хоста (если первый раз)

```bash
ssh -i ~/.ssh/upcloud_trading_bot -o StrictHostKeyChecking=no root@5.22.215.2
```

### Вариант 3: С таймаутом (если проблемы с подключением)

```bash
ssh -i ~/.ssh/upcloud_trading_bot -o ConnectTimeout=30 root@5.22.215.2
```

---

## 🚀 БЫСТРОЕ ПОДКЛЮЧЕНИЕ

Скопируйте и выполните в Terminal (macOS):

```bash
ssh -i ~/.ssh/upcloud_trading_bot root@5.22.215.2
```

---

## 📝 ПОЛЕЗНЫЕ КОМАНДЫ ПОСЛЕ ПОДКЛЮЧЕНИЯ

### Проверка статуса бота:

```bash
systemctl status trading-bot
```

### Просмотр логов:

```bash
tail -50 /root/trading_bot/bot.log
```

### Просмотр ошибок:

```bash
tail -50 /root/trading_bot/bot_error.log
```

### Перезапуск бота:

```bash
systemctl restart trading-bot
```

### Проверка процессов:

```bash
ps aux | grep python
```

### Проверка использования ресурсов:

```bash
df -h
free -h
```

---

## 🔧 НАСТРОЙКА SSH (ОПЦИОНАЛЬНО)

Для удобства можно создать алиас в `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Добавьте:

```
Host upcloud-trading
    HostName 5.22.215.2
    User root
    IdentityFile ~/.ssh/upcloud_trading_bot
    StrictHostKeyChecking no
```

После этого можно подключаться просто:

```bash
ssh upcloud-trading
```

---

## ⚠️ ВОССТАНОВЛЕНИЕ ДОСТУПА

Если подключение не работает:

1. **Проверьте интернет:**
   ```bash
   ping 5.22.215.2
   ```

2. **Проверьте права на ключ:**
   ```bash
   chmod 600 ~/.ssh/upcloud_trading_bot
   ```

3. **Проверьте, что ключ существует:**
   ```bash
   ls -la ~/.ssh/upcloud_trading_bot
   ```

4. **Попробуйте с подробным выводом:**
   ```bash
   ssh -v -i ~/.ssh/upcloud_trading_bot root@5.22.215.2
   ```

---

## 🆘 ЕСЛИ ДОСТУП ПОТЕРЯН

Если SSH не работает, возможно нужно:

1. Проверить статус сервера в панели UpCloud
2. Переустановить SSH ключ через панель управления
3. Использовать пароль (если настроен)

---

**ГОТОВО! Используйте команду выше для подключения!** ✅



