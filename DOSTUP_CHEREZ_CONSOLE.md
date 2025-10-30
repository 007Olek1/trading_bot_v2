# 🔐 ДОСТУП К СЕРВЕРУ ЧЕРЕЗ UPLOUD CONSOLE

**IP:** 5.22.220.105  
**Проблема:** SSH доступ по ключу и паролю не работает

---

## 📋 РЕШЕНИЕ: Используйте UpCloud Console (VNC)

### **ШАГ 1: Откройте панель UpCloud**
1. Зайдите в браузере: **https://hub.upcloud.com/**
2. Войдите в свой аккаунт

### **ШАГ 2: Найдите сервер**
1. В панели найдите сервер с IP **5.22.220.105**
2. Нажмите на сервер, чтобы открыть детали

### **ШАГ 3: Откройте Console**
1. В меню сервера найдите кнопку **"Console"** или **"VNC Console"**
2. Нажмите на неё - откроется веб-консоль

### **ШАГ 4: Войдите в систему**
1. В консоли должен появиться экран входа
2. Введите:
   - **Username:** `root`
   - **Password:** [пароль root из UpCloud панели или тот, что вы настраивали]

---

## 🔧 ПОСЛЕ ВХОДА В КОНСОЛЬ

### **1. Проверьте статус бота:**
```bash
ps aux | grep python | grep -v grep
```

### **2. Проверьте директории:**
```bash
ls -la /opt/bot/
ls -la /root/trading_bot/ 2>/dev/null
```

### **3. Проверьте сервис:**
```bash
systemctl status trading-bot
```

### **4. Посмотрите логи:**
```bash
tail -50 /opt/bot/bot.log
# или
tail -50 /root/trading_bot/bot.log
```

---

## 🔑 НАСТРОЙКА SSH ДОСТУПА (после входа через Console)

После того как войдете через Console, можете добавить SSH ключ:

```bash
# 1. Создать директорию .ssh
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 2. Добавить публичный ключ
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINvt5Nqq602AgsaNjArzniwkEY2uPRXBnqMKLSABZn+I trading_bot_upcloud" >> ~/.ssh/authorized_keys

# 3. Установить права
chmod 600 ~/.ssh/authorized_keys

# 4. Проверить
cat ~/.ssh/authorized_keys
```

После этого SSH доступ через ключ будет работать:
```bash
ssh -i ~/.ssh/upcloud_trading_bot root@5.22.220.105
```

---

## ⚙️ ВКЛЮЧИТЬ ПАРОЛЬН液晶 АУТЕНТИФИКАЦИЮ (если нужно)

Если хотите подключиться по паролю, войдите через Console и выполните:

```bash
# Редактировать конфиг SSH
nano /etc/ssh/sshd_config

# Найти строку:
# PasswordAuthentication no

# Изменить на:
PasswordAuthentication yes

# Сохранить (Ctrl+O, Enter, Ctrl+X)

# Перезапустить SSH
systemctl restart sshd
```

---

## 🆘 ЕСЛИ НЕ МОЖЕТЕ НАЙТИ CONSOLE

1. В панели UpCloud найдите вкладку **"Access"** или **"Доступ"**
2. Ищите **"Console Access"** или **"VNC Access"**
3. Может быть кнопка **"Launch Console"** или **"Open Console"**

---

**После входа через Console отправьте мне результаты проверки бота!** ✅


