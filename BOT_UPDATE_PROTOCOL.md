# 📋 ПРОТОКОЛ ОБНОВЛЕНИЯ БОТА НА СЕРВЕРЕ

**Версия:** 2.0  
**Дата:** 29.10.2025  
**Статус:** ✅ ОБЯЗАТЕЛЬНЫЙ ПРОЦЕДУРА

---

## 🚨 **ВАЖНО: ВСЕГДА СЛЕДОВАТЬ ЭТОМУ ПРОТОКОЛУ!**

**⚠️ НИКОГДА не обновлять бота без остановки!**  
**⚠️ ВСЕГДА тестировать перед запуском!**  
**⚠️ ВСЕГДА проверять работу после запуска!**

---

## 📋 **ПОШАГОВЫЙ ПРОТОКОЛ:**

### **1️⃣ ОСТАНОВКА РАБОТАЮЩЕГО БОТА**

```bash
# Проверяем статус
ssh -i ~/.ssh/upcloud_trading_bot root@213. converting.116
systemctl status trading-bot

# Останавливаем бота
systemctl stop trading-bot

# Проверяем что процессы остановлены
ps aux | grep -E 'super_bot|python.*bot' | grep -v grep

# Если процессы найдены - убиваем их
pkill -f super_bot_v4_mtf.py
```

**✅ Проверка:** Бот должен быть в статусе `inactive (dead)`

---

### **2️⃣ ЗАГРУЗКА ОБНОВЛЕНИЙ НА СЕРВЕР**

```bash
# Из локальной директории загружаем файлы
cd /Users/aleksandrfilippov/Downloads/trading_bot

# Загружаем основные модули
scp -i ~/.ssh/upcloud_trading_bot \
    super_bot_v4_mtf.py \
    universal_learning_system.py \
    data_storage_system.py \
    ai_ml_system.py \
    root@213.163.199.116:/opt/bot/

# Загружаем вспомогательные модули
scp -i ~/.ssh/upcloud_trading_bot \
    adaptive_parameters.py \
    adaptive_trading_system.py \
    smart_coin_selector.py \
    probability_calculator.py \
    strategy_evaluator.py \
    realism_validator.py \
    telegram_commands_handler.py \
    root@213.163.199.116:/opt/bot/

# Загружаем дополнительные системы
scp -i ~/.ssh/upcloud_trading_bot \
    integrate_intelligent_agents.py \
    advanced_manipulation_detector.py \
    advanced_ml_system.py \
    api_optimizer.py \
    fed_event_manager.py \
    advanced_indicators.py \
    llm_monitor.py \
    root@213.163.199.116:/opt/bot/
```

**✅ Проверка:** Все файлы загружены, размеры совпадают

---

### **3️⃣ ТЕСТИРОВАНИЕ**

```bash
# Подключаемся к сер负极
ssh -i ~/.ssh/upcloud_trading_bot root@213.163.199.116

# Переходим в директорию бота
cd /opt/bot

# Проверяем синтаксис Python
python3 -m py_compile super_bot_v4_mtf.py
python3 -m py_compile universal_learning_system.py
python3 -m py_compile smart_coin_selector.py

# Тестируем импорты
python3 -c "
import sys
sys.path.insert(0, '/opt/bot')
try:
    from super_bot_v4_mtf import SuperBotV4MTF
    from universal_learning_system import UniversalLearningSystem
    from smart_coin_selector import SmartCoinSelector
    print('✅ Все модули импортируются успешно')
except Exception as e:
    print(f'❌ Ошибка импорта: {e}')
    sys.exit(1)
"

# Тестируем систему самообучения (если есть скрипт)
if [ -f verify_universal_learning.py ]; then
    python3 verify_universal_learning.py
fi
```

**✅ Проверка:** Все тесты должны пройти успешно

---

### **4️⃣ ПЕРЕЗАПУСК БОТА**

```bash
# Проверяем конфигурацию systemd
cat /etc/systemd/system/trading-bot.service

# Перезапускаем бота
systemctl start trading-bot

# Проверяем статус
systemctl status trading-bot --no-pager

# Проверяем что процесс запущен
ps aux | grep -E 'super_bot|python.*bot' | grep -v grep
```

**✅ Проверка:** Бот должен быть в статусе `active (running)`

---

### **5️⃣ ПРОВЕРКА НОВЫХ ДОРАБОТОК**

```bash
# Смотрим логи в реальном времени
tail -f /opt/bot/logs/bot.log

# ИЛИ
tail -f /opt/bot/super_bot_v4_mtf.log

# Проверяем что новые функции работают:
# - Выбор монет (100-200)
# - Популярные мемкоины включены
# - Система самообучения активна
# - Moroccan-timeframe анализ работает
# - Telegram команды работают

# Ожидаемые логи:
# ✅ "🎯 Умный селектор отобрал XXX монет"
# ✅ "🧠 UniversalLearningSystem инициализирована"
# ✅ "✅ Умный селектор выбрал XXX символов"
# ✅ "📊 V4.0: Рынок XXX"
```

**✅ Проверка:** Все новые функции работают корректно

---

## 🔍 **ДЕТАЛЬНАЯ ПРОВЕРКА:**

### **Проверка выбора монет:**
```bash
# В логах должно быть:
grep -E "Умный селектор|символов|монет" /opt/bot/super_bot_v4_mtf.log | tail -10
```

**Ожидается:**
- `🎯 Умный селектор отобрал 100-200 монет`
- `🐂 Бычий рынок: анализируем 200 символов`
- `🐻 Медвежий рынок: анализируем 100 символов`

### **Проверка мемкоинов:**
```bash
# В логах ищем популярные мемкоины:
grep -E "DOGE|SHIB|PEPE|FLOKI" /opt/bot/super_bot_v4_mtf.log | tail -5
```

**Ожидается:** Мемкоины не должны быть исключены

### **Проверка системы самообучения:**
```bash
# В логах должно быть:
grep -E "Universal|обучени|правило|pattern" /opt/bot/super_bot_v4_mtf.log | tail -10
```

**Ожидается:**
- `🧠 UniversalLearningSystem инициализирована`
- `🧠 Создано X универсальных паттернов`
- Использование диапазонов, не точных значений

---

## 📊 **ЧЕКЛИСТ ПРОВЕРКИ:**

### **Перед запуском:**
- [ ] Бот остановлен (`systemctl stop trading-bot`)
- [ ] Процессы завершены (`ps aux | grep bot`)
- [ ] Файлы загружены на сервер
- [ ] Синтаксис проверен (`py_compile`)
- [ ] Импорты работают (`python3 -c "import ..."`)

### **После запуска:**
- [ ] Бот запущен (`systemctl status trading-bot`)
- [ ] Процесс работает (`ps aux | grep bot`)
- [ ] Логи без критических ошибок
- [ ] Выбор монет работает (100-200)
- [ ] Мемкоины включены
- [ ] Система самообучения активна
- [ ] Telegram команды работают

---

## 🚨 **В СЛУЧАЕ ПРОБЛЕМ:**

### **Бот не запускается:**
```bash
# Смотрим логи ошибок
journalctl -u trading-bot -n 50 --no-pager

# Проверяем Python ошибки
python3 /opt/bot/super_bot_v4_mtf.py 2>&1 | head -20
```

### **Ошибки импорта:**
```bash
# Проверяем зависимости
pip3 list | grep -E "ccxt|telegram|pandas|numpy"

# Устанавливаем недостающие
pip3 install -r /opt/bot/requirements_bot.txt
```

### **Откат к предыдущей версии:**
```bash
# Останавливаем бота
systemctl stop trading-bot

# Восстанавливаем из backup (если есть)
# Или загружаем предыдущую версию

# Перезапускаем
systemctl start trading-bot
```

---

## ✅ **ЗАКЛЮЧЕНИЕ:**

**ВСЕГДА следуйте этому протоколу:**
1. ✅ Остановить бота
2. ✅ Загрузить обновления
3. ✅ Протестировать
4. ✅ Перезапустить бота
5. ✅ Проверить работу

**⚠️ НИКОГДА не пропускайте шаги!**

---

**Протокол создан и сохранен!** ✅







