# 🔑 ИСПРАВЛЕНИЕ BYBIT API КЛЮЧЕЙ

**Проблема:** Bybit возвращает ошибку "API key is invalid"

---

## 📋 ТЕКУЩИЕ КЛЮЧИ В .env

В файле `/Users/aleksandrfilippov/Downloads/.env` сейчас:
```
BYBIT_API_KEY=XRqraBihucPcf4zOfU
BYBIT_API_SECRET=mtXi8k4JiaOtjxOTr7qDol7QPtAx1QzkIQ8h
```

Эти ключи **НЕВАЛИДНЫ** - Bybit возвращает ошибку `10003: API key is invalid`

---

## ✅ ПРАВИЛЬНЫЕ КЛЮЧИ (которые вы давали ранее)

```
BYBIT_API_KEY=44SH7IrmIXtkKHgk1i
BYBIT_API_SECRET=xTR5Rq5yj0F6DnynqYldRHq2ZKO6cZFbQQeg
```

---

## 🔧 КАК ИСПРАВИТЬ

### Вариант 1: Автоматически (через скрипт)

```bash
cd /Users/aleksandrfilippov/Downloads/trading_bot
./fix_bybit_keys.sh
```

### Вариант 2: Вручную

1. Откройте файл `.env`:
```bash
nano /Users/aleksandrfilippov/Downloads/.env
```

2. Замените строки:
```bash
BYBIT_API_KEY=44SH7IrmIXtkKHgk1i
BYBIT_API_SECRET=xTR5Rq5yj0F6DnynqYldRHq2ZKO6cZFbQQeg
```

3. Сохраните файл (Ctrl+O, Enter, Ctrl+X)

---

## ⚠️ ВАЖНО

- После обновления ключей **перезапустите бота** (если он запущен)
- Убедитесь что ключи имеют правильные **права доступа** в Bybit:
  - ✅ **Read** - для получения данных
  - ✅ **Trade** - для торговли (если нужна реальная торговля)

---

## 🧪 ПРОВЕРКА КЛЮЧЕЙ

После обновления проверьте:
```bash
cd /Users/aleksandrfilippov/Downloads/trading_bot
python3 comprehensive_system_test.py
```

Если ключи правильные - ошибка "API key is invalid" исчезнет.

---

**После обновления ключей система будет работать на 100%!**


