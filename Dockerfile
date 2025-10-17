# 🐳 DOCKER IMAGE ДЛЯ TRADING BOT V2.0
FROM python:3.11-slim

# Метаданные
LABEL maintainer="trading_bot_v2"
LABEL version="2.0"
LABEL description="AI-powered crypto trading bot with ML self-learning"

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь код бота
COPY . .

# Создаём необходимые директории
RUN mkdir -p logs ml_data

# Переменные окружения по умолчанию
ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Warsaw

# Healthcheck для Docker
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import os; exit(0 if os.path.exists('logs/bot_v2.log') else 1)"

# Запуск бота
CMD ["python3", "-u", "trading_bot_v2_main.py"]

