#!/bin/bash

# Создание шаблона .env файла для нового сервера

echo "📝 Создание шаблона .env файла..."

cat > /tmp/env_template.txt << 'ENVEOF'
# Bybit API Keys
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Trading Parameters
LEVERAGE=5
TRADE_SIZE_USDT=5
MAX_OPEN_TRADES=3
CONFIDENCE_THRESHOLD=0.75

# Exchange Fees
EXCHANGE_FEE_TAKER=0.00055
EXCHANGE_FEE_MAKER=0.0002
ENVEOF

echo "✅ Шаблон создан в /tmp/env_template.txt"


