"""
🎯 DISCO57 BOT - КОНФИГУРАЦИЯ
Настройки торгового бота для Bybit фьючерсов
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
env_path = Path(__file__).parent.parent / "keys" / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Попытка загрузить из текущей директории


# ═══════════════════════════════════════════════════════════════════
# API КЛЮЧИ
# ═══════════════════════════════════════════════════════════════════
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Проверка наличия ключей
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("❌ BYBIT_API_KEY и BYBIT_API_SECRET должны быть установлены в .env")


# ═══════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ ТОРГОВЛИ
# ═══════════════════════════════════════════════════════════════════
POSITION_SIZE_USD = 1.0  # Размер позиции в USD
LEVERAGE = 10  # Плечо
MAX_CONCURRENT_POSITIONS = 3  # Максимум одновременных сделок

# Риск менеджмент
STOP_LOSS_PERCENT = 20.0  # SL -20% от позиции
TAKE_PROFIT_PERCENT = 10.0  # TP +10% от позиции
USE_TRAILING_TP = True  # Включить Trailing Take Profit
TRAILING_TP_ACTIVATION_PERCENT = 5.0  # Активация trailing при +5%
TRAILING_TP_CALLBACK_PERCENT = 3.0  # Откат для trailing 3%

# Временные лимиты
MAX_POSITION_DURATION_HOURS = 24  # Закрывать позицию через 24 часа


# ═══════════════════════════════════════════════════════════════════
# ТАЙМФРЕЙМЫ
# ═══════════════════════════════════════════════════════════════════
TIMEFRAMES = {
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
PRIMARY_TIMEFRAME = "30m"  # Основной таймфрейм для сигналов
MIN_TIMEFRAME_ALIGNMENT = 2  # Минимум 2 таймфрейма должны подтверждать сигнал


# ═══════════════════════════════════════════════════════════════════
# WATCHLIST - МОНЕТЫ ДЛЯ ТОРГОВЛИ
# ═══════════════════════════════════════════════════════════════════
WATCHLIST = [
    # Топ по капитализации и ликвидности
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
    "UNIUSDT", "ATOMUSDT", "XLMUSDT", "NEARUSDT", "EOSUSDT",
    
    # DeFi и Layer-2
    "AAVEUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT",
    "SEIUSDT", "TIAUSDT", "INJUSDT", "FETUSDT", "RENDERUSDT",
    
    # Популярные альткоины
    "FILUSDT", "ICPUSDT", "TRXUSDT", "TONUSDT", "HBARUSDT",
    "VETUSDT", "ALGOUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    
    # Мемкоины (высокая волатильность)
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT",
    
    # Дополнительные ликвидные
    "BCHUSDT", "ETCUSDT", "LDOUSDT", "IMXUSDT", "GRTUSDT",
    "ORDIUSDT", "STXUSDT", "BLURUSDT", "WLDUSDT", "ARKMUSDT",
]


# ═══════════════════════════════════════════════════════════════════
# ИНДИКАТОРЫ - ПАРАМЕТРЫ
# ═══════════════════════════════════════════════════════════════════
INDICATOR_PARAMS = {
    # EMA
    "ema_short": 20,
    "ema_medium": 50,
    "ema_long": 200,
    
    # RSI
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    
    # Stochastic RSI
    "stoch_rsi_period": 14,
    "stoch_rsi_overbought": 80,
    "stoch_rsi_oversold": 20,
    
    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    # ADX
    "adx_period": 14,
    "adx_min_strength": 25,  # Минимальная сила тренда
    
    # Bollinger Bands
    "bb_period": 20,
    "bb_std": 2.0,
    
    # ATR
    "atr_period": 14,
    
    # Volume
    "volume_sma_period": 20,
}


# ═══════════════════════════════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ
# ═══════════════════════════════════════════════════════════════════
SIGNAL_THRESHOLDS = {
    "min_confidence": 0.70,  # Минимальная уверенность для входа (70%)
    "strong_confidence": 0.85,  # Сильный сигнал (85%)
    "min_volume_ratio": 1.2,  # Минимальное отношение объема к среднему
    "max_spread_percent": 0.5,  # Максимальный спред в %
}


# ═══════════════════════════════════════════════════════════════════
# РЕЖИМЫ РЫНКА
# ═══════════════════════════════════════════════════════════════════
MARKET_MODES = {
    "trending": {
        "adx_min": 25,
        "allow_trades": True,
        "description": "Сильный тренд - активная торговля"
    },
    "ranging": {
        "adx_max": 20,
        "allow_trades": False,  # Не торговать во флэте
        "description": "Флэт - торговля приостановлена"
    },
    "volatile": {
        "atr_multiplier": 1.5,
        "allow_trades": True,
        "description": "Высокая волатильность - торговля с осторожностью"
    },
}


# ═══════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "disco57_bot.log"
TRADES_LOG_FILE = LOG_DIR / "trades.json"

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR


# ═══════════════════════════════════════════════════════════════════
# ЦИКЛ РАБОТЫ
# ═══════════════════════════════════════════════════════════════════
ANALYSIS_INTERVAL_SECONDS = 60  # Анализ каждую минуту
MONITORING_INTERVAL_SECONDS = 10  # Мониторинг позиций каждые 10 секунд

# Использовать testnet для тестирования
USE_TESTNET = False

print("✅ Конфигурация Disco57 загружена")

