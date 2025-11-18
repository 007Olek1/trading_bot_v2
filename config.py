"""
🚀 TRADING BOT V4.0 MTF - КОНФИГУРАЦИЯ
Super Bot с Multi-Timeframe анализом
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# API КЛЮЧИ
# ═══════════════════════════════════════════════════════════════════
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Проверка наличия ключей
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("❌ BYBIT_API_KEY и BYBIT_API_SECRET должны быть установлены в .env")


# ═══════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ ТОРГОВЛИ
# ═══════════════════════════════════════════════════════════════════
POSITION_SIZE_USD = 1.0  # Размер позиции в USD
LEVERAGE = 10  # Плечо x10
MAX_CONCURRENT_POSITIONS = 3  # Максимум 3 одновременных позиции

# ═══════════════════════════════════════════════════════════════════
# АДАПТИВНЫЙ TAKE PROFIT - ПОДТЯГИВАНИЕ ОТ +10% ДО +100% ROI
# ═══════════════════════════════════════════════════════════════════
# Адаптивный TP: начинаем с +10% ROI, подтягиваем до +100% ROI
ADAPTIVE_TP_ENABLED = True
MIN_TP_ROI = 10.0  # Минимальная цель: +10% ROI (1% от цены при 10x)
MAX_TP_ROI = 100.0  # Максимальная цель: +100% ROI (10% от цены при 10x)
TP_TRAIL_STEP_ROI = 10.0  # Шаг подтягивания: каждые +10% ROI

# Старый параметр для совместимости
MIN_PROFIT_TARGET_PERCENT = 1.0  # 1% от цены = 10% ROI при 10x

# Stop Loss
STOP_LOSS_MAX_USD = 1.0  # Максимальный убыток в USD

# Trailing Stop Loss (активируется после достижения прибыли)
USE_TRAILING_SL = True
TRAILING_SL_ACTIVATION_PERCENT = 1.0  # Активируется после +1.0% прибыли (быстрая защита)
TRAILING_SL_CALLBACK_PERCENT = 2.5    # Откат 2.5% от максимума (больше свободы)

# Временные лимиты
MAX_POSITION_DURATION_HOURS = 48  # Закрывать позицию через 48 часов


# ═══════════════════════════════════════════════════════════════════
# 📊 УНИВЕРСАЛЬНЫЕ MTF ТАЙМФРЕЙМЫ
# 5m ⏩ 15m ⏩ 1h ⏩ 4h (адаптивные для любого рынка)
# ═══════════════════════════════════════════════════════════════════
TIMEFRAMES = {
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
}
PRIMARY_TIMEFRAME = "5m"  # Основной таймфрейм для быстрых сигналов
MIN_TIMEFRAME_ALIGNMENT = 3  # Минимум 3 из 4 таймфреймов


# ═══════════════════════════════════════════════════════════════════
# 🎯 УНИВЕРСАЛЬНАЯ СТРАТЕГИЯ
# 💹 Только Тренд + Объём + Bollinger Bands
# ═══════════════════════════════════════════════════════════════════
STRATEGIES = {
    "trend_volume_bb": {
        "enabled": True,
        "weight": 1.0,  # 100% - единственная стратегия
        "description": "Универсальная: Тренд + Объём + Bollinger Bands"
    },
    "manipulation_detector": {
        "enabled": False,  # Отключено для упрощения
        "weight": 0.0,
        "description": "Детектор манипуляций (отключен)"
    },
    "global_trend": {
        "enabled": False,  # Отключено для упрощения
        "weight": 0.0,
        "description": "Глобальный тренд (отключен)"
    }
}


# ═══════════════════════════════════════════════════════════════════
# ДИНАМИЧЕСКОЕ СКАНИРОВАНИЕ РЫНКА
# ═══════════════════════════════════════════════════════════════════
USE_DYNAMIC_SCANNER = True  # Использовать динамическое сканирование
MIN_VOLUME_24H_USD = 2000000  # Минимальный объём за 24ч: $2M
MAX_SYMBOLS_TO_SCAN = 50  # Максимум монет для сканирования
UPDATE_WATCHLIST_HOURS = 6  # Обновлять список каждые 6 часов

# ═══════════════════════════════════════════════════════════════════
# КОМИССИИ И РАСЧЁТЫ
# ═══════════════════════════════════════════════════════════════════
ACCOUNT_FOR_FEES = True  # Учитывать комиссии в расчётах
BYBIT_TAKER_FEE = 0.0006  # 0.06% комиссия taker
BYBIT_MAKER_FEE = 0.0002  # 0.02% комиссия maker

# ═══════════════════════════════════════════════════════════════════
# ЕЖЕДНЕВНЫЕ ОТЧЁТЫ
# ═══════════════════════════════════════════════════════════════════
DAILY_REPORT_ENABLED = True  # Включить ежедневные отчёты
DAILY_REPORT_TIME = "09:00"  # Время отправки отчёта (UTC+1)

# ═══════════════════════════════════════════════════════════════════
# WATCHLIST - МОНЕТЫ ДЛЯ ТОРГОВЛИ (используется если USE_DYNAMIC_SCANNER=False)
# ═══════════════════════════════════════════════════════════════════
WATCHLIST = [
    # Топ по капитализации и ликвидности
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
    "UNIUSDT", "ATOMUSDT", "XLMUSDT", "NEARUSDT",
    
    # DeFi и Layer-2
    "AAVEUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT",
    "SEIUSDT", "TIAUSDT", "INJUSDT", "RENDERUSDT", "JUPUSDT",
    
    # Популярные альткоины
    "FILUSDT", "ICPUSDT", "TRXUSDT", "TONUSDT", "HBARUSDT",
    "VETUSDT", "ALGOUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    
    # Мемкоины (высокая волатильность)
    "DOGEUSDT", "WIFUSDT",
    
    # Gaming
    "GALAUSDT", "ENJUSDT",
    
    # Дополнительные ликвидные
    "BCHUSDT", "ETCUSDT", "LDOUSDT", "IMXUSDT", "GRTUSDT",
    "ORDIUSDT", "STXUSDT", "BLURUSDT", "WLDUSDT", "ARKMUSDT",
    
    # Новые перспективные (2024-2025)
    "PYTHUSDT", "JTOUSDT", "DYMUSDT", "ALTUSDT", "PORTALUSDT",
    "ACEUSDT", "NFPUSDT", "XAIUSDT", "MANTAUSDT",
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
    "volume_spike_multiplier": 1.5,  # Всплеск объёма
}


# ═══════════════════════════════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ
# ═══════════════════════════════════════════════════════════════════
SIGNAL_THRESHOLDS = {
    "min_confidence": 0.65,  # Минимальная уверенность для входа (65% - смягчено)
    "strong_confidence": 0.80,  # Сильный сигнал (80%)
    "min_volume_ratio": 1.3,  # Минимальное отношение объема к среднему (смягчено)
    "max_spread_percent": 0.3,  # Максимальный спред в %
}


# ═══════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "trading_bot_v4.log"
TRADES_LOG_FILE = LOG_DIR / "trades.json"

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR


# ═══════════════════════════════════════════════════════════════════
# ЦИКЛ РАБОТЫ
# ⏱️ Анализ: каждые 15 мин
# 📊 Мониторинг: каждую минуту
# ═══════════════════════════════════════════════════════════════════
ANALYSIS_INTERVAL_SECONDS = 900  # Анализ каждые 15 минут (900 секунд)
MONITORING_INTERVAL_SECONDS = 60  # Мониторинг позиций каждую минуту (60 секунд)

# Использовать testnet для тестирования
USE_TESTNET = False


print("✅ Конфигурация Trading Bot V4.0 - Универсальная стратегия загружена")
print("🎯 Режим: Универсальный (5m-15m-1h-4h)")
print("📊 Адаптивный TP: +10% до +100% ROI")
