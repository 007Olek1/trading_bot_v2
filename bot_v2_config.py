"""
🛡️ УЛЬТРА-БЕЗОПАСНАЯ КОНФИГУРАЦИЯ БОТА V2.0
Без DEMO - максимальная осторожность!
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Конфигурация бота V2.0"""
    
    # ========================================
    # 🔑 API КЛЮЧИ
    # ========================================
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", 0))
    
    # ========================================
    # 💰 РАЗМЕР ПОЗИЦИЙ (УЛЬТРА-КОНСЕРВАТИВНО!)
    # ========================================
    TEST_MODE = False                   # ✅ РЕАЛЬНЫЙ РЕЖИМ АКТИВИРОВАН
    TEST_POSITION_SIZE_USD = 1.0        # $1 на сделку в тесте
    TEST_MAX_TRADES = 3                 # Только 3 тестовых сделки
    
    POSITION_SIZE_USD = 2.0             # $2 после теста
    MAX_POSITIONS = 3                   # МАКСИМУМ 3 позиции одновременно
    LEVERAGE = 5                        # 5x плечо (Cross)
    
    # ========================================
    # 🛡️ ЛИМИТЫ УБЫТКОВ (ЖЕСТКИЕ!)
    # ========================================
    MAX_LOSS_PER_TRADE_PERCENT = 20    # -20% ROI от инвестиций (при 5X = -4% от цены)
    MAX_DAILY_LOSS_USD = 2.0           # $2/день максимум
    MAX_WEEKLY_LOSS_USD = 3.0          # $3/неделю максимум
    CONSECUTIVE_LOSSES_LIMIT = 2       # 2 убытка подряд = СТОП!
    
    # ========================================
    # 🎯 ЦЕЛИ ПРИБЫЛИ
    # ========================================
    TAKE_PROFIT_MIN_PERCENT = 25       # +25% минимум
    REQUIRED_WIN_RATE = 80             # 80% Win Rate требуется
    DAILY_PROFIT_TARGET_USD = 0.5      # $0.50/день цель
    
    # ========================================
    # ⏰ ЧАСТОТА ТОРГОВЛИ
    # ========================================
    TRADING_INTERVAL_SECONDS = 900     # 15 минут между анализами
    COOLDOWN_AFTER_TRADE_SECONDS = 1800  # 30 минут после сделки
    MAX_TRADES_PER_DAY = 10            # Максимум 10 сделок/день
    
    # ========================================
    # 🔍 ФИЛЬТРЫ СИГНАЛОВ (СТРОГИЕ!)
    # ========================================
    MIN_CONFIDENCE_PERCENT = 85        # 85% минимум (только качественные сигналы)
    MIN_SIGNAL_STRENGTH = 0.85         # 85% сила сигнала
    
    # LLM фильтр (временно отключен для обучения)
    USE_LLM_FILTER = False             # False = открываем без LLM проверки
    
    # ========================================
    # 💦 ЛИКВИДНОСТЬ (ДИНАМИЧЕСКИЙ ОТБОР МОНЕТ)
    # ========================================
    # Минимальный суточный оборот в котируемой валюте (USDT)
    LIQUIDITY_MIN_QUOTE_VOLUME_USD = 1_000_000
    # Минимальная цена монеты (исключаем слишком дешевые активы)
    LIQUIDITY_MIN_PRICE_USD = 0.01
    # Максимально допустимый спред между bid/ask в процентах
    LIQUIDITY_MAX_SPREAD_PERCENT = 2.0

    # ========================================
    # 🛡️ БЕЗОПАСНОСТЬ
    # ========================================
    STOP_LOSS_ON_EXCHANGE = True       # ОБЯЗАТЕЛЬНО ордер на бирже!
    TAKE_PROFIT_ON_EXCHANGE = True     # TP тоже на бирже
    EMERGENCY_STOP_ENABLED = True      # Аварийный стоп
    AUTO_CLOSE_ON_ERROR = True         # Закрывать при ошибках
    
    # Если НЕ МОЖЕМ создать SL - НЕ ОТКРЫВАЕМ сделку!
    REQUIRE_SL_ORDER = True
    
    # ========================================
    # 📊 МОНИТОРИНГ
    # ========================================
    HEALTH_CHECK_INTERVAL_SECONDS = 60  # Проверка каждую минуту
    TELEGRAM_ALERTS_ENABLED = True
    LOG_LEVEL = "DEBUG"
    
    # ========================================
    # 📅 РАСПИСАНИЕ
    # ========================================
    WEEKEND_REST = True
    SUNDAY_EVENING_TRADING = True
    SUNDAY_TRADING_START_HOUR = 19
    SUNDAY_TRADING_END_HOUR = 21
    TIMEZONE = "Europe/Warsaw"
    
    # 🌙 НОЧНОЙ РЕЖИМ (100% автономность)
    NIGHT_MODE_ENABLED = True          # Работа без присмотра
    NIGHT_EXTRA_SAFETY = True          # Дополнительные проверки ночью
    NIGHT_HEARTBEAT_INTERVAL = 3600    # Heartbeat каждый час
    
    # ========================================
    # 📊 СПИСОК ТОП 100 ПОПУЛЯРНЫХ МОНЕТ
    # ========================================
    TOP_100_SYMBOLS = [
        # ТОП 20 - Основные
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT',
        'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'AVAX/USDT:USDT',
        'LINK/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'LTC/USDT:USDT', 'APT/USDT:USDT',
        'NEAR/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'SUI/USDT:USDT', 'TIA/USDT:USDT',
        
        # ТОП 40 - DeFi и Layer 1/2
        'AAVE/USDT:USDT', 'MKR/USDT:USDT', 'COMP/USDT:USDT', 'SNX/USDT:USDT', 'CRV/USDT:USDT',
        'INJ/USDT:USDT', 'RUNE/USDT:USDT', 'FTM/USDT:USDT', 'ALGO/USDT:USDT', 'XLM/USDT:USDT',
        'VET/USDT:USDT', 'FIL/USDT:USDT', 'ICP/USDT:USDT', 'HBAR/USDT:USDT', 'GRT/USDT:USDT',
        'THETA/USDT:USDT', 'EOS/USDT:USDT', 'ASTR/USDT:USDT', 'XTZ/USDT:USDT', 'FLOW/USDT:USDT',
        
        # ТОП 60 - Популярные альты
        'SAND/USDT:USDT', 'MANA/USDT:USDT', 'AXS/USDT:USDT', 'IMX/USDT:USDT', 'APE/USDT:USDT',
        'LDO/USDT:USDT', 'STX/USDT:USDT', 'RPL/USDT:USDT', 'BLUR/USDT:USDT', 'WOO/USDT:USDT',
        'GMT/USDT:USDT', 'GAL/USDT:USDT', 'FXS/USDT:USDT', 'DYDX/USDT:USDT', 'GMX/USDT:USDT',
        'PEPE/USDT:USDT', 'SHIB/USDT:USDT', 'FLOKI/USDT:USDT', 'BONK/USDT:USDT', 'WIF/USDT:USDT',
        
        # ТОП 80 - Перспективные
        'RNDR/USDT:USDT', 'FET/USDT:USDT', 'AGIX/USDT:USDT', 'OCEAN/USDT:USDT', 'ROSE/USDT:USDT',
        'AR/USDT:USDT', 'STORJ/USDT:USDT', 'ANKR/USDT:USDT', 'ENS/USDT:USDT', 'LRC/USDT:USDT',
        'BAT/USDT:USDT', 'ZRX/USDT:USDT', '1INCH/USDT:USDT', 'SUSHI/USDT:USDT', 'YFI/USDT:USDT',
        'BAL/USDT:USDT', 'KNC/USDT:USDT', 'ZIL/USDT:USDT', 'QTUM/USDT:USDT', 'ZEN/USDT:USDT',
        
        # ТОП 100 - Стабильные
        'KAVA/USDT:USDT', 'CELO/USDT:USDT', 'ONE/USDT:USDT', 'ZEC/USDT:USDT', 'DASH/USDT:USDT',
        'WAVES/USDT:USDT', 'ICX/USDT:USDT', 'ONT/USDT:USDT', 'IOST/USDT:USDT', 'IOTX/USDT:USDT',
        'RVN/USDT:USDT', 'MINA/USDT:USDT', 'CHZ/USDT:USDT', 'ENJ/USDT:USDT', 'LPT/USDT:USDT',
        'DENT/USDT:USDT', 'RSR/USDT:USDT', 'SKL/USDT:USDT', 'CELR/USDT:USDT', 'CTK/USDT:USDT'
    ]
    
    # ========================================
    # 🔧 ТЕХНИЧЕСКИЕ
    # ========================================
    EXCHANGE_FEE_TAKER = 0.00055       # 0.055% комиссия Bybit
    SLIPPAGE_TOLERANCE = 0.001         # 0.1% проскальзывание
    
    # Логирование
    LOG_FILE = "logs/bot_v2.log"       # Логи в папке logs/
    
    # ========================================
    # 🧪 РЕЖИМЫ РАБОТЫ
    # ========================================
    # Этап 1: TEST_MODE = True, 3 сделки по $1
    # Этап 2: TEST_MODE = False, обычная торговля
    
    @classmethod
    def get_position_size(cls):
        """Получить размер позиции в зависимости от режима"""
        if cls.TEST_MODE:
            return cls.TEST_POSITION_SIZE_USD
        return cls.POSITION_SIZE_USD
    
    @classmethod
    def validate_config(cls):
        """Проверка конфигурации перед запуском"""
        errors = []
        
        if not cls.BYBIT_API_KEY:
            errors.append("❌ BYBIT_API_KEY не установлен")
        if not cls.BYBIT_API_SECRET:
            errors.append("❌ BYBIT_API_SECRET не установлен")
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("❌ TELEGRAM_BOT_TOKEN не установлен")
        if cls.MAX_LOSS_PER_TRADE_PERCENT >= cls.TAKE_PROFIT_MIN_PERCENT:
            errors.append("❌ SL должен быть меньше TP!")
        if cls.LEVERAGE > 5:
            errors.append("❌ Леверидж > 5x слишком рискован!")
        if cls.POSITION_SIZE_USD > 5:
            errors.append("❌ Размер позиции > $5 слишком рискован!")
        
        return errors


