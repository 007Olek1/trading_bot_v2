#!/usr/bin/env python3
"""
🚀 СУПЕР БОТ V4.0 PRO - ENHANCED MULTI-TIMEFRAME STRATEGY
✅ 5 таймфреймов: 15m + 30m + 45m + 1h + 4h  [НОВОЕ]
✅ 6 TP уровней с ML вероятностями           [НОВОЕ]
✅ Оценка стратегии 0-20 баллов             [НОВОЕ]
✅ Проверка реалистичности сигналов         [НОВОЕ]
✅ Топ-5 индикаторов для деривативов
✅ AI+ML адаптация + Disco57 обучение
✅ Полная интеграция всех систем
"""

# Импорты для AI+ML системы
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Отключаем CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем TensorFlow логи
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем оптимизации

# Импорты новых модулей V4.0
try:
    from probability_calculator import ProbabilityCalculator, TPProbability
    from strategy_evaluator import StrategyEvaluator, StrategyScore
    from realism_validator import RealismValidator, RealismCheck
    V4_MODULES_AVAILABLE = True
except ImportError as e:
    V4_MODULES_AVAILABLE = False
    print(f"⚠️ V4.0 модули недоступны: {e}")

try:
    from ai_ml_system import TradingMLSystem, MLPrediction
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ AI+ML система недоступна. Установите зависимости: pip install scikit-learn tensorflow")

try:
    from smart_coin_selector import SmartCoinSelector
    SMART_SELECTOR_AVAILABLE = True
except ImportError:
    SMART_SELECTOR_AVAILABLE = False
    print("⚠️ Умный селектор недоступен")

# Импорт оптимизатора API и интеллектуальных агентов
try:
    from api_optimizer import APIOptimizer
    API_OPTIMIZER_AVAILABLE = True
except ImportError:
    API_OPTIMIZER_AVAILABLE = False
    print("⚠️ API Optimizer недоступен")

try:
    from integrate_intelligent_agents import IntegratedAgentsManager
    INTELLIGENT_AGENTS_AVAILABLE = True
except ImportError as e:
    INTELLIGENT_AGENTS_AVAILABLE = False
    print(f"⚠️ Интеллектуальные агенты недоступны: {e}")
except Exception as e:
    INTELLIGENT_AGENTS_AVAILABLE = False
    print(f"⚠️ Ошибка при импорте интеллектуальных агентов: {type(e).__name__}: {e}")

try:
    from adaptive_parameters import AdaptiveParameterSystem
    ADAPTIVE_PARAMS_AVAILABLE = True
except ImportError:
    ADAPTIVE_PARAMS_AVAILABLE = False
    print("⚠️ Адаптивные параметры недоступны")

try:
    from adaptive_trading_system import FullyAdaptiveSystem
    FULLY_ADAPTIVE_AVAILABLE = True
except ImportError:
    FULLY_ADAPTIVE_AVAILABLE = False
    print("⚠️ Полностью адаптивная система недоступна")

try:
    from data_storage_system import DataStorageSystem, MarketData, TradeDecision
    from universal_learning_system import UniversalLearningSystem
    ADVANCED_LEARNING_AVAILABLE = True
except ImportError:
    ADVANCED_LEARNING_AVAILABLE = False
    print("⚠️ Продвинутые системы обучения недоступны")

# Импорт новых модулей для расширенной функциональности
try:
    from advanced_indicators import AdvancedIndicators, IchimokuCloud, FibonacciLevels, SupportResistance
    ADVANCED_INDICATORS_AVAILABLE = True
except ImportError:
    ADVANCED_INDICATORS_AVAILABLE = False
    print("⚠️ Advanced Indicators недоступны")

try:
    from llm_monitor import (BotHealthMonitor, MLPerformancePredictor, AnomalyDetector, 
                             SmartAlertSystem, LLMAnalyzer)
    LLM_MONITOR_AVAILABLE = True
except ImportError:
    LLM_MONITOR_AVAILABLE = False
    print("⚠️ LLM Monitor недоступен")

# Импорт Advanced ML System с LSTM моделями
try:
    from advanced_ml_system import AdvancedMLSystem
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    ADVANCED_ML_AVAILABLE = False
    print(f"⚠️ Advanced ML System (LSTM) недоступен: {e}")

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

import ccxt.async_support as ccxt
from telegram import Bot
from telegram.ext import Application
from apscheduler.schedulers.asyncio import AsyncIOScheduler
# os уже импортирован на строке 14, не импортируем снова
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения (пробуем несколько мест)
env_files = [
    Path(__file__).parent / "api.env",  # api.env в директории бота
    Path(__file__).parent / ".env",      # .env в директории бота
    Path(__file__).parent.parent / ".env"  # .env в родительской директории (Downloads)
]

loaded = False
for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file, override=False)
        if not loaded:  # Логируем только первый найденный
            print(f"✅ Переменные окружения загружены из {env_file}")
        loaded = True

if not loaded:
    # Последняя попытка - стандартный load_dotenv()
    load_dotenv()
    if os.getenv('BYBIT_API_KEY'):
        print("✅ Переменные окружения загружены из системного .env")
        loaded = True

if not loaded or not os.getenv('BYBIT_API_KEY'):
    print(f"⚠️ API ключи не найдены. Проверьте файлы: {', '.join([str(f) for f in env_files])}")

# Импорт pytz для Warsaw timezone
import pytz

# Настройка часового пояса Варшавы (используется везде для времени)
WARSAW_TZ = pytz.timezone('Europe/Warsaw')

# Настройка логирования с Warsaw timezone
class WarsawFormatter(logging.Formatter):
    """Formatter для логирования с Warsaw timezone"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=WARSAW_TZ)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            # Добавляем часовой пояс для ясности (CET или CEST)
            tz_abbr = dt.strftime('%Z') if dt.strftime('%Z') else 'CET'
            s = dt.strftime(f'%Y-%m-%d %H:%M:%S {tz_abbr}')
        return s

# Настройка путей для логов
log_dir = Path(__file__).parent / "logs" / "system"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = str(log_dir / "bot.log")

log_level_name = os.getenv('BOT_LOG_LEVEL', 'DEBUG')
log_level = getattr(logging, log_level_name.upper(), logging.DEBUG)
logging.basicConfig(
    level=log_level,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
# Применяем Warsaw formatter ко всем handler'ам
for handler in logging.root.handlers:
    handler.setFormatter(WarsawFormatter("[%(asctime)s][%(levelname)s] %(message)s"))

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTakeProfitLevel:
    """Расширенный уровень Take Profit V4.0"""
    level: int
    price: float
    percent: float
    probability: float
    confidence_interval: Tuple[float, float]
    pnl_percent: float
    close_percent: float
    market_condition_factor: float


@dataclass
class EnhancedSignal:
    """Расширенный торговый сигнал V4.0"""
    symbol: str
    direction: str  # 'buy' or 'sell'
    entry_price: float
    confidence: float
    strategy_score: float  # 0-20
    timeframe_analysis: Dict  # 15m, 30m, 45m, 1h, 4h
    tp_levels: List[EnhancedTakeProfitLevel]
    stop_loss: float
    realism_check: RealismCheck
    ml_probability: float
    market_condition: str
    reasons: List[str]


class ManipulationDetector:
    """🎭 Детектор рыночных манипуляций (Pump & Dump, Fakeout)"""
    
    @staticmethod
    def detect_manipulation(df: pd.DataFrame, current_values: dict) -> Optional[Dict[str, Any]]:
        """
        Определяет манипуляцию и возвращает торговый сигнал
        
        Типы манипуляций:
        1. **PUMP** - резкий рост на низком объёме (лови откат)
        2. **DUMP** - резкое падение на низком объёме (лови отскок)
        3. **FAKEOUT** - пробой уровня с возвратом (лови разворот)
        """
        try:
            rsi = current_values['rsi']
            bb_position = (current_values['price'] - current_values['bb_lower']) / (current_values['bb_upper'] - current_values['bb_lower']) * 100
            volume_ratio = current_values['volume_ratio']
            momentum = current_values['momentum']
            
            # 🎭 МАНИПУЛЯЦИЯ #1: PUMP (RSI>85, объём низкий, рост >2%)
            if rsi > 85 and volume_ratio < 1.0 and momentum > 2.0:
                return {
                    'type': 'PUMP',
                    'signal': 'sell',  # Шортим откат!
                    'confidence': 70,
                    'reason': f'PUMP детект: RSI={rsi:.0f}, Vol={volume_ratio:.1f}x, +{momentum:.1f}%',
                    'tp_multiplier': 0.7,  # Короткие TP для быстрого выхода
                }
            
            # 🎭 МАНИПУЛЯЦИЯ #2: DUMP (RSI<15, объём низкий, падение >2%)
            elif rsi < 15 and volume_ratio < 1.0 and momentum < -2.0:
                return {
                    'type': 'DUMP',
                    'signal': 'buy',  # Покупаем отскок!
                    'confidence': 70,
                    'reason': f'DUMP детект: RSI={rsi:.0f}, Vol={volume_ratio:.1f}x, {momentum:.1f}%',
                    'tp_multiplier': 0.7,  # Короткие TP для быстрого выхода
                }
            
            # 🎭 МАНИПУЛЯЦИЯ #3: FAKEOUT (пробой BB с низким объёмом)
            elif (bb_position > 95 or bb_position < 5) and volume_ratio < 0.8:
                signal_type = 'sell' if bb_position > 95 else 'buy'
                return {
                    'type': 'FAKEOUT',
                    'signal': signal_type,
                    'confidence': 65,
                    'reason': f'FAKEOUT детект: BB={bb_position:.0f}%, Vol={volume_ratio:.1f}x',
                    'tp_multiplier': 0.8,  # Средние TP
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка детекции манипуляций: {e}")
            return None


class SuperBotV4MTF:
    """🚀 Супер Бот V4.0 с расширенными возможностями"""
    
    def __init__(self):
        # API ключи (поддерживаем оба варианта имен для совместимости)
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        # Telegram токен может быть под разными именами
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Инициализация V4.0 модулей
        if V4_MODULES_AVAILABLE:
            self.probability_calculator = ProbabilityCalculator()
            self.strategy_evaluator = StrategyEvaluator()
            self.realism_validator = RealismValidator()
            logger.info("✅ V4.0 модули инициализированы")
        else:
            self.probability_calculator = None
            self.strategy_evaluator = None
            self.realism_validator = None
            logger.warning("⚠️ V4.0 модули недоступны")
        
        # Инициализация продвинутых систем обучения
        if ADVANCED_LEARNING_AVAILABLE:
            self.data_storage = DataStorageSystem()
            self.universal_learning = UniversalLearningSystem(self.data_storage)
            logger.info("🧠 Продвинутые системы обучения инициализированы")
        else:
            self.data_storage = None
            self.universal_learning = None
            logger.warning("⚠️ Продвинутые системы обучения недоступны")
        
        # Инициализация AI+ML систем
        if ML_AVAILABLE:
            self.ml_system = TradingMLSystem()
            logger.info("✅ AI+ML система инициализирована")
        else:
            self.ml_system = None
        
        # Инициализация Advanced ML System с LSTM моделями
        if ADVANCED_ML_AVAILABLE:
            self.advanced_ml_system = AdvancedMLSystem()
            # Загружаем ранее обученные модели если есть
            try:
                # os уже импортирован глобально
                bot_dir = "/opt/bot" if os.path.exists("/opt/bot") else os.path.dirname(os.path.abspath(__file__))
                models_dir = os.path.join(bot_dir, "data", "models")
                if os.path.exists(models_dir):
                    # Загружаем модели для популярных символов
                    priority_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
                    for symbol in priority_symbols:
                        model_path = os.path.join(models_dir, f"{symbol}_lstm_model.pkl")
                        if os.path.exists(model_path):
                            try:
                                self.advanced_ml_system.load_models(model_path)
                                logger.info(f"📂 Загружена LSTM модель для {symbol}")
                            except Exception as e:
                                logger.debug(f"⚠️ Ошибка загрузки модели {symbol}: {e}")
            except Exception as e:
                logger.debug(f"⚠️ Ошибка при загрузке моделей: {e}")
            
            logger.info("🧠 Advanced ML System (LSTM + самообучение) инициализирован")
        else:
            self.advanced_ml_system = None
            logger.warning("⚠️ Advanced ML System (LSTM) недоступен")
            
        if SMART_SELECTOR_AVAILABLE:
            self.smart_selector = SmartCoinSelector()
            logger.info("✅ Умный селектор инициализирован")
        else:
            self.smart_selector = None
            
        if ADAPTIVE_PARAMS_AVAILABLE:
            self.adaptive_params_system = AdaptiveParameterSystem()
            logger.info("✅ Адаптивные параметры инициализированы")
        else:
            self.adaptive_params_system = None
            
        if FULLY_ADAPTIVE_AVAILABLE:
            self.fully_adaptive_system = FullyAdaptiveSystem()
            logger.info("✅ Полностью адаптивная система инициализирована")
        else:
            self.fully_adaptive_system = None
        
        # Инициализация Advanced Indicators
        if ADVANCED_INDICATORS_AVAILABLE:
            self.advanced_indicators = AdvancedIndicators()
            logger.info("🎯 Advanced Indicators (Ichimoku, Fibonacci, S/R) инициализированы")
        else:
            self.advanced_indicators = None
            logger.warning("⚠️ Advanced Indicators недоступны")
        
        # Инициализация LLM Monitor
        if LLM_MONITOR_AVAILABLE:
            self.health_monitor = BotHealthMonitor()
            self.ml_predictor = MLPerformancePredictor()
            self.anomaly_detector = AnomalyDetector()
            self.alert_system = SmartAlertSystem(self.health_monitor)
            self.llm_analyzer = LLMAnalyzer()
            logger.info("🤖 ML/LLM Monitoring System инициализирована")
        else:
            self.health_monitor = None
            self.ml_predictor = None
            self.anomaly_detector = None
            self.alert_system = None
            self.llm_analyzer = None
            logger.warning("⚠️ LLM Monitor недоступен")
        
        # Торговые параметры (БАЗОВЫЕ, могут адаптироваться под события)
        self.POSITION_SIZE_BASE = 5.0  # $5 базовая позиция
        self.POSITION_SIZE = 5.0  # Текущая позиция (может меняться)
        self.LEVERAGE_BASE = 5  # 5x плечо базовое
        self.LEVERAGE = 5  # Текущее плечо (может адаптироваться)
        self.MAX_STOP_LOSS_USD = 1.0  # Максимальный убыток $1.0 на сделку (фиксированный)
        self.POSITION_NOTIONAL = 25.0  # $25 позиция (5 * 5x)
        self.STOP_LOSS_PERCENT = (self.MAX_STOP_LOSS_USD / self.POSITION_NOTIONAL) * 100  # ~20% от позиции
        self.MAX_POSITIONS = 3
        self.MIN_VOLUME_24H = 1000000  # Минимальный объем 24h
        # Минимальный баланс для торговли: нужно достаточно для одной позиции ($5) + резерв
        self.MIN_BALANCE_FOR_TRADING = 5.0  # Минимум $5 для одной позиции
        self.MIN_BALANCE_FOR_MAX_POSITIONS = 15.0  # Минимум $15 для 3 позиций (3 * $5)
        
        # Инициализация менеджера событий ФРС
        try:
            from fed_event_manager import FedEventManager
            self.fed_event_manager = FedEventManager()
            
            # Автоматически добавляем сегодняшнее событие ФРС (если есть)
            # Можно добавить вручную через: bot.fed_event_manager.add_fed_event(...)
            
            logger.info("📅 Fed Event Manager инициализирован")
        except ImportError:
            self.fed_event_manager = None
            logger.warning("⚠️ Fed Event Manager недоступен")
        
        # Флаг паузы торговли (управляется через Telegram команды)
        self._trading_paused = False
        
        # Обработчик команд Telegram (инициализируется в initialize)
        self.application = None
        self.commands_handler = None
        
        # Адаптивные параметры (могут изменяться AI+ML)
        # 🎯 АДАПТИВНЫЙ MIN_CONFIDENCE под реальный рынок и направление сделки:
        # Базовая уверенность: 55-60% (для прибыльной торговли)
        # Адаптация:
        #   - BEARISH + SHORT: 55% (более агрессивно)
        #   - BULLISH + LONG: 55% (более агрессивно)
        #   - BEARISH + LONG: 60% (осторожнее)
        #   - BULLISH + SHORT: 60% (осторожнее)
        #   - NEUTRAL: 58% (средний)
        # Бонусы могут повысить до 75-80%:
        #   - Advanced Indicators (+5-12%)
        #   - ML/AI бонусы (+2-5%)
        #   - Strategy Evaluator ≥10 баллов (обязательно)
        self.MIN_CONFIDENCE_BASE = 70  # ✅ Базовая уверенность для точной и проанализированной торговли
        self.MIN_CONFIDENCE = 70  # Начальное значение, будет адаптироваться
        
        # 🚫 ИСКЛЮЧЕННЫЕ СИМВОЛЫ (только самые рискованные мемкоины)
        # Ликвидные мемкоины (DOGE, SHIB, PEPE, FLOKI) теперь РАЗРЕШЕНЫ через SmartCoinSelector
        # Исключаем только малоизвестные/рискованные мемкоины
        self.EXCLUDED_SYMBOLS = [
            # Исключаем только малоизвестные мемкоины, популярные включены
            'BONKUSDT', 'WIFUSDT', 'BOMEUSDT', 'MEMEUSDT', 
            'CATUSDT', 'DOGWIFHATUSDT'  # Только низколиквидные/рискованные
        ]
        
        # V4.0: Расширенные TP уровни (6 уровней) - МИНИМУМ +$1 (+4%) на сделку
        # Размер сделки: $5 x5 плечо = $25 позиция
        # Цель: гарантировать +$1 прибыль минимум
        # TP1 +4% закрывает 40% позиции = $25 * 0.40 * 0.04 = $0.40
        # TP1 + TP2 = (40% + 20%) = 60% позиции, средняя +5% = $25 * 0.60 * 0.05 = $0.75
        # TP1 + TP2 + TP3 = (40% + 20% + 20%) = 80% позиции, средняя +6% = $25 * 0.80 * 0.06 = $1.20 ✅
        # Итого: минимум +$1 гарантируется при закрытии первых 3 TP
        self.TP_LEVELS_V4 = [
            {'level': 1, 'percent': 4, 'portion': 0.40},   # +4%, 40% позиции - быстрый выход с минимальной прибылью
            {'level': 2, 'percent': 6, 'portion': 0.20},   # +6%, 20% позиции - дополнительная прибыль
            {'level': 3, 'percent': 8, 'portion': 0.20},   # +8%, 20% позиции - увеличенная прибыль
            {'level': 4, 'percent': 10, 'portion': 0.10},  # +10%, 10% позиции - максимальная прибыль
            {'level': 5, 'percent': 12, 'portion': 0.05},  # +12%, 5% позиции - премиум прибыль
            {'level': 6, 'percent': 15, 'portion': 0.05}   # +15%, 5% позиции - топ прибыль
        ]
        
        # Инициализация бирж и бота
        self.exchange = None
        self.api_optimizer = None  # Оптимизатор API запросов
        self.telegram_bot = None
        self.scheduler = AsyncIOScheduler()
        self.agents_manager = None  # Менеджер интеллектуальных агентов
        
        # Состояние
        self.active_positions = {}
        self.last_signals = {}
        
        # Неудачные попытки открытия позиций (для предотвращения повторных попыток)
        self.failed_open_attempts = {}  # {symbol: timestamp}
        
        # Флаг отправки стартового сообщения (чтобы не отправлять несколько раз)
        self.startup_message_sent = False
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0
        }
        
        logger.info("🚀 SuperBotV4MTF инициализирован")
    
    async def get_top_symbols_v4(self, top_n: int = 150) -> List[str]:
        """V4.0: Получить топ символы по объему с улучшенной фильтрацией"""
        try:
            logger.info(f"🔍 V4.0: Получаем топ-{top_n} символов по объему...")
            
            # Для Bybit используем правильные параметры (с оптимизацией через кэш)
            try:
                # Используем оптимизатор если доступен
                if self.api_optimizer:
                    # Для fetch_tickers используем прямые запросы, но с rate limiting
                    await self.api_optimizer.rate_limiter.acquire()
                    tickers = await self.exchange.fetch_tickers(params={'category': 'linear'})
                    self.api_optimizer.rate_limiter.on_success()
                else:
                    tickers = await self.exchange.fetch_tickers(params={'category': 'linear'})
            except Exception as e:
                logger.debug(f"⚠️ fetch_tickers с category не сработал: {e}")
                if self.api_optimizer:
                    self.api_optimizer.rate_limiter.on_rate_limit_error()
                # НЕ пробуем без параметров - ТОЛЬКО ФЬЮЧЕРСЫ!
                logger.error(f"❌ Ошибка получения тикеров фьючерсов: {e}")
                raise Exception(f"Не удалось получить фьючерсы (linear): {e}")
            
            # Фильтруем USDT пары с минимальным объемом (обновлено для манипуляций)
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if ':USDT' in symbol and ticker.get('quoteVolume', 0) > self.MIN_VOLUME_24H:
                    # Дополнительные фильтры
                    price = ticker.get('last', 0)
                    change_24h = ticker.get('percentage', 0)
                    
                    # Проверка цены (исключение для BTC/ETH)
                    symbol_upper = symbol.upper()
                    if 'BTC' in symbol_upper or 'ETH' in symbol_upper:
                        # BTC/ETH могут быть выше $100K
                        if price < 0.001:
                            continue
                    else:
                        # Остальные: расширенный диапазон до $500K
                        if price < 0.001 or price > 500000:
                            continue
                    
                    # Расширенный диапазон изменения для поиска манипуляций
                    # Обычные монеты: до -50% и до +200%
                    # Ликвидные мемкоины обрабатываются в SmartCoinSelector
                    if abs(change_24h) > 200:  # Слишком экстремально (может быть ошибка данных)
                        continue
                    
                    usdt_pairs.append((symbol, ticker))
            
            # Сортируем по объему
            sorted_pairs = sorted(usdt_pairs, key=lambda x: x[1]['quoteVolume'], reverse=True)
            
            # Нормализуем символы (предотвращаем дублирование USDT, НО сохраняем префиксы типа 1000FLOKIUSDT)
            def normalize_symbol(sym: str) -> str:
                """Нормализует символ, убирая дублирование USDT, но сохраняя оригинальный формат фьючерсов"""
                norm = sym.upper().replace('/', '').replace('-', '')
                # Убираем :USDT если есть (для формата BTC/USDT:USDT)
                if norm.endswith(':USDT'):
                    norm = norm[:-5] + 'USDT'
                elif ':USDT' in norm:
                    norm = norm.replace(':USDT', '') + 'USDT'
                # Убираем только разделители ':', но сохраняем весь символ как есть
                # НЕ меняем префиксы типа 1000FLOKIUSDT, 100SHIBUSDT и т.д.
                norm = norm.replace(':', '')
                # Убеждаемся, что заканчивается на USDT
                if not norm.endswith('USDT'):
                    norm = norm + 'USDT'
                # Убираем возможное дублирование USDT в конце
                while norm.endswith('USDTUSDT'):
                    norm = norm[:-4]
                return norm
            
            selected_symbols = [normalize_symbol(pair[0]) for pair in sorted_pairs[:top_n]]
            
            logger.info(f"✅ V4.0: Отобрано {len(selected_symbols)} символов из {len(tickers)} доступных")
            
            return selected_symbols
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения символов V4.0: {e}")
            # Fallback список
            return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 
                   'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT']
    
    async def analyze_market_trend_v4(self) -> Dict[str, Any]:
        """V4.0: Анализ общего тренда рынка"""
        try:
            logger.info("📊 V4.0: Анализируем общий тренд рынка...")
            
            # Получаем данные по Bitcoin (главный индикатор)
            # Используем оптимизатор для fetch_ticker
            if self.api_optimizer:
                btc_ticker = await self.api_optimizer.fetch_with_cache(
                    'fetch_ticker', 'BTCUSDT', cache_ttl=60
                )
            else:
                btc_ticker = await self.exchange.fetch_ticker('BTCUSDT')
            
            if not btc_ticker:
                btc_ticker = {}
            
            btc_change = btc_ticker.get('percentage', 0)
            btc_price = btc_ticker.get('last', 0)
            
            # Получаем топ-50 монет для анализа тренда (увеличено для более точного определения)
            top_symbols = await self.get_top_symbols_v4(50)
            
            rising = 0
            falling = 0
            neutral = 0
            total_change = 0
            analyzed_count = 0
            
            for symbol in top_symbols[:50]:
                try:
                    # Используем оптимизатор для fetch_ticker
                    if self.api_optimizer:
                        ticker = await self.api_optimizer.fetch_with_cache(
                            'fetch_ticker', symbol, cache_ttl=60
                        )
                    else:
                        ticker = await self.exchange.fetch_ticker(symbol)
                    
                    if not ticker:
                        continue
                    
                    change_24h = ticker.get('percentage', 0)
                    
                    total_change += change_24h
                    analyzed_count += 1
                    
                    if change_24h > 2:
                        rising += 1
                    elif change_24h < -2:
                        falling += 1
                    else:
                        neutral += 1
                        
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка получения данных {symbol}: {e}")
                    continue
            
            # Рассчитываем общий тренд
            avg_change = total_change / analyzed_count if analyzed_count > 0 else 0
            
            # Определяем тренд
            if rising > falling * 1.5 and avg_change > 1:
                trend = 'bullish'
            elif falling > rising * 1.5 and avg_change < -1:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Рассчитываем score рынка
            market_score = (rising - falling) * 10 + avg_change * 2
            
            market_data = {
                'trend': trend,
                'btc_change': btc_change,
                'btc_price': btc_price,
                'market_score': market_score,
                'rising_count': rising,
                'falling_count': falling,
                'neutral_count': neutral,
                'total_analyzed': analyzed_count,
                'avg_change': avg_change,
                'timestamp': datetime.now(WARSAW_TZ).isoformat()
            }
            
            logger.info(f"📊 V4.0: Рынок {trend.upper()} | "
                       f"BTC: {btc_change:+.1f}% | "
                       f"Растет: {rising} | Падает: {falling} | "
                       f"Score: {market_score:.1f}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа рынка V4.0: {e}")
            return {
                'trend': 'neutral',
                'btc_change': 0,
                'btc_price': 0,
                'market_score': 0,
                'rising_count': 0,
                'falling_count': 0,
                'neutral_count': 0,
                'total_analyzed': 0,
                'avg_change': 0,
                'timestamp': datetime.now(WARSAW_TZ).isoformat()
            }
    
    async def smart_symbol_selection_v4(self, market_data: Dict) -> List[str]:
        """V4.0: Умный выбор символов на основе рыночных условий"""
        try:
            market_condition = market_data.get('trend', 'neutral')
            btc_change = market_data.get('btc_change', 0)
            
            logger.info(f"🎯 V4.0: Умный выбор символов для рынка {market_condition.upper()}")
            
            if self.smart_selector:
                # Используем умный селектор если доступен
                try:
                    # Определяем условие рынка для умного селектора
                    condition_for_selector = market_condition.lower()
                    if condition_for_selector == 'neutral':
                        condition_for_selector = 'normal'
                    
                    symbols = await self.smart_selector.get_smart_symbols(self.exchange, condition_for_selector)
                    if symbols and len(symbols) > 10:
                        logger.info(f"✅ Умный селектор выбрал {len(symbols)} символов")
                        return symbols
                    else:
                        logger.warning(f"⚠️ Умный селектор вернул мало символов ({len(symbols) if symbols else 0}), пробуем fallback")
                except Exception as e:
                    logger.error(f"❌ Ошибка умного селектора: {e}", exc_info=True)
            
            # Fallback: адаптивный выбор на основе рыночных условий (100-200 монет)
            # Используем умный селектор для получения 150-200 монет
            base_symbols = await self.smart_selector.get_smart_symbols(self.exchange, condition_for_selector)
            
            # Убеждаемся, что минимум 150 монет, choices до 200
            if len(base_symbols) < 150:
                # Дополняем через get_top_symbols_v4
                additional_symbols = await self.get_top_symbols_v4(200)
                # Убираем дубликаты
                all_symbols = list(set(base_symbols + additional_symbols))
                # Сортируем по приоритету (сначала из умного селектора)
                base_symbols = base_symbols + [s for s in all_symbols if s not in base_symbols]
                base_symbols = base_symbols[:200]  # Максимум 200
            
            # Адаптируем количество символов под рыночные условия (150-200 монет минимум)
            if market_condition == 'bullish':
                # В бычьем рынке анализируем больше символов
                selected_count = min(200, len(base_symbols))
                logger.info(f"🐂 Бычий рынок: анализируем {selected_count} символов (умный селектор)")
            elif market_condition == 'bearish':
                # В медвежьем рынке все еще анализируем много монет (150+)
                selected_count = min(150, len(base_symbols))
                logger.info(f"🐻 Медвежий рынок: анализируем {selected_count} топ символов (умный селектор)")
            elif market_condition == 'volatile':
                # В волатильном рынке среднее количество
                selected_count = min(175, len(base_symbols))
                logger.info(f"🌊 Волатильный рынок: анализируем {selected_count} символов (умный селектор)")
            else:
                # Нейтральный рынок - минимум 150 монет
                selected_count = min(150, len(base_symbols))
                logger.info(f"⚪ Нейтральный рынок: анализируем {selected_count} символов (умный селектор)")
            
            selected_symbols = base_symbols[:selected_count]
            
            # Добавляем приоритетные символы если их нет (включая популярные мемкоины)
            # ВАЖНО: Используем правильные форматы фьючерсов с биржи (1000FLOKIUSDT, а не FLOKIUSDT)
            priority_symbols = [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT',
                'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', '1000FLOKIUSDT'  # Правильный формат фьючерса
            ]
            for symbol in priority_symbols:
                if symbol not in selected_symbols:
                    selected_symbols.insert(0, symbol)
            
            return selected_symbols
            
        except Exception as e:
            logger.error(f"❌ Ошибка умного выбора символов V4.0: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
    
    async def initialize(self):
        """Инициализация соединений"""
        try:
            # Инициализация биржи
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'linear',
                    'accountType': 'UNIFIED'  # Unified account для Bybit
                }
            })
            
            # Инициализация API оптимизатора
            if API_OPTIMIZER_AVAILABLE:
                # os уже импортирован глобально
                bot_dir = "/opt/bot" if os.path.exists("/opt/bot") else os.path.dirname(os.path.abspath(__file__))
                cache_dir = os.path.join(bot_dir, "data", "cache")
                self.api_optimizer = APIOptimizer(self.exchange, cache_dir=cache_dir)
                logger.info("⚡ API Optimizer инициализирован (кэш + rate limiting)")
            else:
                self.api_optimizer = None
                logger.warning("⚠️ API Optimizer недоступен, используются прямые запросы")
            
            # Инициализация интеллектуальных агентов
            if INTELLIGENT_AGENTS_AVAILABLE:
                # os уже импортирован глобально
                bot_dir = "/opt/bot" if os.path.exists("/opt/bot") else os.path.dirname(os.path.abspath(__file__))
                bot_pid = os.getpid()
                self.agents_manager = IntegratedAgentsManager(bot_dir=bot_dir, bot_pid=bot_pid)
                logger.info("🤖 Интеллектуальные агенты инициализированы (самообучение + обмен знаниями)")
            else:
                self.agents_manager = None
                logger.warning("⚠️ Интеллектуальные агенты недоступны")
            
            # Инициализация Telegram с командами
            if self.telegram_token:
                from telegram import Bot
                from telegram.ext import Application
                from telegram_commands_handler import TelegramCommandsHandler
                
                self.telegram_bot = Bot(token=self.telegram_token)
                
                # Создаем Application для обработки команд
                self.application = Application.builder().token(self.telegram_token).build()
                
                # Регистрируем команды
                self.commands_handler = TelegramCommandsHandler(self)
                await self.commands_handler.register_commands(self.application)
                
                logger.info("✅ Telegram бот инициализирован с командами")
            else:
                self.application = None
                self.commands_handler = None
            
            logger.info("✅ Все соединения инициализированы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            raise
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Получить OHLCV данные с повторами; для 45м есть фолбэк из 15м (3x)."""
        # Нормализация символа
        def normalize_symbol(sym: str) -> str:
            if not sym:
                return sym
            norm = sym.upper().replace('/', '').replace('-', '')
            if norm.endswith(':USDT'):
                norm = norm[:-5] + 'USDT'
            elif ':USDT' in norm:
                norm = norm.replace(':USDT', '') + 'USDT'
            norm = norm.replace(':', '')
            if not norm.endswith('USDT'):
                norm = norm + 'USDT'
            while norm.endswith('USDTUSDT'):
                norm = norm[:-4]
            return norm

        normalized_symbol = normalize_symbol(symbol)

        attempts = 3 if timeframe == '45m' else 1
        last_err = None
        for _ in range(attempts):
            try:
                if self.api_optimizer:
                    ohlcv = await self.api_optimizer.fetch_with_cache(
                        'fetch_ohlcv', normalized_symbol, timeframe, limit, cache_ttl=30
                    )
                    if ohlcv is None:
                        ohlcv = await self.exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
                else:
                    ohlcv = await self.exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
                # Bybit v5 иногда возвращает пустые свечи для '45m'; добавляем безопасный фолбэк через pybit interval=45
                if (not ohlcv or len(ohlcv) == 0) and timeframe == '45m':
                    try:
                        from pybit.unified_trading import HTTP
                        session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False, recv_window=5000, timeout=10)
                        k = session.get_kline(category='linear', symbol=normalized_symbol, interval=45, limit=min(200, max(100, limit)))
                        lst = (k.get('result', {}) or {}).get('list', []) or []
                        ohlcv = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in lst]
                    except Exception as _:
                        pass
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
            except Exception as e:
                last_err = e
        # Фолбэк: синтезируем 45м из 15м
        if timeframe == '45m':
            try:
                df15 = await self._fetch_ohlcv(symbol, '15m', limit=limit * 3)
                if not df15.empty:
                    df15 = df15.sort_values('timestamp').reset_index(drop=True)
                    idx = np.arange(len(df15)) // 3
                    agg = df15.groupby(idx).agg({
                        'timestamp': 'last',
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).reset_index(drop=True)
                    return agg.tail(limit)
            except Exception as e:
                logger.debug(f"⚠️ Фолбэк 45м из 15м не удался: {e}")
        if last_err:
            logger.debug(f"⚠️ Ошибка получения данных {symbol} {timeframe}: {last_err}")
        return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Расчет технических индикаторов"""
        if df.empty or len(df) < 21:
            return {}
        
        try:
            import talib
            
            # Базовые данные
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # EMA (Топ-5 индикатор #1)
            ema_9 = talib.EMA(close, timeperiod=9)[-1]
            ema_21 = talib.EMA(close, timeperiod=21)[-1]
            ema_50 = talib.EMA(close, timeperiod=50)[-1]
            ema_200 = talib.EMA(close, timeperiod=200)[-1] if len(close) >= 200 else ema_50
            
            # RSI (Топ-5 индикатор #2)
            rsi = talib.RSI(close, timeperiod=14)[-1]
            
            # MACD (Топ-5 индикатор #3)
            macd, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Bollinger Bands (Топ-5 индикатор #4)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_position = ((close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100) if bb_upper[-1] != bb_lower[-1] else 50
            
            # ATR (Топ-5 индикатор #5)
            atr = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # Дополнительные индикаторы
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            williams_r = talib.WILLR(high, low, close, timeperiod=14)[-1]
            cci = talib.CCI(high, low, close, timeperiod=14)[-1]
            
            # Объемы
            volume_ma_20 = talib.SMA(volume, timeperiod=20)[-1]
            volume_ma_50 = talib.SMA(volume, timeperiod=50)[-1] if len(volume) >= 50 else volume_ma_20
            volume_ratio = volume[-1] / volume_ma_20 if volume_ma_20 > 0 else 1.0
            
            # Momentum
            momentum = ((close[-1] - close[-21]) / close[-21] * 100) if len(close) >= 21 else 0
            
            # Candle reversal (смягченное условие)
            candle_reversal = (close[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100
            
            indicators = {
                'price': close[-1],
                'ema_9': ema_9,
                'ema_21': ema_21,
                'ema_50': ema_50,
                'ema_200': ema_200,
                'rsi': rsi,
                'macd': macd[-1],
                'macd_signal': macd_signal[-1],
                'macd_histogram': macd_histogram[-1],
                'bb_upper': bb_upper[-1],
                'bb_middle': bb_middle[-1],
                'bb_lower': bb_lower[-1],
                'bb_position': bb_position,
                'atr': atr,
                'stoch_k': stoch_k[-1],
                'stoch_d': stoch_d[-1],
                'williams_r': williams_r,
                'cci': cci,
                'volume': volume[-1],
                'volume_ma_20': volume_ma_20,
                'volume_ma_50': volume_ma_50,
                'volume_ratio': volume_ratio,
                'momentum': momentum,
                'candle_reversal': candle_reversal
            }
            
            # Добавляем Advanced Indicators (Ichimoku, Fibonacci, Support/Resistance)
            if self.advanced_indicators:
                try:
                    advanced = self.advanced_indicators.get_all_indicators(df)
                    
                    # Ichimoku
                    if 'ichimoku' in advanced:
                        ichi = advanced['ichimoku']
                        indicators['ichimoku_trend'] = ichi.get('trend', 'neutral')
                        indicators['ichimoku_signal'] = ichi.get('signal', 'hold')
                        indicators['ichimoku_cloud_top'] = ichi.get('cloud_top', 0)
                        indicators['ichimoku_cloud_bottom'] = ichi.get('cloud_bottom', 0)
                    
                    # Fibonacci
                    if 'fibonacci' in advanced:
                        fib = advanced['fibonacci']
                        indicators['fib_level_382'] = fib.get('level_382', 0)
                        indicators['fib_level_500'] = fib.get('level_500', 0)
                        indicators['fib_level_618'] = fib.get('level_618', 0)
                        indicators['fib_position'] = fib.get('current_position', 50)
                    
                    # Support/Resistance
                    if 'support_resistance' in advanced:
                        sr = advanced['support_resistance']
                        indicators['nearest_support'] = sr.get('nearest_support', 0)
                        indicators['nearest_resistance'] = sr.get('nearest_resistance', 0)
                        indicators['support_distance_pct'] = sr.get('support_distance_pct', 0)
                        indicators['resistance_distance_pct'] = sr.get('resistance_distance_pct', 0)
                        indicators['sr_strength'] = sr.get('strength', 'weak')
                    
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка расчета Advanced Indicators: {e}")
            
            return indicators
            
        except ImportError as e:
            logger.warning(f"⚠️ TA-Lib не установлен: {e}. Установите: pip install TA-Lib и libta-lib0-dev")
            return {}
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета индикаторов: {e}")
            return {}
    
    async def _fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, Dict]:
        """V4.0: Получить данные по 5 таймфреймам (добавлен 45m)"""
        try:
            timeframes = ['15m', '30m', '45m', '1h', '4h']  # ✅ ДОБАВЛЕН 45m
            data = {}
            
            for tf in timeframes:
                df = await self._fetch_ohlcv(symbol, tf, 100)
                if not df.empty:
                    indicators = self._calculate_indicators(df)
                    if indicators:
                        data[tf] = indicators
            
            return data
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка получения MTF данных для {symbol}: {e}")
            return {}
    
    def _get_adaptive_signal_params(self, market_condition: str, symbol_data: Dict, 
                                    trade_direction: Optional[str] = None) -> Dict:
        """🤖 Получить адаптивные параметры для сигналов (AI+ML + адаптация под рынок и направление)"""
        try:
            # 1. РАССЧИТЫВАЕМ АДАПТИВНЫЙ MIN_CONFIDENCE ПОД РЫНОК И НАПРАВЛЕНИЕ
            # Главная задача: ПРИБЫЛЬ, значит нужно быть гибче в выгодных ситуациях
            base_confidence = self.MIN_CONFIDENCE_BASE
            
            # Адаптация под рыночные условия и направление сделки
            if trade_direction:
                market_upper = market_condition.upper()
                
                if market_upper == 'BEARISH':
                    if trade_direction.lower() == 'sell':  # SHORT в медвежьем рынке - АГРЕССИВНЕЕ
                        adaptive_min_confidence = base_confidence - 3  # 55% для SHORT в BEARISH
                        logger.debug(f"🎯 BEARISH + SHORT: снижен порог до {adaptive_min_confidence}% для агрессивной торговли")
                    else:  # LONG в медвежьем рынке - ОСТОРОЖНЕЕ
                        adaptive_min_confidence = base_confidence + 2  # 60% для LONG в BEARISH
                        logger.debug(f"🎯 BEARISH + LONG: повышен порог до {adaptive_min_confidence}% (осторожнее)")
                elif market_upper == 'BULLISH':
                    if trade_direction.lower() == 'buy':  # LONG в бычьем рынке - АГРЕССИВНЕЕ
                        adaptive_min_confidence = base_confidence - 3  # 55% для LONG в BULLISH
                        logger.debug(f"🎯 BULLISH + LONG: снижен порог до {adaptive_min_confidence}% для агрессивной торговли")
                    else:  # SHORT в бычьем рынке - ОСТОРОЖНЕЕ
                        adaptive_min_confidence = base_confidence + 2  # 60% для SHORT в BULLISH
                        logger.debug(f"🎯 BULLISH + SHORT: повышен порог до {adaptive_min_confidence}% (осторожнее)")
                else:  # NEUTRAL
                    adaptive_min_confidence = base_confidence  # 58% для NEUTRAL
            else:
                # Если направление еще не определено, используем базовое
                adaptive_min_confidence = base_confidence
            
            # Ограничиваем диапазон: минимум 65%, максимум 85%
            adaptive_min_confidence = max(65, min(85, adaptive_min_confidence))
            
            # 2. Базовые адаптивные параметры
            if hasattr(self, 'adaptive_params_system') and self.adaptive_params_system:
                adaptive_params = self.adaptive_params_system.get_adaptive_parameters(symbol_data)
                # Переопределяем min_confidence адаптивным значением
                adaptive_params.min_confidence = adaptive_min_confidence
            else:
                # Fallback значения
                from dataclasses import dataclass
                @dataclass
                class FallbackParams:
                    rsi_oversold: float = 35
                    rsi_overbought: float = 65
                    min_confidence: float = adaptive_min_confidence  # ✅ АДАПТИВНЫЙ под рынок
                    volume_filter: float = 0.3
                adaptive_params = FallbackParams()
            
            # 2. AI+ML предсказания
            ml_confidence_bonus = 0
            if hasattr(self, 'ml_system') and self.ml_system:
                try:
                    # Создаем фичи для ML
                    features = self.ml_system.create_features({
                        'close': [symbol_data.get('price', 0)] * 21,  # минимум для расчета
                        'volume': [symbol_data.get('volume', 0)] * 21,
                        'rsi': [symbol_data.get('rsi', 50)] * 21,
                        'macd': [symbol_data.get('macd', 0)] * 21
                    })
                    if features is not None and len(features) > 0:
                        ml_confidence_bonus = min(15, len(features) * 0.01)  # бонус за количество паттернов
                except Exception as e:
                    logger.debug(f"⚠️ ML система недоступна: {e}")
            
            # 3. Fully Adaptive динамические пороги
            dynamic_adjustment = 0
            if hasattr(self, 'fully_adaptive_system') and self.fully_adaptive_system:
                try:
                    # Адаптация на основе недавней производительности
                    recent_performance = getattr(self, 'recent_trades_performance', {'win_rate': 0.5})
                    if recent_performance.get('win_rate', 0.5) > 0.7:
                        dynamic_adjustment = -5  # ужесточаем при высокой успешности
                    elif recent_performance.get('win_rate', 0.5) < 0.4:
                        dynamic_adjustment = +5  # смягчаем при низкой успешности
                except Exception as e:
                    logger.debug(f"⚠️ Fully Adaptive система недоступна: {e}")
            
            return {
                'rsi_oversold': max(20, min(50, adaptive_params.rsi_oversold + dynamic_adjustment)),
                'rsi_overbought': max(50, min(80, adaptive_params.rsi_overbought - dynamic_adjustment)),
                'min_confidence': adaptive_params.min_confidence,  # ✅ АДАПТИВНЫЙ (65-85% в зависимости от рынка/монеты)
                'ml_confidence_bonus': ml_confidence_bonus,
                'bb_adjustment': dynamic_adjustment,
                'market_condition': market_condition,
                'trade_direction': trade_direction  # Сохраняем для логирования
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка адаптивных параметров: {e}")
            # Возвращаем безопасные значения по умолчанию (адаптивные)
            base_confidence = self.MIN_CONFIDENCE_BASE
            if trade_direction and market_condition:
                market_upper = market_condition.upper()
                if market_upper == 'BEARISH' and trade_direction.lower() == 'sell':
                    base_confidence = 55  # SHORT в BEARISH
                elif market_upper == 'BULLISH' and trade_direction.lower() == 'buy':
                    base_confidence = 55  # LONG в BULLISH
                else:
                    base_confidence = 60  # Осторожнее для противоположных направлений
            
            return {
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'min_confidence': max(65, min(85, base_confidence)),  # ✅ АДАПТИВНЫЙ
                'ml_confidence_bonus': 0,
                'bb_adjustment': 0,
                'market_condition': market_condition,
                'trade_direction': trade_direction
            }
    
    def _get_bollinger_signal(self, c_45m: Dict) -> Tuple[str, float, List[str]]:
        """V4.0: Получить сигнал Bollinger Reversion с 45m подтверждением"""
        # Рассчитываем BB позицию (0-100%)
        bb_range = c_45m['bb_upper'] - c_45m['bb_lower']
        if bb_range > 0:
            bb_position = (c_45m['price'] - c_45m['bb_lower']) / bb_range * 100
        else:
            bb_position = 50
        
        # BUY: цена в нижней зоне BB (≤25%) + RSI не перекуплен (≤65)
        if (bb_position <= 25 and c_45m['rsi'] <= 65):
            # Дополнительные бонусы
            rsi_bonus = max(0, 65 - c_45m['rsi']) * 0.5  # бонус за низкий RSI
            bb_bonus = max(0, 25 - bb_position) * 0.8     # бонус за близость к границе
            candle_bonus = 5 if c_45m.get('candle_reversal', 0) > 0 else 0
            
            confidence = 55 + rsi_bonus + bb_bonus + candle_bonus
            reasons = [
                'BUY-BB_REVERSION_V4',
                f"BB={bb_position:.0f}%",
                f"RSI={c_45m['rsi']:.0f}",
                f"45m_confirm"  # V4.0: подтверждение 45m
            ]
            if c_45m.get('candle_reversal', 0) > 0:
                reasons.append(f"Candle↗️{c_45m['candle_reversal']:.1f}%")
            return 'buy', min(90, confidence), reasons

        # SELL: цена в верхней зоне BB (≥75%) + RSI не перепродан (≥35)
        elif (bb_position >= 75 and c_45m['rsi'] >= 35):
            # Дополнительные бонусы
            rsi_bonus = max(0, c_45m['rsi'] - 35) * 0.5   # бонус за высокий RSI
            bb_bonus = max(0, bb_position - 75) * 0.8     # бонус за близость к границе
            candle_bonus = 5 if c_45m.get('candle_reversal', 0) < 0 else 0
            
            confidence = 55 + rsi_bonus + bb_bonus + candle_bonus
            reasons = [
                'SELL-BB_REVERSION_V4',
                f"BB={bb_position:.0f}%",
                f"RSI={c_45m['rsi']:.0f}",
                f"45m_confirm"  # V4.0: подтверждение 45m
            ]
            if c_45m.get('candle_reversal', 0) < 0:
                reasons.append(f"Candle↘️{c_45m['candle_reversal']:.1f}%")
            return 'sell', min(90, confidence), reasons

        return None, 0, []
    
    async def analyze_symbol_v4(self, symbol: str) -> Optional[EnhancedSignal]:
        """V4.0: Расширенный анализ символа с новыми возможностями"""
        try:
            # Получаем данные по 5 таймфреймам
            mtf_data = await self._fetch_multi_timeframe_data(symbol)
            if len(mtf_data) < 4:  # Минимум 4 из 5 таймфреймов
                return None
            
            current_15m = mtf_data.get('15m', {})
            current_30m = mtf_data.get('30m', {})
            current_45m = mtf_data.get('45m', {})  # V5.0: ОСНОВНОЙ таймфрейм для анализа
            current_1h = mtf_data.get('1h', {})
            current_4h = mtf_data.get('4h', {})
            
            # ОСНОВНОЙ АНАЛИЗ НА 45m - требует наличие данных
            if not all([current_45m, current_1h, current_4h]):
                return None
            
            # 🤖 ПОЛУЧАЕМ АДАПТИВНЫЕ ПАРАМЕТРЫ (AI+ML + 1000+ ПАТТЕРНОВ)
            market_condition = getattr(self, '_current_market_condition', 'NEUTRAL')
            
            # Предварительно определяем потенциальное направление для адаптации порога
            # (будет уточнено позже)
            potential_direction = None
            
            # Анализ глобального тренда
            global_trend_bullish = current_4h.get('ema_50', 0) > current_4h.get('ema_200', 0)
            global_trend_bearish = current_4h.get('ema_50', 0) < current_4h.get('ema_200', 0)
            
            signal = None
            confidence = 0
            reasons = []
            
            # 🟢 BUY СИГНАЛ (ТОП-5 ИНДИКАТОРОВ: EMA, RSI, MACD, BB, ATR)
            # Получаем adaptive_params заранее (без направления для начальной проверки)
            # ОСНОВНОЙ АНАЛИЗ НА 45m
            temp_adaptive_params = self._get_adaptive_signal_params(market_condition, current_45m, None)
            
            buy_normal = {
                # EMA ТРЕНД (9, 21, 50)
                'global_trend_ok': global_trend_bullish or abs(current_1h.get('ema_50', 0) - current_1h.get('ema_200', 0)) < current_1h.get('ema_200', 1) * 0.01,
                '4h_trend_up': current_4h['ema_9'] > current_4h['ema_21'],
                '1h_trend_up': current_1h['ema_9'] > current_1h['ema_21'],
                '45m_trend_up': current_45m['ema_9'] > current_45m['ema_21'],  # V5.0: 45m - ОСНОВНОЙ
                '30m_trend_up': current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0),  # Подтверждение
                '45m_price_above': current_45m['price'] > current_45m['ema_9'],  # V5.0: проверка цены на 45m
                
                # RSI ЗОНЫ (🤖 АДАПТИВНЫЕ ПОРОГИ AI+ML) - НА 45m
                '45m_rsi': current_45m['rsi'] <= temp_adaptive_params['rsi_overbought'],
                '45m_rsi_not_extreme': current_45m['rsi'] >= 20,
                
                # MACD ПОДТВЕРЖДЕНИЕ - НА 45m
                'macd_bullish': current_45m.get('macd', 0) > current_45m.get('macd_signal', 0),
                
                # BOLLINGER BANDS (с запасом ±15%) - НА 45m
                'bb_position': current_45m.get('bb_position', 50) <= 75,
                
                # ATR + MOMENTUM - НА 45m
                '45m_momentum': current_45m['momentum'] > 0.1,
            }
            
            # 🔴 SELL СИГНАЛ (ТОП-5 ИНДИКАТОРОВ: EMA, RSI, MACD, BB, ATR)
            sell_conditions = {
                # EMA ТРЕНД (нисходящий)
                'global_trend_ok': global_trend_bearish or abs(current_1h.get('ema_50', 0) - current_1h.get('ema_200', 0)) < current_1h.get('ema_200', 1) * 0.01,
                '4h_trend_down': current_4h['ema_9'] < current_4h['ema_21'],
                '1h_trend_down': current_1h['ema_9'] < current_1h['ema_21'],
                '45m_trend_down': current_45m['ema_9'] < current_45m['ema_21'],  # V5.0: 45m - ОСНОВНОЙ
                '30m_trend_down': current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0),  # Подтверждение
                '45m_price_below': current_45m['price'] < current_45m['ema_9'],  # V5.0: проверка цены на 45m
                
                # RSI ЗОНЫ (🤖 АДАПТИВНЫЕ ПОРОГИ AI+ML) - НА 45m
                '45m_rsi': current_45m['rsi'] >= temp_adaptive_params['rsi_oversold'],
                '45m_rsi_not_extreme': current_45m['rsi'] <= 80,
                
                # MACD ПОДТВЕРЖДЕНИЕ - НА 45m
                'macd_bearish': current_45m.get('macd', 0) < current_45m.get('macd_signal', 0),
                
                # BOLLINGER BANDS (с запасом ±15%) - НА 45m
                'bb_position': current_45m.get('bb_position', 50) >= 25,
                
                # ATR + MOMENTUM - НА 45m
                '45m_momentum': current_45m['momentum'] < -0.1,
            }
            
            # 🎯 ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ: Advanced Indicators (Ichimoku, Fibonacci, S/R)
            advanced_bonus = 0
            advanced_reasons = []
            
            if self.advanced_indicators:
                try:
                    # Получаем данные для расчета Advanced Indicators (используем 45m - ОСНОВНОЙ)
                    df_45m = await self._fetch_ohlcv(symbol, '45m', 100)
                    if not df_45m.empty and len(df_45m) >= 52:
                        advanced_data = self.advanced_indicators.get_all_indicators(df_45m)
                        
                        # Ichimoku фильтр
                        if 'ichimoku' in advanced_data:
                            ichi = advanced_data['ichimoku']
                            if ichi.get('signal') == 'buy' and ichi.get('trend') == 'bullish':
                                advanced_bonus += 5
                                advanced_reasons.append('Ichimoku🟢')
                            elif ichi.get('signal') == 'sell' and ichi.get('trend') == 'bearish':
                                advanced_bonus += 5
                                advanced_reasons.append('Ichimoku🔴')
                        
                        # Fibonacci фильтр
                        if 'fibonacci' in advanced_data:
                            fib_pos = advanced_data['fibonacci'].get('current_position', 50)
                            # На уровнях 38.2%, 50%, 61.8% - хорошие точки входа
                            if 35 <= fib_pos <= 65:
                                advanced_bonus += 3
                                advanced_reasons.append('Fib📊')
                        
                        # Support/Resistance фильтр
                        if 'support_resistance' in advanced_data:
                            sr = advanced_data['support_resistance']
                            support_dist = sr.get('support_distance_pct', 100)
                            resistance_dist = sr.get('resistance_distance_pct', 100)
                            
                            # Если цена близко к поддержке (низкий риск) - BUY
                            if support_dist < 2.0 and resistance_dist > 5.0:
                                advanced_bonus += 4
                                advanced_reasons.append('S/R🟢')
                            # Если цена близко к сопротивлению (низкий риск) - SELL
                            elif resistance_dist < 2.0 and support_dist > 5.0:
                                advanced_bonus += 4
                                advanced_reasons.append('S/R🔴')
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка Advanced Indicators для {symbol}: {e}")
            
            # Проверяем Bollinger Reversion с 45m подтверждением
            if current_45m:
                bb_signal, bb_confidence, bb_reasons = self._get_bollinger_signal(current_45m)
                if bb_signal:
                    signal = bb_signal
                    confidence = bb_confidence
                    reasons = bb_reasons
                    potential_direction = signal  # Определили направление
            
            # Если нет BB сигнала, проверяем обычные условия
            if not signal:
                buy_count = sum(buy_normal.values())
                sell_count = sum(sell_conditions.values())
                
                if buy_count >= 7:  # Минимум 7 из 10 условий
                    signal = 'buy'
                    potential_direction = 'buy'
                    confidence = 50 + (buy_count - 7) * 5
                    reasons = ['BUY-NORMAL_V4', f'Conditions:{buy_count}/10']
                elif sell_count >= 7:  # Минимум 7 из 10 условий
                    signal = 'sell'
                    potential_direction = 'sell'
                    confidence = 50 + (sell_count - 7) * 5
                    reasons = ['SELL-NORMAL_V4', f'Conditions:{sell_count}/10']
            
            # 🤖 ПОЛУЧАЕМ АДАПТИВНЫЕ ПАРАМЕТРЫ С УЧЕТОМ НАПРАВЛЕНИЯ СДЕЛКИ
            # (делаем это после определения направления для правильной адаптации)
            # ОСНОВНОЙ АНАЛИЗ НА 45m
            adaptive_params = self._get_adaptive_signal_params(market_condition, current_45m, potential_direction)
            
            logger.debug(f"🤖 {symbol}: AI+ML параметры - RSI:{adaptive_params['rsi_oversold']}-{adaptive_params['rsi_overbought']}, "
                        f"MinConf:{adaptive_params['min_confidence']}% (Рынок: {market_condition}, Направление: {potential_direction}), ML+{adaptive_params['ml_confidence_bonus']:.0f}")
            
            # 🤖 ДОБАВЛЯЕМ AI+ML БОНУС К УВЕРЕННОСТИ
            if signal and confidence > 0:
                ml_bonus = adaptive_params.get('ml_confidence_bonus', 0)
                confidence += ml_bonus
                if ml_bonus > 0:
                    reasons.append(f'🤖ML+{ml_bonus:.0f}')
                
                # 🎯 ДОБАВЛЯЕМ ADVANCED INDICATORS БОНУС
                if advanced_bonus > 0:
                    confidence += advanced_bonus
                    reasons.extend(advanced_reasons)
                    logger.debug(f"🎯 {symbol}: Advanced Indicators бонус +{advanced_bonus}")
                
                logger.debug(f"🤖 {symbol}: {signal.upper()} базовая={confidence-ml_bonus-advanced_bonus:.0f} + ML={ml_bonus:.0f} + Advanced={advanced_bonus:.0f} = {confidence:.0f}")
            
            # 🤖 ПРИМЕНЯЕМ АДАПТИВНЫЙ МИНИМАЛЬНЫЙ ПОРОГ УВЕРЕННОСТИ
            # (уже рассчитан с учетом рынка и направления сделки)
            adaptive_min_confidence = adaptive_params.get('min_confidence', self.MIN_CONFIDENCE_BASE)
            
            # 📅 БОНУС УВЕРЕННОСТИ ПЕРЕД ВАЖНЫМИ СОБЫТИЯМИ (ФРС и т.д.)
            if self.fed_event_manager:
                risk_adjustments = self.fed_event_manager.get_risk_adjustments()
                confidence_bonus = risk_adjustments.get('confidence_bonus', 0)
                if confidence_bonus > 0:
                    adaptive_min_confidence += confidence_bonus
                    logger.info(f"📅 {symbol}: MIN_CONFIDENCE повышен на +{confidence_bonus}% "
                              f"из-за важного события. Требуется: {adaptive_min_confidence:.0f}%")
            
            # Логируем адаптивный порог для отладки
            logger.debug(
                f"🎯 {symbol}: Адаптивный MIN_CONFIDENCE={adaptive_min_confidence}% | "
                f"Рынок={market_condition} | Направление={signal if signal else 'n/a'}"
            )

            # Детальный срез индикаторов по MTF для отладки
            try:
                logger.debug(
                    f"🔎 {symbol} 45m: EMA9={current_45m.get('ema_9')} EMA21={current_45m.get('ema_21')} "
                    f"RSI={current_45m.get('rsi')} MACD={current_45m.get('macd')} MACDsig={current_45m.get('macd_signal')} "
                    f"BBpos={current_45m.get('bb_position')} ATR={current_45m.get('atr')} VolRatio={current_45m.get('volume_ratio')}"
                )
                logger.debug(
                    f"🔎 {symbol} 1h:  EMA9={current_1h.get('ema_9')} EMA21={current_1h.get('ema_21')} RSI={current_1h.get('rsi')}"
                )
                logger.debug(
                    f"🔎 {symbol} 4h:  EMA9={current_4h.get('ema_9')} EMA21={current_4h.get('ema_21')} RSI={current_4h.get('rsi')}"
                )
            except Exception:
                pass
            
            # 🔒 Требуем подтверждение 45m + 1h + 4h по направлению, 15m/30m используем как тайминговые триггеры
            def _mtf_confirm(dir_: str) -> bool:
                if dir_ == 'buy':
                    c45 = current_45m.get('ema_9', 0) > current_45m.get('ema_21', 0)
                    c1h = current_1h.get('ema_9', 0) > current_1h.get('ema_21', 0)
                    c4h = current_4h.get('ema_9', 0) > current_4h.get('ema_21', 0)
                    logger.debug(f"✅ MTF {symbol} LONG check 45m={c45} 1h={c1h} 4h={c4h}")
                    return c45 and c1h and c4h
                if dir_ == 'sell':
                    c45 = current_45m.get('ema_9', 0) < current_45m.get('ema_21', 0)
                    c1h = current_1h.get('ema_9', 0) < current_1h.get('ema_21', 0)
                    c4h = current_4h.get('ema_9', 0) < current_4h.get('ema_21', 0)
                    logger.debug(f"✅ MTF {symbol} SHORT check 45m={c45} 1h={c1h} 4h={c4h}")
                    return c45 and c1h and c4h
                return False

            if signal and not _mtf_confirm(signal):
                logger.info(f"🚫 {symbol}: Отклонено из-за отсутствия подтверждения 45m+1h+4h для {signal.upper()}")
                signal = None

            # ДОП. ФИЛЬТРЫ ДЛЯ НОВЫХ СДЕЛОК: требуем реальный потенциал для +1%
            def _has_min_potential(direction: str) -> bool:
                try:
                    price = float(current_45m.get('price', 0) or 0)
                    atr = float(current_45m.get('atr', 0) or 0)
                    vol_ratio = float(current_45m.get('volume_ratio', 0) or 0)
                    if price <= 0:
                        return False
                    atr_pct = (atr / price) * 100.0
                    # Требуемая волатильность и ликвидность для покрытия fee/скольжения и достижения +1%
                    if atr_pct < 1.2:
                        logger.info(f"🚫 {symbol}: Отклонено | ATR45m={atr_pct:.2f}% < 1.2%")
                        return False
                    if vol_ratio < 1.2:
                        logger.info(f"🚫 {symbol}: Отклонено | VolumeRatio45m={vol_ratio:.2f} < 1.2")
                        return False
                    # Направление импульса на 15m/30m как тайминг-триггеры
                    ema_up_15 = current_15m.get('ema_9', 0) > current_15m.get('ema_21', 0)
                    ema_up_30 = current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0)
                    ema_dn_15 = current_15m.get('ema_9', 0) < current_15m.get('ema_21', 0)
                    ema_dn_30 = current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0)
                    if direction == 'buy' and not (ema_up_15 and ema_up_30):
                        logger.info(f"🚫 {symbol}: Отклонено | Нет импульса на 15m/30m для LONG")
                        return False
                    if direction == 'sell' and not (ema_dn_15 and ema_dn_30):
                        logger.info(f"🚫 {symbol}: Отклонено | Нет импульса на 15m/30m для SHORT")
                        return False
                    logger.debug(f"✅ {symbol}: Потенциал OK | ATR%={atr_pct:.2f} VolRatio={vol_ratio:.2f} Dir={direction}")
                    return True
                except Exception:
                    return False

            if signal and confidence >= adaptive_min_confidence and _has_min_potential(signal):
                # V5.0: Создаем расширенный сигнал (ОСНОВНОЙ АНАЛИЗ НА 45m)
                enhanced_signal = await self._create_enhanced_signal_v4(
                    symbol, signal, current_45m['price'], confidence, reasons,
                    mtf_data, market_condition
                )
                return enhanced_signal
            elif signal and confidence < adaptive_min_confidence:
                # Логируем отклонение по уверенности
                logger.info(f"🚫 {symbol}: {signal.upper()} отклонен | "
                           f"Уверенность {confidence:.0f}% < {adaptive_min_confidence:.0f}% | "
                           f"Причины: {', '.join(reasons)}")
            elif signal:
                # Если не прошли доп. фильтры
                logger.info(f"🚫 {symbol}: {signal.upper()} отклонен доп. фильтрами (волатильность/ликвидность/импульс)")
            else:
                # Логируем отсутствие сигнала (ОСНОВНОЙ АНАЛИЗ НА 45m)
                logger.info(f"⚪ {symbol}: Нет сигнала | "
                           f"RSI={current_45m.get('rsi', 0):.0f} | "
                           f"BB={current_45m.get('bb_position', 50):.0f}% | "
                           f"Vol={current_45m.get('volume_ratio', 0):.1f}x | "
                           f"Рынок={market_condition}")
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка анализа {symbol}: {e}")
            return None
    
    async def _create_enhanced_signal_v4(self, symbol: str, direction: str, entry_price: float,
                                       confidence: float, reasons: List[str], mtf_data: Dict,
                                       market_condition: str) -> EnhancedSignal:
        """V4.0: Создать расширенный сигнал с новыми возможностями"""
        try:
            # 1. Рассчитываем вероятности TP уровней
            tp_probabilities = []
            if self.probability_calculator:
                market_data = mtf_data.get('45m', {})  # ОСНОВНОЙ АНАЛИЗ НА 45m
                tp_probs = self.probability_calculator.calculate_tp_probabilities(
                    symbol, market_data, market_condition
                )
                tp_probabilities = tp_probs
            
            # 2. Создаем расширенные TP уровни с учетом лимитов для крупных активов и текущей волатильности
            enhanced_tp_levels = []
            
            # Определяем максимальный TP для символа (лимиты для крупных активов)
            major_assets_limits = {
                'BTCUSDT': {'max_tp_percent': 10},
                'ETHUSDT': {'max_tp_percent': 12},
                'BNBUSDT': {'max_tp_percent': 15},
            }
            max_tp_for_symbol = 20  # Дефолт для обычных активов
            if symbol.upper() in major_assets_limits:
                max_tp_for_symbol = major_assets_limits[symbol.upper()]['max_tp_percent']
                logger.info(f"🔒 {symbol}: Применен лимит максимального TP: {max_tp_for_symbol}%")

            # Динамический лимит по ATR: чем меньше волатильность, тем ниже допустимый TP
            market_45m = mtf_data.get('45m', {}) or {}  # ОСНОВНОЙ АНАЛИЗ НА 45m
            price_45m = float(market_45m.get('price', entry_price) or entry_price)
            atr_45m = float(market_45m.get('atr', 0) or 0)
            atr_percent = (atr_45m / price_45m * 100) if price_45m > 0 else 0.0
            # Для крупных активов: лимит = min(фикс.лимит, max(6%, ATR% * 2.5))
            # Для прочих активов: лимит = min(фикс.лимит, max(12%, ATR% * 3.0))
            if symbol.upper() in major_assets_limits:
                dynamic_tp_limit = max(6.0, atr_percent * 2.5)
            else:
                dynamic_tp_limit = max(12.0, atr_percent * 3.0)
            effective_tp_limit = min(max_tp_for_symbol, dynamic_tp_limit)
            logger.info(f"📏 {symbol}: ATR={atr_percent:.2f}% → динамический лимит TP={effective_tp_limit:.1f}% (жесткий={max_tp_for_symbol}%)")
            
            for i, tp_config in enumerate(self.TP_LEVELS_V4):
                # Пропускаем TP уровни, которые превышают эффективный лимит
                if tp_config['percent'] > effective_tp_limit:
                    logger.info(f"⏭️ {symbol}: TP{tp_config['level']} пропущен ({tp_config['percent']}% > лимит {effective_tp_limit:.1f}%)")
                    continue
                
                tp_price = entry_price * (1 + tp_config['percent'] / 100) if direction == 'buy' else entry_price * (1 - tp_config['percent'] / 100)
                
                # Получаем вероятность из расчета или используем дефолтную
                probability = 85 - (i * 10)  # Дефолтные вероятности
                confidence_interval = (probability - 5, probability + 5)
                
                if i < len(tp_probabilities):
                    tp_prob = tp_probabilities[i]
                    probability = tp_prob.probability
                    confidence_interval = tp_prob.confidence_interval
                
                enhanced_tp = EnhancedTakeProfitLevel(
                    level=tp_config['level'],
                    price=tp_price,
                    percent=tp_config['percent'],
                    probability=probability,
                    confidence_interval=confidence_interval,
                    pnl_percent=tp_config['percent'],
                    close_percent=tp_config['portion'],
                    market_condition_factor=1.0
                )
                enhanced_tp_levels.append(enhanced_tp)

            # 2.1 ГАРАНТИЯ МИНИМАЛЬНОЙ ПРИБЫЛИ: +$1 (≈ +4% от позиции)
            try:
                position_notional = float(self.POSITION_SIZE) * float(self.LEVERAGE)
                expected_profit_usd = sum([
                    (tp.close_percent * (tp.percent / 100.0)) * position_notional for tp in enhanced_tp_levels[:3]
                ])
                if expected_profit_usd < 1.0:
                    # Увеличиваем долю ранних TP, уменьшая поздние, сохраняя сумму=1.0
                    # Стратегия: поднять TP1 до 0.50, TP2 до 0.25, TP3 до 0.20, хвост 0.05
                    total_tail = 1.0 - (0.50 + 0.25 + 0.20)
                    for tp in enhanced_tp_levels:
                        if tp.level == 1:
                            tp.close_percent = 0.50
                        elif tp.level == 2:
                            tp.close_percent = 0.25
                        elif tp.level == 3:
                            tp.close_percent = 0.20
                        else:
                            # распределяем остаток равномерно по хвосту
                            tp.close_percent = max(0.0, total_tail / max(1, (len(enhanced_tp_levels) - 3)))

                    # Повторная проверка ожиданий
                    expected_profit_usd = sum([
                        (tp.close_percent * (tp.percent / 100.0)) * position_notional for tp in enhanced_tp_levels[:3]
                    ])
                    logger.info(f"🛡️ Гарантия +$1: ожидаемая прибыль по TP1-TP3 скорректирована до ${expected_profit_usd:.2f}")
            except Exception as e:
                logger.debug(f"⚠️ Ошибка перераспределения TP для гарантии +$1: {e}")
            
            # 3. Оценка стратегии
            strategy_score = 10.0  # Дефолтная оценка
            if self.strategy_evaluator:
                signal_data = {
                    'direction': direction,
                    'confidence': confidence,
                    'reasons': reasons
                }
                market_data = mtf_data.get('30m', {})
                score_result = self.strategy_evaluator.evaluate_strategy(
                    signal_data, market_data, market_condition
                )
                strategy_score = score_result.total_score
            
            # 4. Проверка реалистичности
            realism_check = None
            if self.realism_validator:
                signal_data = {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'direction': direction,
                    'stop_loss_percent': self.STOP_LOSS_PERCENT,
                    'tp_levels': [{'percent': tp.percent} for tp in enhanced_tp_levels]
                }
                market_data = mtf_data.get('30m', {})
                realism_check = self.realism_validator.validate_signal(
                    signal_data, market_data, tp_probabilities
                )
            
            # 5. ML вероятность (с использованием Advanced ML System с LSTM)
            ml_probability = confidence / 100.0
            lstm_prediction = None
            
            # Используем Advanced ML System для LSTM предсказаний
            if self.advanced_ml_system:
                try:
                    # Получаем данные для создания фичей
                    df_45m = await self._fetch_ohlcv(symbol, '45m', 100)
                    if not df_45m.empty and len(df_45m) >= 50:
                        # Создаем фичи для LSTM (ОСНОВНОЙ АНАЛИЗ НА 45m)
                        current_45m_data = mtf_data.get('45m', {})
                        features = self.advanced_ml_system.create_features(df_45m, current_45m_data)
                        
                        if not features.empty:
                            # Получаем LSTM предсказание
                            prediction_result = self.advanced_ml_system.predict_price(symbol, features)
                            if prediction_result:
                                lstm_prediction = prediction_result
                                # Используем уверенность от LSTM модели
                                ml_probability = min(0.95, max(0.5, prediction_result.confidence))
                                
                                # Добавляем бонус к confidence если LSTM предсказывает правильное направление
                                if (direction == 'buy' and prediction_result.trend_direction in ['bullish', 'sideways']) or \
                                   (direction == 'sell' and prediction_result.trend_direction in ['bearish', 'sideways']):
                                    # LSTM подтверждает направление - добавляем бонус
                                    confidence += 5
                                    reasons.append(f'🧠LSTM+5%')
                                    logger.debug(f"🧠 {symbol}: LSTM подтверждает {direction.upper()} | "
                                               f"Предсказанная цена: ${prediction_result.predicted_price:.4f} | "
                                               f"Тренд: {prediction_result.trend_direction} | "
                                               f"Уверенность: {prediction_result.confidence:.0%}")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка LSTM предсказания: {e}")
            
            # Fallback на базовую ML систему если Advanced недоступна
            elif self.ml_system:
                try:
                    # Простой расчет ML вероятности на основе confidence
                    ml_probability = min(0.95, confidence / 100.0)
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка ML вероятности: {e}")
            
            # 6. Stop Loss (фиксированный $5 убыток на сделку)
            # Рассчитываем SL так, чтобы максимальный убыток = $5
            position_notional = self.POSITION_SIZE * self.LEVERAGE  # $25
            max_loss_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100  # 20%
            stop_loss = entry_price * (1 - max_loss_percent / 100) if direction == 'buy' else entry_price * (1 + max_loss_percent / 100)
            
            # Создаем расширенный сигнал
            enhanced_signal = EnhancedSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                confidence=confidence,
                strategy_score=strategy_score,
                timeframe_analysis=mtf_data,
                tp_levels=enhanced_tp_levels,
                stop_loss=stop_loss,
                realism_check=realism_check,
                ml_probability=ml_probability,
                market_condition=market_condition,
                reasons=reasons
            )
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания расширенного сигнала: {e}")
            # Возвращаем базовый сигнал
            return EnhancedSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                confidence=confidence,
                strategy_score=10.0,
                timeframe_analysis=mtf_data,
                tp_levels=[],
                stop_loss=entry_price * (1 - self.MAX_STOP_LOSS_USD / self.POSITION_NOTIONAL) if direction == 'buy' else entry_price * (1 + self.MAX_STOP_LOSS_USD / self.POSITION_NOTIONAL),
                realism_check=None,
                ml_probability=confidence / 100.0,
                market_condition=market_condition,
                reasons=reasons
            )
    
    def _format_price(self, price: float) -> str:
        """
        Умное форматирование цены в зависимости от её величины
        Для маленьких цен (менее 0.01) показывает больше знаков
        """
        if price == 0:
            return "0.00"
        
        if price >= 1:
            return f"{price:.2f}"
        elif price >= 0.1:
            return f"{price:.4f}"
        elif price >= 0.01:
            return f"{price:.5f}"
        elif price >= 0.001:
            return f"{price:.6f}"
        elif price >= 0.0001:
            return f"{price:.7f}"
        elif price >= 0.00001:
            return f"{price:.8f}"
        else:
            # Для очень маленьких цен показываем научную нотацию или много знаков
            return f"{price:.10f}".rstrip('0').rstrip('.')
    
    async def send_enhanced_signal_v4(self, signal: EnhancedSignal):
        """V4.0: Отправить расширенный сигнал в Telegram"""
        try:
            if not self.telegram_bot:
                return
            
            # 🛑 ЗАЩИТА ОТ ДУБЛИКАТОВ: Проверяем, не отправляли ли мы уже этот сигнал
            signal_key = f"{signal.symbol}_{signal.direction}"
            
            if signal_key in self.last_signals:
                last_signal_data = self.last_signals[signal_key]
                last_time = last_signal_data.get('timestamp')
                if last_time:
                    # datetime уже импортирован глобально на строке 109
                    time_diff = datetime.now(WARSAW_TZ) - last_time
                    # Не отправляем тот же сигнал в течение 60 минут
                    if time_diff.total_seconds() < 3600:
                        logger.debug(f"⏭️ {signal.symbol}: Пропущен дубликат сигнала {signal.direction.upper()} "
                                   f"(последний был {int(time_diff.total_seconds()/60)} минут назад)")
                        return False  # Возвращаем False если пропустили дубликат
            
            # Формируем направление на русском
            direction_text = "Лонг" if signal.direction == 'buy' else "Шорт"
            
            # Получаем текущее количество открытых позиций
            current_positions = await self._get_current_open_positions_count()
            
            # Форматируем цены умно (для маленьких цен показывает больше знаков)
            entry_price_str = self._format_price(signal.entry_price)
            
            # Формируем сообщение (без лишних пробелов)
            message = f"""📥 #{signal.symbol} | {direction_text}
Текущая цена: {entry_price_str}

ТП: +$1 (Гарантированно) + трейлинг 0.5%"""
            
            # MTF Таймфреймы (45m - ОСНОВНОЙ)
            message += f"\n📊 MTF Таймфреймы"
            message += f"\n15m ⏩ 30m ⏩ 45m ⭐ ⏩ 1h ⏩ 4h"
            
            # Stop Loss с Trailing
            message += f"\n🛑 SL: -${self.MAX_STOP_LOSS_USD:.1f} максимум на сделку → Trailing"
            
            # Информация о торговле
            message += f"\n📈 Торговля"
            trade_size = self.POSITION_SIZE
            leverage = self.LEVERAGE
            total_size = trade_size * leverage
            message += f"\n⚡ Сделка: ${trade_size:.0f} x{leverage} = ${total_size:.0f}"
            
            # Количество позиций
            message += f"\n📌 Позиции: {current_positions}!"""
            
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ V4.0 сигнал отправлен: {signal.symbol} {signal.direction.upper()}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки V4.0 сигнала: {e}")
    
    async def _get_current_open_positions_count(self) -> int:
        """Получить текущее количество открытых позиций на бирже"""
        try:
            if not self.exchange:
                # Если exchange не инициализирован, используем словарь
                return len(self.active_positions)
            
            # Получаем открытые позиции с биржи (с правильными параметрами для Bybit)
            try:
                positions = await self.exchange.fetch_positions(params={'category': 'linear', 'accountType': 'UNIFIED'})
            except Exception as e1:
                # Попробуем без параметров
                try:
                    positions = await self.exchange.fetch_positions()
                except Exception as e2:
                    logger.warning(f"⚠️ Ошибка получения позиций: {e1} / {e2}")
                    positions = []
            
            if not positions:
                # Синхронизируем - если позиций нет на бирже, очищаем словарь
                if self.active_positions:
                    logger.info(f"📊 Синхронизация: на бирже позиций нет, очищаем локальный словарь ({len(self.active_positions)} позиций)")
                    self.active_positions.clear()
                return 0
            
            # Фильтруем только позиции с ненулевым размером
            open_positions = [p for p in positions if p.get('contracts', 0) > 0 or p.get('size', 0) > 0]
            
            # Обновляем словарь активных позиций
            current_count = len(open_positions)
            
            # Синхронизируем словарь с реальными позициями
            active_symbols = set()
            for pos in open_positions:
                symbol = pos.get('symbol', '')
                if symbol:
                    active_symbols.add(symbol)
                    # Обновляем или добавляем позицию
                    if symbol not in self.active_positions:
                        self.active_positions[symbol] = {
                            'side': pos.get('side', ''),
                            'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                            'size': pos.get('contracts', pos.get('size', 0)),
                            'pnl_percent': pos.get('percentage', 0),
                            'confidence': 0  # Будет обновлено
                        }
            
            # Удаляем закрытые позиции из словаря
            closed_symbols = set(self.active_positions.keys()) - active_symbols
            for symbol in closed_symbols:
                del self.active_positions[symbol]
            
            return current_count
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка получения позиций с биржи: {e}")
            # Fallback на словарь
            return len(self.active_positions)
    
    async def _set_position_sl_tp_bybit(self, symbol: str, side: str, size: float, 
                                        stop_loss_price: float = None, take_profit_prices: list = None) -> bool:
        """
        Устанавливает Stop Loss и Take Profit для позиции на Bybit
        
        Args:
            symbol: Торговая пара
            side: Направление позиции ('buy' или 'sell')
            size: Размер позиции
            stop_loss_price: Цена Stop Loss (опционально)
            take_profit_prices: Список цен Take Profit (опционально)
        
        Returns:
            True если хотя бы один ордер установлен успешно
        """
        success = False
        
        try:
            # Нормализуем символ для Bybit
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            # 1. Устанавливаем Stop Loss через conditional order
            if stop_loss_price:
                try:
                    # Определяем направление триггера для SL
                    # Для LONG (buy): SL срабатывает когда цена идет ВНИЗ (descending)
                    # Для SHORT (sell): SL срабатывает когда цена идет ВВЕРХ (ascending)
                    if side == 'buy':
                        trigger_direction_sl = 'descending'  # Цена падает до SL
                    else:
                        trigger_direction_sl = 'ascending'  # Цена растет до SL
                    
                    # Для Bybit используем правильный метод создания conditional order
                    sl_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='StopMarket',
                        side='sell' if side == 'buy' else 'buy',
                        amount=size,
                        params={
                            'category': 'linear',
                            'stopPrice': stop_loss_price,
                            'triggerPrice': stop_loss_price,
                            'triggerDirection': trigger_direction_sl,  # КРИТИЧНО для Bybit!
                            'reduceOnly': True,
                            'closeOnTrigger': True,
                            'positionIdx': 0  # 0 - one-way mode, 1 - hedge-mode buy, 2 - hedge-mode sell
                        }
                    )
                    logger.info(f"🛑 {symbol}: Stop Loss установлен через conditional order: ${stop_loss_price:.4f} | Order ID: {sl_order.get('id', 'N/A')}")
                    success = True
                except Exception as e1:
                    logger.warning(f"⚠️ {symbol}: Не удалось установить SL через conditional order: {e1}")
                    # Пробуем альтернативный метод через прямой API вызов
                    try:
                        # Используем прямое обращение к Bybit API
                        if hasattr(self.exchange, 'private_post_position_trading_stop'):
                            response = await self.exchange.private_post_position_trading_stop({
                                'category': 'linear',
                                'symbol': bybit_symbol,
                                'stopLoss': str(stop_loss_price),
                                'positionIdx': 0
                            })
                            if response.get('retCode') == 0:
                                logger.info(f"🛑 {symbol}: Stop Loss установлен через прямой API: ${stop_loss_price:.4f}")
                                success = True
                            else:
                                logger.warning(f"⚠️ {symbol}: Ошибка установки SL через прямой API: {response}")
                        else:
                            logger.debug(f"⚠️ Метод private_post_position_trading_stop недоступен")
                            
                        # Пробуем через pybit (работает для существующих позиций)
                        if not success:
                            pybit_success = await self._set_sl_tp_pybit(symbol, stop_loss_price, None)
                            if pybit_success:
                                logger.info(f"🛑 {symbol}: Stop Loss установлен через pybit: ${stop_loss_price:.4f}")
                                success = True
                    except Exception as e2:
                        logger.warning(f"⚠️ {symbol}: Не удалось установить SL через прямой API: {e2}")
                        # Пробуем pybit как последний вариант
                        if not success:
                            pybit_success = await self._set_sl_tp_pybit(symbol, stop_loss_price, None)
                            if pybit_success:
                                logger.info(f"🛑 {symbol}: Stop Loss установлен через pybit: ${stop_loss_price:.4f}")
                                success = True
                            else:
                                logger.info(f"📝 {symbol}: Stop Loss будет контролироваться через мониторинг на ${stop_loss_price:.4f}")
            
            # 2. Устанавливаем Take Profit ордера для каждого уровня
            if take_profit_prices:
                # Определяем направление триггера для TP
                # Для LONG (buy): TP срабатывает когда цена идет ВВЕРХ (ascending)
                # Для SHORT (sell): TP срабатывает когда цена идет ВНИЗ (descending)
                if side == 'buy':
                    trigger_direction_tp = 'ascending'  # Цена растет до TP
                else:
                    trigger_direction_tp = 'descending'  # Цена падает до TP
                
                for i, tp_price in enumerate(take_profit_prices, 1):
                    tp_set = False
                    try:
                        # Пробуем через conditional order
                        tp_order = await self.exchange.create_order(
                            symbol=symbol,
                            type='TakeProfitMarket',
                            side='sell' if side == 'buy' else 'buy',
                            amount=size,  # Для частичного закрытия нужно будет указывать часть
                            params={
                                'category': 'linear',
                                'stopPrice': tp_price,
                                'triggerPrice': tp_price,
                                'triggerDirection': trigger_direction_tp,  # КРИТИЧНО для Bybit!
                                'reduceOnly': True,
                                'closeOnTrigger': True,
                                'positionIdx': 0
                            }
                        )
                        logger.info(f"🎯 {symbol}: TP{i} установлен через conditional order: ${tp_price:.4f} | Order ID: {tp_order.get('id', 'N/A')}")
                        success = True
                        tp_set = True
                    except Exception as e:
                        logger.debug(f"⚠️ {symbol}: Не удалось установить TP{i} через conditional order: {e}")
                        # Пробуем через прямой API для существующих позиций
                        try:
                            if hasattr(self.exchange, 'private_post_position_trading_stop'):
                                response = await self.exchange.private_post_position_trading_stop({
                                    'category': 'linear',
                                    'symbol': bybit_symbol,
                                    'takeProfit': str(tp_price),
                                    'positionIdx': 0
                                })
                                if response.get('retCode') == 0:
                                    logger.info(f"🎯 {symbol}: TP{i} установлен через прямой API: ${tp_price:.4f}")
                                    success = True
                                    tp_set = True
                                else:
                                    logger.debug(f"⚠️ {symbol}: Ошибка установки TP{i} через прямой API: {response}")
                        except Exception as e_api:
                            logger.debug(f"⚠️ {symbol}: Не удалось установить TP{i} через прямой API: {e_api}")
                        
                        # Пробуем через pybit (работает для существующих позиций)
                        if not tp_set:
                            pybit_success = await self._set_sl_tp_pybit(symbol, None, tp_price)
                            if pybit_success:
                                logger.info(f"🎯 {symbol}: TP{i} установлен через pybit: ${tp_price:.4f}")
                                success = True
                                tp_set = True
                    
                    if not tp_set:
                        logger.info(f"📝 {symbol}: TP{i} будет контролироваться через мониторинг на ${tp_price:.4f}")
        
        except Exception as e:
            logger.error(f"❌ {symbol}: Ошибка установки SL/TP: {e}")
        
        return success
    
    async def _set_sl_tp_pybit(self, symbol: str, stop_loss_price: float = None, take_profit_price: float = None) -> bool:
        """
        Устанавливает SL/TP используя официальную библиотеку pybit
        Работает для существующих позиций
        """
        try:
            try:
                from pybit.unified_trading import HTTP
            except ImportError:
                logger.warning("⚠️ pybit не установлена. Установите: pip install pybit")
                return False
            
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            session = HTTP(
                testnet=False,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            params = {
                'category': 'linear',
                'symbol': bybit_symbol,
                'positionIdx': 0
            }
            
            if stop_loss_price:
                params['stopLoss'] = str(stop_loss_price)
            if take_profit_price:
                params['takeProfit'] = str(take_profit_price)
            
            response = session.set_trading_stop(**params)
            
            if response.get('retCode') == 0:
                return True
            else:
                logger.debug(f"⚠️ {symbol}: Ошибка pybit set_trading_stop: {response}")
                return False
                
        except Exception as e:
            logger.debug(f"⚠️ {symbol}: Ошибка установки SL/TP через pybit: {e}")
            return False
    
    async def _update_stop_loss_on_exchange(self, symbol: str, stop_loss_price: float) -> bool:
        """
        Обновляет Stop Loss на бирже для открытой позиции
        
        Args:
            symbol: Торговая пара
            stop_loss_price: Новая цена Stop Loss
        
        Returns:
            True если успешно обновлено
        """
        try:
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            # Используем метод бота для установки SL/TP через прямой API
            return await self._set_position_sl_tp_bybit(
                symbol=symbol,
                side='buy',  # Направление не важно для обновления SL
                size=0,  # Размер не важен для обновления
                stop_loss_price=stop_loss_price,
                take_profit_prices=None
            )
        except Exception as e:
            logger.debug(f"⚠️ {symbol}: Ошибка обновления SL: {e}")
            return False
    
    async def add_sl_tp_to_existing_position(self, symbol: str, side: str, entry_price: float) -> bool:
        """
        Добавляет SL/TP к существующей позиции на бирже
        
        Args:
            symbol: Торговая пара
            side: Направление позиции ('buy' или 'sell')
            entry_price: Цена входа для расчета SL/TP
        
        Returns:
            True если SL/TP установлены успешно
        """
        try:
            # Рассчитываем SL и TP
            position_notional = self.POSITION_SIZE * self.LEVERAGE
            stop_loss_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100
            tp_percent = 1.0  # Стартовый TP +1% с дальнейшим шаговым трейлингом до +5%
            
            if side == 'buy':
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + tp_percent / 100.0)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - tp_percent / 100.0)
            
            # Получаем размер позиции
            positions = await self.exchange.fetch_positions(params={'category': 'linear'})
            position = next((p for p in positions if p.get('symbol') == symbol and 
                           (p.get('contracts', 0) or p.get('size', 0)) > 0), None)
            
            if not position:
                logger.warning(f"⚠️ {symbol}: Позиция не найдена")
                return False
            
            size = float(position.get('contracts', 0) or position.get('size', 0))
            
            # Используем pybit для установки SL/TP (лучше работает для существующих позиций)
            success = await self._set_sl_tp_pybit(symbol, stop_loss_price, tp_price)
            
            # Если pybit не сработала, пробуем обычный метод
            if not success:
                success = await self._set_position_sl_tp_bybit(
                    symbol=symbol,
                    side=side,
                    size=size,
                    stop_loss_price=stop_loss_price,
                    take_profit_prices=[tp_price]
                )
            
            if success:
                logger.info(f"✅ {symbol}: SL/TP добавлены к существующей позиции | SL: ${stop_loss_price:.4f}, TP: ${tp_price:.4f}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Ошибка добавления SL/TP к существующей позиции: {e}")
            return False
    
    async def open_position_automatically(self, signal: EnhancedSignal) -> bool:
        """
        🚀 Автоматически открывает позицию на бирже
        
        Args:
            signal: Торговый сигнал
        
        Returns:
            True если позиция успешно открыта, False в противном случае
        """
        try:
            if not self.exchange:
                logger.error("❌ Биржа не инициализирована")
                return False
            
            symbol = signal.symbol
            side = 'Buy' if signal.direction == 'buy' else 'Sell'
            
            # Проверяем, нет ли уже открытой позиции по этому символу
            if symbol in self.active_positions:
                logger.warning(f"⚠️ {symbol}: Позиция уже в активных позициях")
                return False
            
            try:
                positions = await self.exchange.fetch_positions([symbol], params={'category': 'linear'})
                for pos in positions:
                    size = pos.get('contracts', 0) or pos.get('size', 0)
                    if size > 0:
                        logger.warning(f"⚠️ {symbol}: Позиция уже открыта на бирже (размер: {size})")
                        # Добавляем в активные позиции для синхронизации
                        self.active_positions[symbol] = {
                            'side': pos.get('side', ''),
                            'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                            'size': size,
                            'pnl_percent': pos.get('percentage', 0)
                        }
                        return False
            except Exception as e:
                logger.debug(f"⚠️ Ошибка проверки позиций для {symbol}: {e}")
            
            # 1. Устанавливаем плечо (для Bybit используется другой метод)
            try:
                # Для Bybit используем правильный метод установки плеча
                # Нужно использовать правильный формат для unified account
                await self.exchange.set_leverage(
                    self.LEVERAGE, 
                    symbol,
                    params={
                        'category': 'linear',
                        'symbol': symbol
                    }
                )
                logger.info(f"✅ {symbol}: Плечо установлено {self.LEVERAGE}x")
            except Exception as e1:
                # Пробуем альтернативный способ
                try:
                    # Для unified account может потребоваться другой формат
                    await self.exchange.set_leverage(self.LEVERAGE, symbol)
                    logger.info(f"✅ {symbol}: Плечо установлено {self.LEVERAGE}x (альтернативный метод)")
                except Exception as e2:
                    logger.warning(f"⚠️ {symbol}: Не удалось установить плечо: {e1} / {e2}. Продолжаем...")
                    # Продолжаем, так как плечо может быть уже установлено глобально
            
            # 2. СТРОГАЯ ПРОВЕРКА БАЛАНСА перед открытием позиции
            try:
                balance = await self.exchange.fetch_balance({'accountType': 'UNIFIED'})
                usdt_info = balance.get('USDT', {})
                if isinstance(usdt_info, dict):
                    available_balance = usdt_info.get('free', 0) or usdt_info.get('available', 0) or 0
                    total_balance = usdt_info.get('total', 0) or usdt_info.get('used', 0) + available_balance
                else:
                    available_balance = float(usdt_info) if usdt_info else 0
                    total_balance = available_balance
                
                # КРИТИЧНАЯ ПРОВЕРКА: минимум баланса для торговли
                if available_balance < self.MIN_BALANCE_FOR_TRADING:
                    logger.error(f"❌ {symbol}: НЕДОСТАТОЧНО БАЛАНСА! Доступно: ${available_balance:.2f}, требуется минимум: ${self.MIN_BALANCE_FOR_TRADING:.2f} для одной позиции")
                    logger.warning(f"⚠️ Общий баланс: ${total_balance:.2f}, доступно: ${available_balance:.2f}")
                    return False
                
                if available_balance <= 0:
                    logger.error(f"❌ {symbol}: Нет доступного баланса для открытия позиции (баланс: ${available_balance:.2f})")
                    return False
                
                # Рассчитываем требуемую маржу для позиции
                # Для позиции $25 с плечом 5x нужна маржа $5
                required_margin = self.POSITION_SIZE  # $5
                
                # Проверяем, сколько уже используется в открытых позициях
                current_positions_count = await self._get_current_open_positions_count()
                used_margin = current_positions_count * self.POSITION_SIZE  # Маржа для каждой позиции $5
                total_required = used_margin + required_margin
                
                # КРИТИЧНАЯ ПРОВЕРКА: достаточно ли баланса для новой позиции
                if available_balance < required_margin:
                    logger.error(f"❌ {symbol}: Недостаточно баланса! Требуется: ${required_margin:.2f}, доступно: ${available_balance:.2f}")
                    logger.warning(f"⚠️ Общий баланс: ${total_balance:.2f} | Использовано: ${used_margin:.2f} ({current_positions_count} позиций)")
                    return False
                
                # КРИТИЧНАЯ ПРОВЕРКА: не превысим ли максимальную маржу (3 позиции = $15)
                if total_required > available_balance:
                    logger.error(f"❌ {symbol}: Недостаточно баланса с учетом открытых позиций!")
                    logger.error(f"   Используется: ${used_margin:.2f} ({current_positions_count}/{self.MAX_POSITIONS} позиций)")
                    logger.error(f"   Требуется еще: ${required_margin:.2f}")
                    logger.error(f"   Доступно: ${available_balance:.2f}")
                    logger.error(f"   Общий баланс: ${total_balance:.2f}")
                    return False
                
                # Дополнительная проверка: оставляем небольшой резерв
                reserve_margin = 0.50  # $0.50 резерв
                if (available_balance - total_required) < reserve_margin:
                    logger.warning(f"⚠️ {symbol}: После открытия позиции останется мало баланса (${available_balance - total_required:.2f} < ${reserve_margin:.2f})")
                
                logger.info(f"💰 {symbol}: Баланс проверен | Доступно: ${available_balance:.2f} | Используется: ${used_margin:.2f} ({current_positions_count}/{self.MAX_POSITIONS}) | Требуется: ${required_margin:.2f} | Останется: ${available_balance - total_required:.2f}")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Ошибка проверки баланса: {e}")
                return False
            
            # 3. Рассчитываем размер позиции
            entry_price = signal.entry_price
            position_size_usdt = self.POSITION_SIZE  # $5
            position_notional = position_size_usdt * self.LEVERAGE  # $25 с плечом 5x
            
            # Получаем информацию о рынке для расчета количества контрактов
            try:
                market = self.exchange.market(symbol)
                if not market:
                    logger.error(f"❌ {symbol}: Рынок не найден")
                    return False
                
                # Для маржинальных контрактов используем не размер в USDT, а количество контрактов
                # Минимальный размер обычно указан в market['limits']['amount']['min']
                contract_size = market.get('contractSize', 1)
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
                
                # Рассчитываем количество контрактов
                # Для USDT-контрактов: количество = notional / цена
                if market.get('linear'):
                    # Линейные контракты (USDT)
                    qty = position_notional / entry_price
                    # Округляем до минимального шага
                    precision = market.get('precision', {}).get('amount', 0.001)
                    qty = round(qty / precision) * precision
                    qty = max(qty, min_amount)
                    
                    # Проверяем фактический размер позиции после округления
                    actual_notional = qty * entry_price
                    actual_margin = actual_notional / self.LEVERAGE
                    
                    # КРИТИЧНАЯ ПРОВЕРКА: фактическая маржа не должна превышать $5.10 (с учетом округления)
                    max_allowed_margin = self.POSITION_SIZE * 1.02  # Максимум $5.10 (2% запас на округление)
                    if actual_margin > max_allowed_margin:
                        logger.error(f"❌ {symbol}: Фактическая маржа (${actual_margin:.2f}) превышает допустимую (${max_allowed_margin:.2f})!")
                        logger.error(f"   Рассчитываем меньший размер позиции...")
                        # Пересчитываем с точным контролем маржи - максимальная нотиональная стоимость $25
                        max_allowed_notional = self.POSITION_SIZE * self.LEVERAGE  # Максимум $25
                        qty = max_allowed_notional / entry_price
                        qty = round(qty / precision) * precision
                        qty = max(qty, min_amount)
                        actual_notional = qty * entry_price
                        actual_margin = actual_notional / self.LEVERAGE
                        # Если все еще больше - уменьшаем
                        if actual_margin > max_allowed_margin:
                            # Уменьшаем количество на один шаг precision
                            qty = qty - precision
                            qty = max(qty, min_amount)
                            actual_notional = qty * entry_price
                            actual_margin = actual_notional / self.LEVERAGE
                        logger.warning(f"⚠️ {symbol}: Размер скорректирован | Маржа: ${actual_margin:.2f}")
                    
                    # КРИТИЧНАЯ ПРОВЕРКА: фактическая маржа не должна превышать доступный баланс
                    if actual_margin > available_balance:
                        logger.error(f"❌ {symbol}: Фактическая маржа (${actual_margin:.2f}) превышает доступный баланс (${available_balance:.2f})")
                        return False
                    
                    # Логируем информацию о размере позиции
                    logger.info(f"📊 {symbol}: Размер позиции | Нотиональная: ${actual_notional:.2f} | Маржа: ${actual_margin:.2f} (контроль: максимум ${self.POSITION_SIZE:.2f})")
                else:
                    logger.error(f"❌ {symbol}: Поддерживаются только линейные контракты")
                    return False
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Ошибка расчета размера позиции: {e}")
                return False
            
            # 3. Открываем позицию (Market Order)
            try:
                logger.info(f"🚀 {symbol}: Открываю позицию {side} | Размер: {qty:.6f} | Цена входа: ${entry_price:.4f}")
                
                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side='buy' if signal.direction == 'buy' else 'sell',
                    amount=qty,
                    params={
                        'category': 'linear',
                        'reduceOnly': False
                    }
                )
                
                logger.info(f"✅ {symbol}: Ордер размещен | ID: {order.get('id', 'N/A')}")
                
                # Ждем немного для подтверждения позиции
                await asyncio.sleep(2)
                
                # Проверяем что позиция действительно открылась
                try:
                    try:
                        positions = await self.exchange.fetch_positions([symbol], params={'category': 'linear'})
                    except:
                        positions = await self.exchange.fetch_positions([symbol])
                    position_opened = False
                    for pos in positions:
                        size = pos.get('contracts', 0) or pos.get('size', 0)
                        if size > 0:
                            position_opened = True
                            logger.info(f"✅ {symbol}: Позиция подтверждена на бирже! Размер: {size}")
                            break
                    
                    if not position_opened:
                        logger.warning(f"⚠️ {symbol}: Позиция не найдена на бирже после открытия ордера")
                        # Пробуем еще раз через секунду
                        await asyncio.sleep(1)
                        try:
                            positions = await self.exchange.fetch_positions([symbol], params={'category': 'linear'})
                        except:
                            positions = await self.exchange.fetch_positions([symbol])
                        for pos in positions:
                            size = pos.get('contracts', 0) or pos.get('size', 0)
                            if size > 0:
                                position_opened = True
                                logger.info(f"✅ {symbol}: Позиция найдена после повторной проверки. Размер: {size}")
                                break
                        
                        if not position_opened:
                            logger.error(f"❌ {symbol}: Позиция не открыта! Возможно ордер не исполнился.")
                            return False
                            
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Не удалось проверить позицию: {e}. Продолжаем...")
                
                # 4. Устанавливаем Stop Loss и Take Profit через правильный метод
                stop_loss_price = signal.stop_loss
                
                # Один TP +10% (стратегия фиксированного профита $2.5)
                tp_percent = 10.0  # +10%
                if signal.direction == 'buy':
                    tp_price = entry_price * (1 + tp_percent / 100.0)
                else:
                    tp_price = entry_price * (1 - tp_percent / 100.0)
                tp_prices = [tp_price]  # Список с одним TP
                
                # Используем новую функцию для установки SL/TP
                sl_tp_set = await self._set_position_sl_tp_bybit(
                    symbol=symbol,
                    side=signal.direction,
                    size=qty,
                    stop_loss_price=stop_loss_price,
                    take_profit_prices=tp_prices
                )
                
                if not sl_tp_set:
                    logger.warning(f"⚠️ {symbol}: SL/TP не установлены на бирже. Контролируются через мониторинг.")
                else:
                    logger.info(f"✅ {symbol}: SL/TP установлены на бирже!")
                
                # 5. Сохраняем информацию о позиции
                self.active_positions[symbol] = {
                    'side': signal.direction,
                    'entry_price': entry_price,
                    'size': qty,
                    'stop_loss': stop_loss_price,
                    'initial_sl': stop_loss_price,  # Начальный SL для трейлинга
                    'tp_levels': signal.tp_levels,
                    'signal': signal,
                    'opened_at': datetime.now(WARSAW_TZ),
                    'order_id': order.get('id'),
                    'leverage': self.LEVERAGE,
                    'position_notional': position_notional,  # $25
                    'max_loss_usd': self.MAX_STOP_LOSS_USD  # $5
                }
                
                # 6. Обновляем статистику
                self.performance_stats['total_trades'] = self.performance_stats.get('total_trades', 0) + 1
                
                logger.info(f"✅ {symbol}: Позиция успешно открыта! | Размер: {qty:.6f} | SL: ${stop_loss_price:.4f}")
                return True
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Ошибка открытия позиции: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка автоматического открытия позиции для {signal.symbol}: {e}", exc_info=True)
            return False
    
    async def monitor_positions(self):
        """
        📊 Мониторинг открытых позиций и автоматическое закрытие по TP/SL
        """
        try:
            if not self.exchange:
                return
            
            # Получаем все открытые позиции с биржи (с правильными параметрами для Bybit)
            try:
                positions = await self.exchange.fetch_positions(params={'category': 'linear'})
            except Exception as e1:
                # Fallback: пробуем без параметров
                try:
                    positions = await self.exchange.fetch_positions()
                except Exception as e2:
                    logger.warning(f"⚠️ Ошибка получения позиций для мониторинга: {e1} / {e2}")
                    positions = []
            except Exception as e:
                logger.debug(f"⚠️ Ошибка получения позиций: {e}")
                return
            
            if not positions:
                # Если позиций нет, очищаем словарь
                if self.active_positions:
                    self.active_positions.clear()
                return
            
            # Фильтруем только позиции с размером > 0
            open_positions = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]

            # === Снимок прошлых позиций (для детекта частичных закрытий) ===
            import json, os
            state_path = '/opt/bot/state/positions.json'
            prev_snapshot = {}
            try:
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        prev_snapshot = json.load(f) or {}
            except Exception:
                prev_snapshot = {}
            
            for position in open_positions:
                try:
                    symbol = position.get('symbol', '')
                    if not symbol:
                        continue
                    
                    current_price = position.get('markPrice', 0) or position.get('lastPrice', 0)
                    size = position.get('contracts', 0) or position.get('size', 0)
                    side = position.get('side', '')
                    entry_price = position.get('entryPrice', 0) or position.get('averagePrice', 0)
                    
                    if not current_price or not entry_price or not size:
                        continue
                    
                    # Получаем информацию о позиции из нашего словаря
                    pos_info = self.active_positions.get(symbol)
                    if not pos_info:
                        # Если позиция есть на бирже, но не в нашем словаре - пропускаем
                        continue
                    
                    signal = pos_info.get('signal')
                    if not signal:
                        continue
                    
                    # MTF валидация 45m+1h+4h: авто-закрытие, если подтверждения нет
                    try:
                        mtf_data_live = await self._fetch_multi_timeframe_data(symbol)
                        c45 = mtf_data_live.get('45m', {}) or {}
                        c1h = mtf_data_live.get('1h', {}) or {}
                        c4h = mtf_data_live.get('4h', {}) or {}
                        def _confirmed(dir_):
                            if dir_ == 'buy':
                                return (
                                    c45.get('ema_9', 0) > c45.get('ema_21', 0)
                                    and c1h.get('ema_9', 0) > c1h.get('ema_21', 0)
                                    and c4h.get('ema_9', 0) > c4h.get('ema_21', 0)
                                )
                            if dir_ == 'sell':
                                return (
                                    c45.get('ema_9', 0) < c45.get('ema_21', 0)
                                    and c1h.get('ema_9', 0) < c1h.get('ema_21', 0)
                                    and c4h.get('ema_9', 0) < c4h.get('ema_21', 0)
                                )
                            return False
                        intended_dir = 'buy' if (side.lower() == 'long' or signal.direction == 'buy') else 'sell'
                        if not _confirmed(intended_dir):
                            logger.warning(f"🚫 {symbol}: нет подтверждения 45m+1h+4h для {intended_dir.upper()} — закрываем позицию")
                            try:
                                await self.exchange.create_market_order(
                                    symbol=symbol,
                                    side='sell' if intended_dir == 'buy' else 'buy',
                                    amount=size,
                                    params={'category': 'linear', 'reduceOnly': True}
                                )
                                if self.telegram_bot:
                                    await self.send_position_closed_v4(
                                        symbol=symbol,
                                        side=intended_dir,
                                        entry_price=float(entry_price),
                                        exit_price=float(current_price),
                                        pnl_percent=0.0,
                                        profit_usd=0.0,
                                        reason='Авто-закрытие: нет подтверждения 45m+1h+4h'
                                    )
                            except Exception as e:
                                logger.error(f"❌ {symbol}: Ошибка авто-закрытия без подтверждения: {e}")
                            # Удаляем и переходим к следующей позиции
                            if symbol in self.active_positions:
                                del self.active_positions[symbol]
                            continue
                    except Exception as _:
                        pass

                    # Рассчитываем текущий PnL
                    if side.lower() == 'long' or signal.direction == 'buy':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # === Перенос SL в безубыток после +$1.0 (с подушкой на fee/скольжение) ===
                    try:
                        position_notional = pos_info.get('position_notional', self.POSITION_NOTIONAL)
                        current_pnl_usd = pnl_percent / 100 * position_notional
                        be_flag_key = 'be_sl_set'
                        if current_pnl_usd >= 1.0 and not pos_info.get(be_flag_key):
                            pad = 0.001  # 0.1% подушка
                            if (side.lower() == 'long') or (signal.direction == 'buy'):
                                be_sl = entry_price * (1 - pad)
                            else:
                                be_sl = entry_price * (1 + pad)
                            # Обновляем на бирже через pybit (более надёжно для существующих позиций)
                            pybit_ok = await self._set_sl_tp_pybit(symbol, stop_loss_price=be_sl)
                            if pybit_ok:
                                pos_info['stop_loss'] = be_sl
                                pos_info[be_flag_key] = True
                                logger.info(f"🟩 {symbol}: SL перенесён в безубыток: ${be_sl:.6f}")
                                if self.telegram_bot:
                                    await self.send_telegram_v4(
                                        f"🟩 {symbol}: SL перенесён в безубыток\n"
                                        f"Вход: ${entry_price:.6f} → SL: ${be_sl:.6f}\n"
                                        f"Учтена подушка 0.1% для fee/скольжения"
                                    )
                    except Exception as _:
                        pass

                    # Проверяем Take Profit уровни
                    tp_levels = pos_info.get('tp_levels', signal.tp_levels)
                    closed_tps = pos_info.get('closed_tps', set())
                    
                    for tp in tp_levels:
                        if tp.level in closed_tps:
                            continue
                        
                        # Проверяем, достигнут ли TP уровень
                        if signal.direction == 'buy':
                            tp_hit = current_price >= tp.price
                        else:
                            tp_hit = current_price <= tp.price
                        
                        if tp_hit:
                            # Закрываем часть позиции
                            close_percent = tp.close_percent / 100
                            close_size = size * close_percent

                            try:
                                close_order = await self.exchange.create_market_order(
                                    symbol=symbol,
                                    side='sell' if signal.direction == 'buy' else 'buy',
                                    amount=close_size,
                                    params={
                                        'category': 'linear',
                                        'reduceOnly': True
                                    }
                                )

                                # Оценка прибыли по закрытой части (с учётом fee)
                                position_notional = float(pos_info.get('position_notional', self.POSITION_NOTIONAL) or self.POSITION_NOTIONAL)
                                closed_notional = position_notional * close_percent
                                if (signal.direction == 'buy'):
                                    profit_chunk = (current_price - entry_price) * close_size
                                else:
                                    profit_chunk = (entry_price - current_price) * close_size
                                fee_estimate = closed_notional * 0.0006  # ~0.06% round-trip
                                profit_chunk_net = profit_chunk - fee_estimate
                                tp_pct_display = (tp.percent if hasattr(tp, 'percent') else None)
                                tp_pct_text = f"+{tp_pct_display:.0f}%" if tp_pct_display is not None else f"{pnl_percent:.2f}%"

                                logger.info(f"✅ {symbol}: TP{tp.level} достигнут! Закрыто {close_percent*100:.0f}% позиции на ${tp.price:.4f} | ~${profit_chunk_net:.2f} net")

                                # Отмечаем TP как закрытый
                                closed_tps.add(tp.level)
                                pos_info['closed_tps'] = closed_tps

                                # Отправляем уведомление в Telegram с явной суммой прибыли
                                if self.telegram_bot:
                                    try:
                                        await self.send_telegram_v4(
                                            f"✅ {symbol}: TP{tp.level} достигнут\n"
                                            f"Цена: ${tp.price:.5f}\n"
                                            f"Закрыто: {close_percent*100:.0f}%\n"
                                            f"🎯 TP: {tp_pct_text} (от ${position_notional:.0f}) → ${profit_chunk_net:+.2f}"
                                        )
                                    except Exception:
                                        logger.debug("⚠️ Ошибка отправки Telegram по TP")

                            except Exception as e:
                                logger.error(f"❌ {symbol}: Ошибка закрытия TP{tp.level}: {e}")
                    
                    # Проверяем Stop Loss с трейлингом
                    stop_loss = pos_info.get('stop_loss', signal.stop_loss)
                    initial_sl = pos_info.get('initial_sl', stop_loss)
                    
                    # Трейлинг стоп: если цена в прибыли, двигаем SL следом
                    current_pnl_usd = pnl_percent / 100 * (pos_info.get('position_notional', self.POSITION_NOTIONAL))
                    
                    if stop_loss:
                        # Обновляем трейлинг стоп
                        trailing_distance = self.MAX_STOP_LOSS_USD / (pos_info.get('position_notional', self.POSITION_NOTIONAL)) * entry_price
                        
                        if signal.direction == 'buy':
                            # Для лонга: если цена выше входа, двигаем SL вверх
                            if current_price > entry_price:
                                new_sl = current_price - trailing_distance
                                if new_sl > stop_loss:
                                    stop_loss = new_sl
                                    pos_info['stop_loss'] = stop_loss
                                    logger.info(f"📈 {symbol}: Трейлинг SL обновлен: ${stop_loss:.4f}")
                            
                            sl_hit = current_price <= stop_loss
                        else:
                            # Для шорта: если цена ниже входа, двигаем SL вниз
                            if current_price < entry_price:
                                new_sl = current_price + trailing_distance
                                if new_sl < stop_loss or stop_loss == initial_sl:
                                    stop_loss = new_sl
                                    pos_info['stop_loss'] = stop_loss
                                    logger.info(f"📈 {symbol}: Трейлинг SL обновлен: ${stop_loss:.4f}")
                            
                            sl_hit = current_price >= stop_loss
                        
                        if sl_hit:
                            # Закрываем всю позицию
                            try:
                                close_order = await self.exchange.create_market_order(
                                    symbol=symbol,
                                    side='sell' if signal.direction == 'buy' else 'buy',
                                    amount=size,
                                    params={
                                        'category': 'linear',
                                        'reduceOnly': True
                                    }
                                )
                                
                                logger.warning(f"🛑 {symbol}: Stop Loss сработал! Позиция закрыта на ${current_price:.4f}")
                                
                                # Удаляем позицию из словаря
                                del self.active_positions[symbol]
                                
                                # Отправляем уведомление
                                if self.telegram_bot:
                                    await self.send_telegram_v4(
                                        f"🛑 {symbol}: Stop Loss сработал!\n"
                                        f"Цена закрытия: ${current_price:.4f}\n"
                                        f"PnL: {pnl_percent:.2f}%"
                                    )
                                
                            except Exception as e:
                                logger.error(f"❌ {symbol}: Ошибка закрытия по SL: {e}")
                    
                    # Обновляем информацию о позиции
                    pos_info['current_price'] = current_price
                    pos_info['pnl_percent'] = pnl_percent

                    # === Шаговый трейлинг TP: старт 1% и шаг 0.5% до 5% ===
                    try:
                        import math
                        favorable = pnl_percent if ((signal.direction=='buy' and current_price>=entry_price) or (signal.direction=='sell' and current_price<=entry_price)) else 0.0
                        target_tp_pct = 1.0 if favorable >= 0 else 1.0
                        if favorable >= 0.5:
                            steps = math.floor(favorable / 0.5)
                            target_tp_pct = min(5.0, max(1.0, steps * 0.5))
                        last_applied = float(pos_info.get('tp_trail_percent', 0.0) or 0.0)
                        if target_tp_pct > last_applied + 1e-9:
                            if signal.direction == 'buy':
                                new_tp_price = entry_price * (1 + target_tp_pct/100.0)
                            else:
                                new_tp_price = entry_price * (1 - target_tp_pct/100.0)
                            py_ok = await self._set_sl_tp_pybit(symbol, None, new_tp_price)
                            if py_ok:
                                pos_info['tp_trail_percent'] = target_tp_pct
                                logger.info(f"🎯 {symbol}: Обновлён TP трейлингом: {target_tp_pct:.1f}% → ${new_tp_price:.6f}")
                    except Exception:
                        pass
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка мониторинга позиции {position.get('symbol', 'unknown')}: {e}")
            
            # Удаляем закрытые позиции из словаря
            active_symbols = {p.get('symbol', '') for p in open_positions if (p.get('contracts', 0) or p.get('size', 0)) > 0}
            closed_symbols = set(self.active_positions.keys()) - active_symbols
            for symbol in closed_symbols:
                try:
                    pos_info = self.active_positions.get(symbol, {})
                    signal = pos_info.get('signal')
                    side = 'buy' if (signal and signal.direction == 'buy') else 'sell'
                    entry_price = float(pos_info.get('entry_price') or 0)
                    last_price = float(pos_info.get('current_price') or 0)
                    pnl_percent = float(pos_info.get('pnl_percent') or 0)
                    profit_usd = pnl_percent / 100 * float(pos_info.get('position_notional', self.POSITION_NOTIONAL) or self.POSITION_NOTIONAL)

                    # Попытаться получить ФАКТИЧЕСКИЙ closed PnL и цены из Bybit API
                    real_entry = entry_price
                    real_exit = last_price if last_price > 0 else entry_price
                    try:
                        from pybit.unified_trading import HTTP
                        session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False, recv_window=5000, timeout=10)
                        bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
                        cp = session.get_closed_pnl(category='linear', symbol=bybit_symbol, limit=5)
                        items = cp.get('result',{}).get('list',[]) or []
                        if items:
                            it = items[0]
                            closed_pnl = float(it.get('closedPnl') or 0)
                            avg_entry = float(it.get('avgEntryPrice') or 0) or entry_price
                            avg_exit = float(it.get('avgExitPrice') or 0) or last_price
                            profit_usd = closed_pnl
                            real_entry = avg_entry
                            real_exit = avg_exit
                            # Пересчёт процента от базового нотионала
                            base_notional = float(pos_info.get('position_notional', self.POSITION_NOTIONAL) or self.POSITION_NOTIONAL)
                            pnl_percent = (profit_usd / base_notional) * 100 if base_notional > 0 else pnl_percent
                    except Exception as _:
                        pass

                    logger.info(f"✅ {symbol}: Позиция закрыта (обнаружено по сверке) | PnL=${profit_usd:.2f}")
                    if self.telegram_bot:
                        try:
                            await self.send_position_closed_v4(
                                symbol=symbol,
                                side=side,
                                entry_price=real_entry,
                                exit_price=real_exit,
                                pnl_percent=pnl_percent,
                                profit_usd=profit_usd,
                                reason="Закрыта на бирже/по TP/SL (сверка)"
                            )
                        except Exception:
                            logger.debug("⚠️ Ошибка отправки Telegram при сверке закрытия")
                finally:
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]

            # === Детект частичных закрытий с учётом комиссий и анти‑шумом ===
            try:
                curr_snapshot = {}
                for p in open_positions:
                    sym = p.get('symbol', '')
                    if not sym:
                        continue
                    curr_snapshot[sym] = {
                        'size': float(p.get('contracts', 0) or p.get('size', 0) or 0.0),
                        'side': p.get('side', ''),
                        'avgPrice': float(p.get('entryPrice', 0) or p.get('averagePrice', 0) or 0.0),
                        'markPrice': float(p.get('markPrice', 0) or p.get('lastPrice', 0) or 0.0),
                        'takeProfit': p.get('takeProfit') or '-',
                        'stopLoss': p.get('stopLoss') or '-',
                    }

                for sym, cur in curr_snapshot.items():
                    try:
                        prev = prev_snapshot.get(sym, {})
                        prev_size = float(prev.get('size', 0) or 0.0)
                        cur_size = float(cur.get('size', 0) or 0.0)
                        if prev_size > 0 and cur_size < prev_size:
                            reduced = max(0.0, prev_size - cur_size)
                            side = (cur.get('side') or '').lower()
                            entry = float(cur.get('avgPrice') or 0.0)
                            mark = float(cur.get('markPrice') or 0.0)
                            if entry > 0 and mark > 0 and reduced > 0:
                                # PnL в USDT: (mark - entry)*size для LONG; (entry - mark)*size для SHORT
                                if side == 'buy' or side == 'long':
                                    realized = (mark - entry) * reduced
                                else:
                                    realized = (entry - mark) * reduced
                                notional_closed = reduced * mark
                                fee_estimate = notional_closed * 0.0006  # ~0.06% round-trip оценка
                                realized_net = realized - fee_estimate
                                if abs(realized_net) >= 0.05:
                                    # Процент относительно исходного нотионала (если доступен)
                                    # Пытаемся взять из активной позиции
                                    try:
                                        pos_info_local = self.active_positions.get(sym, {})
                                        position_notional_local = float(pos_info_local.get('position_notional', self.POSITION_NOTIONAL) or self.POSITION_NOTIONAL)
                                    except Exception:
                                        position_notional_local = self.POSITION_NOTIONAL
                                    tp_pct_text = f"+{(realized_net/position_notional_local*100):.1f}%"
                                    msg = (
                                        f"✂️ Частичное закрытие {sym} ({'LONG' if (side=='buy' or side=='long') else 'SHORT'})\n"
                                        f"Размер: {prev_size:.6f} → {cur_size:.6f} (−{reduced:.6f})\n"
                                        f"entry={entry:.6f} | mark={mark:.6f}\n"
                                        f"🎯 TP: {tp_pct_text} (от ${position_notional_local:.0f}) → ${realized_net:+.2f}\n"
                                        f"TP={cur.get('takeProfit','-')} SL={cur.get('stopLoss','-')}"
                                    )
                                    if self.telegram_bot:
                                        try:
                                            await self.send_telegram_v4(msg)
                                        except Exception:
                                            logger.debug("⚠️ Ошибка отправки Telegram по частичному закрытию")
                    except Exception as _:
                        continue

                # Сохраняем снимок
                try:
                    os.makedirs(os.path.dirname(state_path), exist_ok=True)
                    with open(state_path, 'w') as f:
                        json.dump(curr_snapshot, f)
                except Exception:
                    pass
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"❌ Ошибка мониторинга позиций: {e}")
    
    async def trading_loop_v4(self):
        """V4.0: Основной торговый цикл с расширенными возможностями"""
        try:
            # Проверяем флаг паузы торговли
            if hasattr(self, '_trading_paused') and self._trading_paused:
                logger.debug("⏸️ Торговля на паузе (используйте /resume в Telegram)")
                return
            
            # 📅 ПРОВЕРКА ВАЖНЫХ СОБЫТИЙ (ФРС, макро-новости)
            # Восстанавливаем базовые значения перед проверкой событий
            self.LEVERAGE = self.LEVERAGE_BASE
            self.POSITION_SIZE = self.POSITION_SIZE_BASE
            
            if self.fed_event_manager:
                risk_adjustments = self.fed_event_manager.get_risk_adjustments()
                
                # Применяем корректировки рисков
                self.LEVERAGE = max(1, int(self.LEVERAGE_BASE * risk_adjustments['leverage_multiplier']))
                self.POSITION_SIZE = self.POSITION_SIZE_BASE * risk_adjustments['position_size_multiplier']
                
                # Логируем предупреждение если режим осторожности
                if risk_adjustments['mode'] != 'NORMAL':
                    logger.warning(f"⚠️ {risk_adjustments['message']}")
                    logger.info(f"📊 Корректировки рисков: Плечо {self.LEVERAGE}x (было {self.LEVERAGE_BASE}x), "
                              f"Размер позиции ${self.POSITION_SIZE:.2f} (было ${self.POSITION_SIZE_BASE:.2f}), "
                              f"MIN_CONFIDENCE +{risk_adjustments['confidence_bonus']:.0f}%")
                    
                    # Отправляем уведомление в Telegram (только при критическом режиме или раз в час)
                    if risk_adjustments['mode'] == 'WAIT' and self.telegram_bot:
                        # Проверяем, не отправляли ли уже сегодня
                        last_fed_alert_key = 'last_fed_alert_time'
                        if not hasattr(self, last_fed_alert_key):
                            setattr(self, last_fed_alert_key, None)
                        
                        # datetime уже импортирован глобально на строке 109
                        now = datetime.now(WARSAW_TZ)
                        last_alert = getattr(self, last_fed_alert_key)
                        
                        if last_alert is None or (now - last_alert).total_seconds() > 3600:  # Раз в час
                            try:
                                await self.telegram_bot.send_message(
                                    chat_id=self.telegram_chat_id,
                                    text=f"⚠️ *РЕЖИМ ОСТОРОЖНОСТИ*\n\n{risk_adjustments['message']}\n\n"
                                         f"📊 *Текущие настройки:*\n"
                                         f"⚙️ Плечо: {self.LEVERAGE}x (базовое: {self.LEVERAGE_BASE}x)\n"
                                         f"💸 Размер позиции: ${self.POSITION_SIZE:.2f} (базовый: ${self.POSITION_SIZE_BASE:.2f})\n"
                                         f"🎯 MIN_CONFIDENCE: +{risk_adjustments['confidence_bonus']:.0f}%"
                                         f"\n\n💡 *Рекомендация:* Лучше дождаться подтверждений перед крупными сделками.",
                                    parse_mode='Markdown'
                                )
                                setattr(self, last_fed_alert_key, now)
                            except Exception as e:
                                logger.debug(f"⚠️ Ошибка отправки уведомления: {e}")
            
            logger.info("🔍 V4.0: Начинаем анализ рынка...")
            
            # V4.0: Анализ рыночных условий
            market_data = await self.analyze_market_trend_v4()
            market_condition = market_data.get('trend', 'neutral').upper()
            self._current_market_condition = market_condition
            
            # V4.0: Умный выбор символов на основе рыночных условий
            symbols = await self.smart_symbol_selection_v4(market_data)
            
            # Статистика анализа
            total_symbols = len(symbols)
            excluded_count = 0
            analyzed_count = 0
            signals_found = 0
            rejected_signals = 0
            
            logger.info(f"🔍 V4.0: Анализируем {total_symbols} символов в условиях рынка {market_condition}")
            
            # Сохраняем рыночные данные для обучения
            if self.data_storage:
                try:
                    # Сохраняем общие рыночные данные
                    for symbol in symbols[:5]:  # Сохраняем данные по топ-5 для анализа
                        try:
                            # Используем оптимизатор для fetch_ticker
                            if self.api_optimizer:
                                ticker = await self.api_optimizer.fetch_with_cache(
                                    'fetch_ticker', symbol, cache_ttl=60
                                )
                            else:
                                ticker = await self.exchange.fetch_ticker(symbol)
                            
                            if not ticker:
                                continue
                            
                            market_data_obj = MarketData(
                                timestamp=datetime.now(WARSAW_TZ).isoformat(),
                                symbol=symbol,
                                timeframe='market_overview',
                                price=ticker.get('last', 0),
                                volume=ticker.get('quoteVolume', 0),
                                rsi=50,  # Будет обновлено при детальном анализе
                                macd=0,
                                bb_position=50,
                                ema_9=0, ema_21=0, ema_50=0,
                                volume_ratio=1.0,
                                momentum=ticker.get('percentage', 0),
                                market_condition=market_condition
                            )
                            self.data_storage.store_market_data(market_data_obj)
                            
                            # 🧠 САМООБУЧЕНИЕ LSTM: Автоматическое обучение и переобучение моделей
                            if self.advanced_ml_system:
                                try:
                                    # Получаем исторические данные для обучения (200 свечей 1h)
                                    df_1h = await self._fetch_ohlcv(symbol, '1h', 200)
                                    if not df_1h.empty and len(df_1h) >= 100:
                                        # Рассчитываем индикаторы для обучения
                                        indicators_dict = self._calculate_indicators(df_1h)
                                        
                                        # Создаем фичи для LSTM
                                        features = self.advanced_ml_system.create_features(df_1h, indicators_dict)
                                        
                                        if not features.empty and len(features) >= 50:
                                            # Создаем целевые переменные
                                            targets = self.advanced_ml_system.create_targets(
                                                df_1h, 
                                                prediction_horizons=[1, 3, 6, 12]
                                            )
                                            
                                            # Проверяем, нужно ли обучить модель впервые
                                            if symbol not in self.advanced_ml_system.price_prediction_models:
                                                logger.info(f"🧠 {symbol}: Начинаем автоматическое обучение LSTM модели (первый раз)")
                                                result = self.advanced_ml_system.train_price_prediction_model(
                                                    symbol, features,
                                                    {'price_target_1h': targets.get('price_target_1h', pd.Series())}
                                                )
                                                if result:
                                                    logger.info(f"✅ {symbol}: LSTM модель обучена | "
                                                               f"Лучшая модель: {result.get('best_model', 'N/A')}")
                                                    
                                                    # Сохраняем модель после первого обучения
                                                    try:
                                                        import os
                                                        bot_dir = "/opt/bot" if os.path.exists("/opt/bot") else os.path.dirname(os.path.abspath(__file__))
                                                        models_dir = os.path.join(bot_dir, "data", "models")
                                                        os.makedirs(models_dir, exist_ok=True)
                                                        model_path = os.path.join(models_dir, f"{symbol}_lstm_model.pkl")
                                                        self.advanced_ml_system.save_models(model_path)
                                                        logger.info(f"💾 {symbol}: LSTM модель сохранена: {model_path}")
                                                    except Exception as e:
                                                        logger.debug(f"⚠️ Ошибка сохранения LSTM модели: {e}")
                                            
                                            # Периодическое переобучение (каждые 100 циклов)
                                            if not hasattr(self, '_lstm_retrain_counter'):
                                                self._lstm_retrain_counter = {}
                                            if symbol not in self._lstm_retrain_counter:
                                                self._lstm_retrain_counter[symbol] = 0
                                            
                                            self._lstm_retrain_counter[symbol] += 1
                                            # Переобучение каждые 100 торговых циклов (примерно 25 часов)
                                            if self._lstm_retrain_counter[symbol] >= 100:
                                                logger.info(f"🔄 {symbol}: Автоматическое переобучение LSTM модели...")
                                                result = self.advanced_ml_system.train_price_prediction_model(
                                                    symbol, features,
                                                    {'price_target_1h': targets.get('price_target_1h', pd.Series())}
                                                )
                                                if result:
                                                    logger.info(f"✅ {symbol}: LSTM модель переобучена | "
                                                               f"Лучшая модель: {result.get('best_model', 'N/A')}")
                                                self._lstm_retrain_counter[symbol] = 0
                                                
                                                # Сохраняем переобученную модель
                                                try:
                                                    import os
                                                    bot_dir = "/opt/bot" if os.path.exists("/opt/bot") else os.path.dirname(os.path.abspath(__file__))
                                                    models_dir = os.path.join(bot_dir, "data", "models")
                                                    model_path = os.path.join(models_dir, f"{symbol}_lstm_model.pkl")
                                                    self.advanced_ml_system.save_models(model_path)
                                                    logger.info(f"💾 {symbol}: Переобученная LSTM модель сохранена")
                                                except Exception as e:
                                                    logger.debug(f"⚠️ Ошибка сохранения переобученной модели: {e}")
                                except Exception as e:
                                    logger.debug(f"⚠️ Ошибка автоматического обучения LSTM для {symbol}: {e}")
                        except Exception as e:
                            logger.debug(f"⚠️ Ошибка сохранения данных {symbol}: {e}")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка сохранения рыночных данных: {e}")
            
            # 🛑 КРИТИЧЕСКАЯ ПРОВЕРКА В НАЧАЛЕ: Максимум открытых позиций
            # Проверяем ОДИН РАЗ в начале цикла, чтобы не тратить время на анализ если лимит достигнут
            current_open_positions = 0
            try:
                current_open_positions = await self._get_current_open_positions_count()
            except Exception as e:
                logger.warning(f"⚠️ Ошибка проверки открытых позиций в начале цикла: {e}")
                current_open_positions = len(self.active_positions)
            
            if current_open_positions >= self.MAX_POSITIONS:
                logger.warning(f"🚫 ЛИМИТ ДОСТИГНУТ! Открытых позиций: {current_open_positions}/{self.MAX_POSITIONS}. Пропускаем весь цикл анализа.")
                return
            
            # КРИТИЧНАЯ ПРОВЕРКА БАЛАНСА в начале цикла
            try:
                balance = await self.exchange.fetch_balance({'accountType': 'UNIFIED'})
                usdt_info = balance.get('USDT', {})
                if isinstance(usdt_info, dict):
                    available_balance = usdt_info.get('free', 0) or usdt_info.get('available', 0) or 0
                    total_balance = usdt_info.get('total', 0) or (usdt_info.get('used', 0) + available_balance)
                else:
                    available_balance = float(usdt_info) if usdt_info else 0
                    total_balance = available_balance
                
                used_margin_start = current_open_positions * self.POSITION_SIZE
                
                # Если баланс меньше минимума для торговли - прекращаем анализ
                if available_balance < self.MIN_BALANCE_FOR_TRADING:
                    logger.error(f"🚫 НЕДОСТАТОЧНО БАЛАНСА ДЛЯ ТОРГОВЛИ!")
                    logger.error(f"   Доступно: ${available_balance:.2f}")
                    logger.error(f"   Минимум требуется: ${self.MIN_BALANCE_FOR_TRADING:.2f} для одной позиции")
                    logger.error(f"   Общий баланс: ${total_balance:.2f}")
                    logger.error(f"   Используется в позициях: ${used_margin_start:.2f} ({current_open_positions} позиций)")
                    logger.error(f"   ⚠️ БОТ НЕ БУДЕТ ОТКРЫВАТЬ НОВЫЕ ПОЗИЦИИ!")
                    return
                
                # Логируем статус баланса
                logger.info(f"💰 Баланс в начале цикла: Доступно: ${available_balance:.2f} | Используется: ${used_margin_start:.2f} ({current_open_positions}/{self.MAX_POSITIONS}) | Общий: ${total_balance:.2f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка проверки баланса в начале цикла: {e}")
                # Продолжаем, но с осторожностью
            
            # Счетчик успешно открытых позиций в этом цикле (контроль через MAX_POSITIONS)
            positions_opened_this_cycle = 0
            
            for symbol in symbols:
                
                # 🚫 ПРОВЕРКА НА ИСКЛЮЧЕННЫЕ СИМВОЛЫ
                if symbol in self.EXCLUDED_SYMBOLS:
                    excluded_count += 1
                    logger.debug(f"🚫 {symbol}: Исключен из анализа")
                    continue
                
                # Пропускаем символы с уже открытыми позициями (проверяем и словарь, и биржу)
                if symbol in self.active_positions:
                    continue
                
                # КРИТИЧНО: Проверяем позиции на бирже перед анализом
                position_exists_on_exchange = False
                try:
                    positions = await self.exchange.fetch_positions([symbol], params={'category': 'linear'})
                    for pos in positions:
                        pos_size = pos.get('contracts', 0) or pos.get('size', 0)
                        if pos_size > 0:
                            logger.info(f"⏸️ {symbol}: Пропущен - уже есть открытая позиция на бирже (размер: {pos_size})")
                            # Добавляем в active_positions для синхронизации
                            self.active_positions[symbol] = {
                                'side': pos.get('side', ''),
                                'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                                'size': pos_size,
                                'pnl_percent': pos.get('percentage', 0)
                            }
                            position_exists_on_exchange = True
                            break
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка проверки позиции на бирже для {symbol}: {e}")
                
                if position_exists_on_exchange:
                    continue  # Переходим к следующему символу
                
                # Пропускаем символы с недавними неудачными попытками открытия (cooldown 30 минут)
                if symbol in self.failed_open_attempts:
                    last_attempt = self.failed_open_attempts[symbol]
                    time_since_attempt = (datetime.now(WARSAW_TZ) - last_attempt).total_seconds() / 60
                    if time_since_attempt < 30:
                        logger.debug(f"⏸️ {symbol}: Пропущен (недавняя неудачная попытка {time_since_attempt:.0f} мин назад)")
                        continue
                    else:
                        # Удаляем старую запись (прошло более 30 минут)
                        del self.failed_open_attempts[symbol]
                
                try:
                    # 🛑 СТРОГАЯ ПРОВЕРКА ПЕРЕД АНАЛИЗОМ: Проверяем позиции перед каждым анализом
                    try:
                        current_open_positions_check = await self._get_current_open_positions_count()
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка проверки позиций для {symbol}: {e}")
                        current_open_positions_check = len(self.active_positions)
                    
                    if current_open_positions_check >= self.MAX_POSITIONS:
                        logger.warning(f"🚫 {symbol}: Пропущен! ЛИМИТ ДОСТИГНУТ ({current_open_positions_check}/{self.MAX_POSITIONS}). Прекращаем анализ.")
                        break
                    
                    analyzed_count += 1
                    
                    # V4.0: Расширенный анализ
                    signal = await self.analyze_symbol_v4(symbol)
                    
                    if signal:
                        signals_found += 1
                        
                        # 🛑 ФИНАЛЬНАЯ ПРОВЕРКА ПЕРЕД ОТПРАВКОЙ: Максимум открытых позиций
                        # Проверяем еще раз перед отправкой сигнала (могут открыться позиции во время анализа)
                        try:
                            final_open_positions = await self._get_current_open_positions_count()
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка финальной проверки позиций: {e}")
                            final_open_positions = len(self.active_positions)
                        
                        if final_open_positions >= self.MAX_POSITIONS:
                            logger.warning(f"🚫 {signal.symbol}: Пропущен! ЛИМИТ ДОСТИГНУТ ПЕРЕД ОТПРАВКОЙ ({final_open_positions}/{self.MAX_POSITIONS})")
                            rejected_signals += 1
                            continue
                        
                        # Логируем проверку
                        logger.info(f"✅ {signal.symbol}: Проверка позиций OK ({final_open_positions}/{self.MAX_POSITIONS}) - открываем позицию")
                        
                        # 🚀 АВТОМАТИЧЕСКОЕ ОТКРЫТИЕ ПОЗИЦИИ
                        position_opened = await self.open_position_automatically(signal)
                        
                        if position_opened:
                            # Отправляем уведомление о открытии позиции
                            await self.send_enhanced_signal_v4(signal)
                            
                            # Увеличиваем счетчик открытых позиций в этом цикле
                            positions_opened_this_cycle += 1
                            
                            # Удаляем из неудачных попыток если была там
                            if signal.symbol in self.failed_open_attempts:
                                del self.failed_open_attempts[signal.symbol]
                            
                            # Проверяем, не достигли ли лимита позиций после открытия
                            try:
                                current_after_open = await self._get_current_open_positions_count()
                                if current_after_open >= self.MAX_POSITIONS:
                                    logger.info(f"✅ Достигнут лимит позиций ({current_after_open}/{self.MAX_POSITIONS}) после открытия {signal.symbol}. Прекращаем анализ.")
                                    break
                            except Exception as e:
                                logger.debug(f"⚠️ Ошибка проверки позиций после открытия: {e}")
                        else:
                            # Записываем неудачную попытку (cooldown 30 минут)
                            self.failed_open_attempts[signal.symbol] = datetime.now(WARSAW_TZ)
                            logger.warning(f"⚠️ {signal.symbol}: Не удалось открыть позицию, добавлен cooldown 30 минут. Сигнал НЕ отправлен.")
                            # НЕ отправляем сигнал, если позиция не открылась
                        
                        # Логируем детальную информацию
                        logger.info(f"🎯 V4.0 СИГНАЛ: {signal.symbol} {signal.direction.upper()} "
                                  f"Цена=${signal.entry_price:.4f} Уверенность={signal.confidence:.0f}% "
                                  f"Оценка={signal.strategy_score:.1f}/20 "
                                  f"Реалистичен={signal.realism_check.is_realistic if signal.realism_check else 'N/A'} "
                                  f"Позиций: {current_open_positions}/{self.MAX_POSITIONS}")
                        
                        # Сохраняем сигнал
                        self.last_signals[symbol] = {
                            'signal': signal,
                            'timestamp': datetime.now(WARSAW_TZ)
                        }
                    else:
                        # Логируем причину отклонения (будет добавлено в analyze_symbol_v4)
                        pass
                    
                    # Небольшая пауза между анализами
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка анализа {symbol}: {e}")
                    continue
            
            # Детальная статистика цикла
            logger.info(f"✅ V4.0: Цикл завершен | "
                       f"Всего: {total_symbols} | "
                       f"Исключено: {excluded_count} | "
                       f"Проанализировано: {analyzed_count} | "
                       f"Сигналов: {signals_found} | "
                       f"Рынок: {market_condition}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка торгового цикла V4.0: {e}")
    
    async def send_startup_message_v4(self):
        """V4.0: Отправить стартовое сообщение в Telegram (только один раз)"""
        try:
            # Проверяем, не отправляли ли уже стартовое сообщение
            if self.startup_message_sent:
                logger.debug("⏸️ Стартовое сообщение уже было отправлено, пропускаем")
                return
            
            if not self.telegram_bot:
                return
            
            # Получаем актуальный баланс
            try:
                balance = await self.exchange.fetch_balance({'accountType': 'UNIFIED'})
                usdt_info = balance.get('USDT', {})
                usdt_total = usdt_info.get('total') if isinstance(usdt_info, dict) else 0
                usdt_free = usdt_info.get('free') or usdt_total if isinstance(usdt_info, dict) else usdt_total
                active_positions = await self._get_current_open_positions_count()
            except:
                usdt_total = 0
                usdt_free = 0
                active_positions = 0
            
            message = f"""🚀 *БОТ V4.0 PRO — ЗАПУЩЕН!*

📊 *MTF Таймфреймы*
15m ⏩ 30m ⏩ 45m ⭐ ⏩ 1h ⏩ 4h

🎯 *Стратегии*
💹 Тренд + Объём + Bollinger
🎭 Детектор манипуляций

🎯 *TP: +$1 (Гарантированно) + Trailing trail0.5%*
🛑 *SL: -\${self.MAX_STOP_LOSS_USD:.1f} максимум → Trailing*

💰 *Баланс*
💵 Всего: ${usdt_total:.2f}
💸 Свободно: ${usdt_free:.2f}

📈 *Торговля*
⚡ Сделка: $5 x5 = $25
📌 Позиции: {active_positions}/3

⏱️ *Анализ:* каждые 15 мин
📊 *Мониторинг:* каждую минуту
⏰ *Время:* {datetime.now(WARSAW_TZ).strftime('%H:%M:%S %d.%m.%Y')}"""
            
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            # Помечаем что сообщение отправлено
            self.startup_message_sent = True
            
            logger.info("✅ V4.0: Стартовое сообщение отправлено в Telegram")
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки стартового сообщения V4.0: {e}")

    async def send_telegram_v4(self, message: str):
        """V4.0: Отправка сообщения в Telegram"""
        try:
            if not self.telegram_bot:
                return
                
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сообщения V4.0: {e}")

    async def send_position_opened_v4(self, symbol: str, side: str, entry_price: float, 
                                    amount_usdt: float, confidence: float, strategy_score: float):
        """V4.0: Уведомление об открытии позиции"""
        try:
            side_emoji = "🟢" if side == 'buy' else "🔴"
            direction = "LONG" if side == 'buy' else "SHORT"
            
            message = f"""
{side_emoji} **ПОЗИЦИЯ ОТКРЫТА V4.0**

💎 **{symbol}**
📊 {direction} | ${entry_price:.5f}
💰 Размер: ${amount_usdt:.0f} (5x)
🎯 Уверенность: {confidence:.0f}%
🏆 Оценка стратегии: {strategy_score:.1f}/20

🎯 **TP: +$1 (Гарантированно) + Trailing trail0.5%**
🛑 **SL:** -\${self.MAX_STOP_LOSS_USD:.1f} максимум → Trailing
⏰ {datetime.now(WARSAW_TZ).strftime('%H:%M:%S %d.%m.%Y')}
"""
            
            await self.send_telegram_v4(message)
            logger.info(f"✅ V4.0: Уведомление об открытии {symbol} отправлено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка уведомления об открытии V4.0: {e}")

    async def send_position_closed_v4(self, symbol: str, side: str, entry_price: float, 
                                    exit_price: float, pnl_percent: float, profit_usd: float, 
                                    reason: str):
        """V4.0: Уведомление о закрытии позиции"""
        try:
            result_emoji = "💰" if pnl_percent > 0 else "💸"
            direction = "LONG" if side == 'buy' else "SHORT"
            
            message = f"""
{result_emoji} **ПОЗИЦИЯ ЗАКРЫТА V4.0**

💎 **{symbol}** {direction}
📥 Вход: ${entry_price:.5f}
📤 Выход: ${exit_price:.5f}

💹 **Результат:**
{'+' if pnl_percent > 0 else ''}{pnl_percent:.2f}% | ${'+' if profit_usd > 0 else ''}{profit_usd:.2f}

📋 **Причина:** {reason}
⏰ {datetime.now(WARSAW_TZ).strftime('%H:%M:%S %d.%m.%Y')}
"""
            
            await self.send_telegram_v4(message)
            logger.info(f"✅ V4.0: Уведомление о закрытии {symbol} отправлено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка уведомления о закрытии V4.0: {e}")

    async def send_tp_hit_v4(self, symbol: str, tp_level: int, pnl_percent: float, 
                           profit_usd: float, remaining_percent: float):
        """V4.0: Уведомление о достижении TP"""
        try:
            message = f"""
🎯 **TP{tp_level} ДОСТИГНУТ V4.0**

💎 **{symbol}**
💰 Прибыль: +{pnl_percent:.2f}% (${profit_usd:.2f})
📊 Осталось позиции: {remaining_percent:.0f}%

⏰ {datetime.now(WARSAW_TZ).strftime('%H:%M:%S %d.%m.%Y')}
"""
            
            await self.send_telegram_v4(message)
            logger.info(f"✅ V4.0: Уведомление TP{tp_level} {symbol} отправлено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка уведомления TP V4.0: {e}")

    async def send_daily_report_v4(self):
        """V4.0: Ежедневный отчет в 9:00"""
        try:
            # Получаем баланс (Bybit Unified Account)
            balance = await self.exchange.fetch_balance({'accountType': 'UNIFIED'})
            usdt_info = balance.get('USDT', {})
            # Для Unified Account может быть None в free/used, используем total как основу
            usdt_total = usdt_info.get('total') if isinstance(usdt_info, dict) else 0
            usdt_free = usdt_info.get('free') or usdt_total if isinstance(usdt_info, dict) else usdt_total
            usdt_used = usdt_info.get('used') or 0 if isinstance(usdt_info, dict) else 0
            
            # Статистика производительности
            total_trades = self.performance_stats.get('total_trades', 0)
            winning_trades = self.performance_stats.get('winning_trades', 0)
            total_pnl = self.performance_stats.get('total_pnl', 0.0)
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Активные позиции
            active_positions = len(self.active_positions)
            
            report = f"""
📊 **ЕЖЕДНЕВНЫЙ ОТЧЕТ V4.0 PRO**

💰 **Баланс:**
💵 Всего: ${usdt_total:.2f}
💸 Свободно: ${usdt_free:.2f}
🔒 В торговле: ${usdt_used:.2f}

📈 **Статистика за сегодня:**
🎯 Сделок: {total_trades}
✅ Прибыльных: {winning_trades}
📊 Винрейт: {win_rate:.1f}%
💹 Общий PnL: ${total_pnl:.2f}

🔄 **Активные позиции:** {active_positions}/3

🤖 **Системы V4.0:**
✅ ProbabilityCalculator
✅ StrategyEvaluator  
✅ RealismValidator
✅ AI+ML Adaptive
✅ 5 таймфреймов (15m-4h)
✅ 6 TP уровней

📅 {datetime.now(WARSAW_TZ).strftime('%d.%m.%Y')} | ⏰ {datetime.now(WARSAW_TZ).strftime('%H:%M')}

**Super Bot V4.0 PRO работает стабильно!** 🚀
"""
            
            await self.send_telegram_v4(report)
            logger.info("📊 V4.0: Ежедневный отчёт отправлен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка ежедневного отчёта V4.0: {e}")

    async def run_v4(self):
        """V4.0: Запуск бота с расширенными возможностями"""
        try:
            logger.info("🚀 Запуск SuperBotV4MTF...")
            
            # Инициализация
            await self.initialize()
            
            # Отправляем стартовое сообщение V4.0
            await self.send_startup_message_v4()
            
            # Настройка планировщика
            self.scheduler.add_job(
                self.trading_loop_v4,
                'interval',
                minutes=15,
                id='trading_loop_v4'
            )
            
            # 📊 Мониторинг позиций (каждую минуту)
            self.scheduler.add_job(
                self.monitor_positions,
                'interval',
                minutes=1,
                id='monitor_positions'
            )
            
            # 📊 Ежедневный отчёт V4.0 в 09:00 Warsaw (Europe/Warsaw = UTC+1/+2)
            try:
                from pytz import timezone as tz
                warsaw_tz = tz('Europe/Warsaw')
                
                self.scheduler.add_job(
                    self.send_daily_report_v4,
                    'cron',
                    hour=9,
                    minute=0,
                    timezone=warsaw_tz,
                    id='daily_report_v4'
                )
                logger.info("✅ V4.0: Ежедневный отчет настроен на 09:00")
            except ImportError:
                logger.warning("⚠️ pytz не установлен, ежедневный отчет отключен")
            
            # Запуск планировщика
            self.scheduler.start()
            logger.info("✅ V4.0: Планировщик запущен (анализ: 15мин, отчет: 09:00)")
            
            # Запуск Telegram бота для обработки команд (если есть)
            if self.application:
                # Запускаем polling в фоне
                await self.application.initialize()
                await self.application.start()
                await self.application.updater.start_polling(drop_pending_updates=True)
                logger.info("✅ Telegram бот запущен и готов к командам")
            
            # Запуск интеллектуальных агентов в фоне (если доступны)
            if self.agents_manager:
                agents_task = asyncio.create_task(
                    self.agents_manager.run_periodic_with_learning()
                )
                logger.info("🤖 Интеллектуальные агенты запущены (самообучение каждые 15 мин)")
            
            # Первый запуск торгового цикла
            await self.trading_loop_v4()
            
            # Бесконечный цикл
            while True:
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 V4.0: Остановка по запросу пользователя")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка V4.0: {e}")
        finally:
            # Останавливаем Telegram polling
            if hasattr(self, 'application') and self.application:
                try:
                    await self.application.updater.stop()
                    await self.application.stop()
                    await self.application.shutdown()
                    logger.info("✅ Telegram бот остановлен")
                except Exception as e:
                    logger.debug(f"Ошибка остановки Telegram: {e}")
            
            # Останавливаем планировщик
            if self.scheduler.running:
                self.scheduler.shutdown()
            
            # Останавливаем интеллектуальных агентов
            if self.agents_manager:
                try:
                    self.agents_manager.intelligent_system.running = False
                    # Останавливаем агентов
                    for agent in self.agents_manager.intelligent_system.agents.values():
                        agent.stop() if hasattr(agent, 'stop') else None
                    logger.info("🤖 Интеллектуальные агенты остановлены")
                except Exception as e:
                    logger.debug(f"Ошибка остановки агентов: {e}")
            
            # Очистка кэша API оптимизатора (опционально)
            if self.api_optimizer:
                try:
                    # Очищаем только старый кэш, новый оставляем
                    self.api_optimizer.cache.clear_old_cache(max_age_hours=24)
                    # Выводим статистику оптимизации
                    stats = self.api_optimizer.get_stats()
                    logger.info(f"⚡ API Optimizer статистика: {stats}")
                except Exception as e:
                    logger.debug(f"Ошибка очистки кэша: {e}")
            
            # Закрываем exchange
            if self.exchange:
                try:
                    await self.exchange.close()
                except:
                    pass
            
            logger.info("🏁 V4.0: Бот остановлен")


async def main():
    """Главная функция"""
    bot = SuperBotV4MTF()
    await bot.run_v4()


if __name__ == "__main__":
    asyncio.run(main())
