#!/usr/bin/env python3
"""
🚀 СУПЕР БОТ V4.0 PRO - ENHANCED MULTI-TIMEFRAME STRATEGY
✅ ИЗМЕНЕНО: 4 таймфрейма: 5m ⏩ 15m ⏩ 30m ⏩ 1h
✅ 6 TP уровней с ML вероятностями           [НОВОЕ]
✅ Оценка стратегии 0-20 баллов             [НОВОЕ]
✅ Проверка реалистичности сигналов         [НОВОЕ]
✅ Топ-5 индикаторов для деривативов
✅ AI+ML адаптация + Disco57 обучение
✅ Полная интеграция всех систем
"""

# Импорты для AI+ML системы
import os
import gc  # Для очистки памяти
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
    # Используем print только если logger еще не инициализирован
    # Предупреждение будет выведено позже через logger в __init__
    pass

try:
    from adaptive_trading_system import FullyAdaptiveSystem
    FULLY_ADAPTIVE_AVAILABLE = True
except ImportError as e:
    FULLY_ADAPTIVE_AVAILABLE = False
    # Не логируем здесь, т.к. logger еще не инициализирован
    # Логирование будет в __init__ бота

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

# 🔴 ПРИОРИТЕТ 2.2: Система резервного копирования
try:
    from backup_system import BackupSystem, get_backup_system
    BACKUP_SYSTEM_AVAILABLE = True
except ImportError:
    BACKUP_SYSTEM_AVAILABLE = False
    print("⚠️ Система резервного копирования недоступна")

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

try:
    from high_performance_trading_system import HighPerformanceTradingSystem
    HIGH_PERFORMANCE_AVAILABLE = True
except ImportError:
    HIGH_PERFORMANCE_AVAILABLE = False
    # УДАЛЕНО: Предупреждение о High Performance System (не используется в V5.0 LIGHTNING)
    # print("⚠️ High Performance Trading System недоступна")

# Импорт Disco57 Integration
try:
    from disco57_integration import Disco57Integration
    DISCO57_INTEGRATION_AVAILABLE = True
except ImportError as e:
    DISCO57_INTEGRATION_AVAILABLE = False
    print(f"⚠️ Disco57 Integration недоступна: {e}")

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# Импорт psutil для проверки процессов (опционально) - будет импортирован после logger
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

# Используем RotatingFileHandler для ротации логов
from logging.handlers import RotatingFileHandler

# Настройки ротации: максимум 200MB на файл, 2 файла бэкапа = до 400MB логов (оптимизировано для экономии места)
max_bytes = 200 * 1024 * 1024  # 200 MB (было 500MB)
backup_count = 2  # Храним 2 ротированных файла (было 3)

# Обработчики логов
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=max_bytes,
    backupCount=backup_count,
    encoding='utf-8'
)
file_handler.setLevel(log_level)

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

# Форматтер
formatter = WarsawFormatter("[%(asctime)s][%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Настройка логирования
logging.basicConfig(
    level=log_level,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[file_handler, console_handler]
)
# Применяем Warsaw formatter ко всем handler'ам
for handler in logging.root.handlers:
    handler.setFormatter(WarsawFormatter("[%(asctime)s][%(levelname)s] %(message)s"))

logger = logging.getLogger(__name__)

# Импорт psutil для проверки процессов (опционально)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("⚠️ psutil не установлен. Проверка дубликатов ботов будет ограничена. Установите: pip install psutil")


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
    timeframe_analysis: Dict  # ИЗМЕНЕНО: 5m, 15m, 30m, 1h
    tp_levels: List[EnhancedTakeProfitLevel]
    stop_loss: float
    realism_check: RealismCheck
    ml_probability: float
    market_condition: str
    reasons: List[str]
    # 📊 РАСШИРЕННЫЕ ДАННЫЕ (как в TradeGPT боте)
    strategies: Optional[List[Dict]] = None  # Сгенерированные стратегии с ценами входа/выхода/SL
    volume_1h_vs_3d_ratio: float = 0  # Объем за последний час vs 3-дневное среднее
    volume_analysis_text: str = ""  # Текст анализа объема
    price_change_5m: float = 0  # Изменение цены за последние 5 минут
    price_change_5m_text: str = ""  # Текст анализа 5-минутного движения
    market_sentiment_index: float = 50  # Индекс рыночного настроения (0-100)
    market_sentiment_text: str = ""  # Текст настроения (Страх/Жадность)
    short_term_support: Optional[float] = None  # Краткосрочный уровень поддержки
    short_term_resistance: Optional[float] = None  # Краткосрочный уровень сопротивления


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
    
    def _check_duplicate_bot(self):
        """
        🔴 КРИТИЧЕСКАЯ ПРОВЕРКА: Проверяет, не запущен ли уже другой экземпляр бота.
        Использует lock-файл и проверку процессов.
        """
        bot_dir = '/opt/bot' if os.path.exists('/opt/bot') else os.path.dirname(os.path.abspath(__file__))
        lock_file = os.path.join(bot_dir, '.bot.lock')
        
        # Проверяем lock-файл
        if os.path.exists(lock_file):
            try:
                with open(lock_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Проверяем, существует ли процесс с этим PID
                if PSUTIL_AVAILABLE:
                    if psutil.pid_exists(old_pid):
                        try:
                            proc = psutil.Process(old_pid)
                            cmdline = ' '.join(proc.cmdline())
                            # Проверяем, что это действительно наш бот
                            if 'super_bot_v4_mtf.py' in cmdline:
                                logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: Бот уже запущен (PID {old_pid})!")
                                logger.error(f"   Команда: {cmdline}")
                                raise RuntimeError(f"Бот уже запущен (PID {old_pid}). Остановите старый процесс перед запуском нового.")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Процесс не существует или нет доступа - удаляем старый lock-файл
                            os.remove(lock_file)
                            logger.warning(f"⚠️ Удалён устаревший lock-файл (PID {old_pid} не существует)")
                    else:
                        # Процесс не существует - удаляем старый lock-файл
                        os.remove(lock_file)
                        logger.warning(f"⚠️ Удалён устаревший lock-файл (PID {old_pid} не существует)")
                else:
                    # Если psutil недоступен, проверяем через os.kill (только проверка существования)
                    try:
                        os.kill(old_pid, 0)  # Сигнал 0 не убивает процесс, только проверяет существование
                        # Процесс существует - проверяем через /proc (Linux)
                        if os.path.exists(f'/proc/{old_pid}'):
                            try:
                                with open(f'/proc/{old_pid}/cmdline', 'r') as f:
                                    cmdline = f.read().replace('\x00', ' ')
                                if 'super_bot_v4_mtf.py' in cmdline:
                                    logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: Бот уже запущен (PID {old_pid})!")
                                    logger.error(f"   Команда: {cmdline}")
                                    raise RuntimeError(f"Бот уже запущен (PID {old_pid}). Остановите старый процесс перед запуском нового.")
                            except (IOError, OSError):
                                # Не удалось прочитать - удаляем lock-файл
                                os.remove(lock_file)
                        else:
                            # Процесс не существует - удаляем старый lock-файл
                            os.remove(lock_file)
                            logger.warning(f"⚠️ Удалён устаревший lock-файл (PID {old_pid} не существует)")
                    except (OSError, ProcessLookupError):
                        # Процесс не существует - удаляем старый lock-файл
                        os.remove(lock_file)
                        logger.warning(f"⚠️ Удалён устаревший lock-файл (PID {old_pid} не существует)")
            except (ValueError, IOError) as e:
                logger.warning(f"⚠️ Ошибка чтения lock-файла: {e}. Удаляем и продолжаем.")
                try:
                    os.remove(lock_file)
                except:
                    pass
        
        # Проверяем процессы по имени (если psutil доступен)
        if PSUTIL_AVAILABLE:
            current_pid = os.getpid()
            bot_processes = []
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'super_bot_v4_mtf.py' in cmdline and proc.info['pid'] != current_pid:
                            bot_processes.append((proc.info['pid'], cmdline))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if bot_processes:
                    logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: Найдены другие экземпляры бота:")
                    for pid, cmdline in bot_processes:
                        logger.error(f"   PID {pid}: {cmdline}")
                    raise RuntimeError(f"Найдены другие экземпляры бота: {[p[0] for p in bot_processes]}. Остановите их перед запуском.")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось проверить процессы: {e}. Продолжаем запуск.")
        
        # Создаём lock-файл
        try:
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.info(f"✅ Lock-файл создан: {lock_file} (PID {os.getpid()})")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось создать lock-файл: {e}")
    
    def __init__(self):
        # 🔴 КРИТИЧЕСКАЯ ПРОВЕРКА: Проверяем дубликаты ботов при старте
        self._check_duplicate_bot()
        
        # Проверка доступности TA-Lib (оптимизация - проверяем один раз при инициализации)
        try:
            import talib
            self._talib_available = True
            self._talib = talib
        except ImportError:
            self._talib_available = False
            self._talib = None
            logger.warning("⚠️ TA-Lib не установлен. Установите: pip install TA-Lib и libta-lib0-dev")
        
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
            logger.error("🚨 КРИТИЧЕСКАЯ ОШИБКА: V4.0 модули недоступны! Бот может работать некорректно.")
            # Не останавливаем бот, но предупреждаем
        
        # ⚠️ ОТКЛЮЧЕНО: Продвинутые системы обучения (UniversalLearningSystem, TradingMLSystem, AdvancedMLSystem)
        # ПРИЧИНА: Перегруженность ML систем, конфликты, высокое потребление памяти
        # ОСТАВЛЕНО: Только Disco57 (PPO Agent) для RL обучения на реальных данных
        self.data_storage = None
        self.universal_learning = None
        self.ml_system = None
        self.advanced_ml_system = None
        logger.info("ℹ️ Продвинутые ML системы отключены (упрощение архитектуры)")
            
        if SMART_SELECTOR_AVAILABLE:
            self.smart_selector = SmartCoinSelector()
            logger.info("✅ Умный селектор инициализирован")
        else:
            self.smart_selector = None
            logger.error("🚨 КРИТИЧЕСКАЯ ОШИБКА: SmartCoinSelector недоступен! Бот может работать некорректно.")
            
        if ADAPTIVE_PARAMS_AVAILABLE:
            try:
                self.adaptive_params_system = AdaptiveParameterSystem()
                logger.info("✅ Адаптивные параметры инициализированы")
            except Exception as e:
                self.adaptive_params_system = None
                logger.warning(f"⚠️ Адаптивные параметры недоступны: {e}")
        else:
            self.adaptive_params_system = None
            logger.warning("⚠️ Адаптивные параметры недоступны (модуль не найден)")
        
        # ⚠️ ОТКЛЮЧЕНО: High Performance Trading System
        # ПРИЧИНА: Перегруженность ML систем, не используется в торговых решениях
        self.high_performance_system = None
        logger.info("ℹ️ High Performance Trading System отключена (упрощение архитектуры)")
        
        # 🔗 DISCO57 INTEGRATION (Feature Bus + RL Agent + Shadow Learning)
        if DISCO57_INTEGRATION_AVAILABLE:
            try:
                self.disco57 = Disco57Integration('risk_profile.yml')
                logger.info("✅ Disco57 Integration инициализирована (Feature Bus + Shadow Learning)")
                logger.info(f"   Режим: {self.disco57.mode}")
                # Загружаем параметры из risk_profile.yml
                risk_profile = self.disco57.get_risk_profile()
                self.POSITION_SIZE = risk_profile.position_size_base
                self.LEVERAGE = risk_profile.leverage_base
                self.MAX_POSITIONS = risk_profile.max_positions
                self.STOP_LOSS_PERCENT = risk_profile.stop_loss_percent
                self.MAX_STOP_LOSS_USD = risk_profile.max_stop_loss_usd
                self.TP_LEVELS_V4 = risk_profile.tp_levels
                self.MIN_CONFIDENCE_BASE = risk_profile.min_confidence_base
                self.MIN_CONFIDENCE_FOR_BIG_MOVE = risk_profile.min_confidence_big_move
                logger.info("✅ Параметры загружены из risk_profile.yml")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка инициализации Disco57 Integration: {e}")
                self.disco57 = None
        else:
            self.disco57 = None
            logger.warning("⚠️ Disco57 Integration недоступна")
            
        if FULLY_ADAPTIVE_AVAILABLE:
            try:
                self.fully_adaptive_system = FullyAdaptiveSystem()
                logger.info("✅ Полностью адаптивная система инициализирована")
            except Exception as e:
                self.fully_adaptive_system = None
                logger.warning(f"⚠️ Полностью адаптивная система недоступна: {e}")
        else:
            self.fully_adaptive_system = None
            logger.debug("⚠️ Полностью адаптивная система недоступна (модуль не найден)")
        
        # Инициализация Advanced Indicators
        if ADVANCED_INDICATORS_AVAILABLE:
            self.advanced_indicators = AdvancedIndicators()
            logger.info("🎯 Advanced Indicators (Ichimoku, Fibonacci, S/R) инициализированы")
        else:
            self.advanced_indicators = None
            logger.warning("⚠️ Advanced Indicators недоступны")
        
        # 🔍 Market Trend Validator (проверка направления рынка на основе EMA50/EMA200)
        try:
            from market_trend_validator import MarketTrendValidator
            # Загружаем параметры из risk_profile если доступен
            trend_threshold = 0.5  # Дефолт 0.5%
            allow_flat = True  # Дефолт разрешать FLAT
            if hasattr(self, 'risk_profile') and self.risk_profile:
                # Можно добавить параметры в risk_profile.yml позже
                pass
            self.trend_validator = MarketTrendValidator(
                trend_threshold_percent=trend_threshold,
                allow_flat=allow_flat,
                log_all_signals=True
            )
            logger.info(f"🔍 Market Trend Validator инициализирован | Threshold: {trend_threshold}% | Allow FLAT: {allow_flat}")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Market Trend Validator: {e}")
            self.trend_validator = None
        
        # 🔍 Advanced Trend Detector (расширенный детектор тренда с множеством индикаторов)
        try:
            from advanced_trend_detector import AdvancedTrendDetector
            self.advanced_trend_detector = AdvancedTrendDetector()
            logger.info("🔍 Advanced Trend Detector инициализирован (7 индикаторов)")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Advanced Trend Detector: {e}")
            self.advanced_trend_detector = None
        
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
        self.POSITION_SIZE_BASE = 1.0  # $1 базовая позиция
        self.POSITION_SIZE = 1.0  # Текущая позиция (может меняться)
        self.LEVERAGE_BASE = 20  # 20x плечо базовое
        self.LEVERAGE = 20  # Текущее плечо (может адаптироваться)
        self.MAX_STOP_LOSS_USD = 0.15  # Максимальный убыток $0.15 на сделку (0.75% от позиции)
        self.POSITION_NOTIONAL = 20.0  # $20 позиция (1 * 20x)
        self.STOP_LOSS_PERCENT = 0.75  # 0.75% от позиции (было 0.3%)
        
        # Комиссии Bybit (taker fee)
        self.TAKER_FEE_RATE = 0.0006  # 0.06% на сторону (вход или выход)
        self.TOTAL_FEE_RATE = self.TAKER_FEE_RATE * 2  # 0.12% общая (вход + выход)
        
        self.MAX_POSITIONS = 3
        self.MIN_VOLUME_24H = 1000000  # Минимальный объем 24h
        # Минимальный баланс для торговли: нужно достаточно для одной позиции ($1) + резерв
        self.MIN_BALANCE_FOR_TRADING = 1.0  # Минимум $1 для одной позиции
        self.MIN_BALANCE_FOR_MAX_POSITIONS = 3.0  # Минимум $3 для 3 позиций (3 * $1)
        
        # Единая функция нормализации символов (используется везде)
        # Единая функция нормализации символов (метод класса)
        def normalize_symbol_universal(sym: str) -> str:
            """
            Универсальная нормализация символов для предотвращения дубликатов.
            Обрабатывает все форматы: BTC/USDT, BTCUSDT, BTC/USDT:USDT, BTC:USDT и т.д.
            ИСПРАВЛЕНО: Правильно обрабатывает случаи типа BTCUSDC -> BTCUSDT (не BTCUSDCUSDT)
            """
            if not sym:
                return sym
            # Убираем все разделители
            norm = sym.upper().replace('/', '').replace('-', '').replace(':', '')
            
            # 🔴 ИСПРАВЛЕНИЕ: Обрабатываем случаи с двойными валютами (BTCUSDC, ETHUSDC и т.д.)
            # Если символ содержит USDC, заменяем на USDT (не добавляем USDT к USDC)
            if 'USDC' in norm and not norm.endswith('USDT'):
                # Заменяем USDC на USDT
                norm = norm.replace('USDC', 'USDT')
                # Убираем возможные дубликаты
                while norm.endswith('USDTUSDT'):
                    norm = norm[:-4]
                return norm
            
            # Убираем дублирование USDT в конце
            while norm.endswith('USDTUSDT'):
                norm = norm[:-4]
            
            # Убеждаемся что заканчивается на USDT
            if not norm.endswith('USDT'):
                # Если есть BASE:QUOTE формат, берём BASE и добавляем USDT
                if 'USDT' in norm:
                    # Убираем всё после первого USDT
                    parts = norm.split('USDT', 1)
                    norm = parts[0] + 'USDT'
                else:
                    norm = norm + 'USDT'
            return norm
        
        # Сохраняем функцию как метод класса
        self.normalize_symbol = normalize_symbol_universal
        
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
        
        # Кэш волатильности рынка для адаптивных фильтров
        self._market_volatility_cache = None
        self._market_volatility_cache_time = None
        
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
        # 🎯 ОПТИМИЗАЦИЯ: Баланс между качеством и количеством сигналов
        self.MIN_CONFIDENCE_BASE = 85  # TRADEGPT ЛОГИКА: Качество > Количество (было 75%)
        self.MIN_CONFIDENCE_FOR_BIG_MOVE = 85  # 🚀 Для больших движений (30-90%) требуется 85%+ (было 90%)
        self.MIN_CONFIDENCE = 80  # Начальное значение, будет адаптироваться
        
        # 🚫 ИСКЛЮЧЕННЫЕ СИМВОЛЫ (только самые рискованные мемкоины)
        # Ликвидные мемкоины (DOGE, SHIB, PEPE, FLOKI) теперь РАЗРЕШЕНЫ через SmartCoinSelector
        # Исключаем только малоизвестные/рискованные мемкоины
        self.EXCLUDED_SYMBOLS = [
            # Исключаем только малоизвестные мемкоины, популярные включены
            'BONKUSDT', 'WIFUSDT', 'BOMEUSDT', 'MEMEUSDT', 
            'CATUSDT', 'DOGWIFHATUSDT'  # Только низколиквидные/рискованные
        ]
        
        # V4.0: Расширенные TP уровни (6 уровней) - ОПТИМИЗИРОВАНО ДЛЯ ROE 50-120%
        # 🚀 Фокус на движения 2.5-6% для достижения ROE 50-120% при 20x leverage
        # ROE 50% = движение 2.5%, ROE 80% = 4%, ROE 100% = 5%, ROE 120% = 6%
        # Размер сделки: $1 x20 плечо = $20 позиция
        # Комиссия: 0.12% (0.06% вход + 0.06% выход) = $0.024 на позицию $20
        # ИЗМЕНЕНО: TP1 +1.15% закрывает 100% позиции = $20 * 1.15 * 0.01 = $0.23 - компенсация комиссии, сразу в без убыток
        # TP2 +2.0% закрывает 100% позиции = $20 * 1.0 * 0.02 = $0.40
        # TP3 +3.0% закрывает 100% позиции = $20 * 1.0 * 0.03 = $0.60
        # TP4 +6.0% закрывает 8% позиции = $20 * 0.08 * 0.06 = $0.096 (ROE 120%)
        # ИЗМЕНЕНО: TP уровни TP1=+1.0%, TP2=+2.0%, TP3=+3.0%
        # TP1: +1.0% = +$0.20 (100% позиции) - сразу в без убыток
        # TP2: +2.0% = +$0.40 (100% позиции)
        # TP3: +3.0% = +$0.60 (100% позиции)
        self.TP_LEVELS_V4 = [
            {'level': 1, 'percent': 1.15, 'portion': 1.0},   # +1.15%, 100% позиции - компенсация комиссии, сразу в без убыток
            {'level': 2, 'percent': 2.0, 'portion': 1.0},   # +2.0%, 100% позиции
            {'level': 3, 'percent': 3.0, 'portion': 1.0}    # +3.0%, 100% позиции
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
        
        # НОВОЕ: Словарь для отслеживания убыточных монет (cooldown 12 часов после убытка)
        self.losing_symbols = {}  # {normalized_symbol: (loss_amount, timestamp)}
        
        # 🔴 КРИТИЧНО: Постоянный blacklist проблемных символов (на основе анализа за 7 дней)
        # Символы с 0% Win Rate или критически низким Win Rate
        self.problem_symbols_blacklist = {
            'BRETTUSDT',      # 0% Win Rate, -$1.64
            'BANANAS31USDT',  # 0% Win Rate, -$1.25
            'ZBCNUSDT',       # 0% Win Rate, -$1.08
            'MEWUSDT',        # Критический убыток -$1.80
            'DENTUSDT',       # Критический убыток -$1.27
            'LIGHTUSDT',      # Критический убыток -$1.06
            'HUSDT',          # Критический убыток -$1.05
            '1000BTTUSDT',    # Критический убыток -$1.01
        }
        logger.info(f"🚫 Blacklist проблемных символов инициализирован: {len(self.problem_symbols_blacklist)} символов")
        
        # Флаг отправки стартового сообщения (чтобы не отправлять несколько раз)
        self.startup_message_sent = False
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0
        }
        
        # 🔗 DISCO57: Обучение RL-агента после КАЖДОЙ сделки для быстрого улучшения
        self.rl_training_counter = 0
        self.rl_training_interval = 1  # ИЗМЕНЕНО: Обучение после каждой сделки (было 50)
        
        # 📊 ОТСЛЕЖИВАНИЕ ДНЕВНОЙ ПРОСАДКИ (MAX_DAILY_DRAWDOWN)
        self.daily_pnl_tracker = {}  # {date: {'pnl': float, 'peak': float, 'drawdown': float}}
        self.max_daily_drawdown_percent = 10.0  # Максимальная дневная просадка 10%
        self._trading_paused_due_to_drawdown = False
        
        # 🚨 ПСИХОЛОГИЧЕСКИЙ СТОП-КОНТУР (MAX_CONSECUTIVE_LOSSES)
        self.consecutive_losses = 0  # Счетчик последовательных убытков
        self.max_consecutive_losses = 3  # Пауза после 3 убытков подряд
        self._trading_paused_due_to_losses = False
        self.last_loss_time = None  # Время последнего убытка
        
        # 🚨 ЗАЩИТА ОТ ДУБЛИРОВАНИЯ: Отслеживание отправленных уведомлений о закрытии
        self.sent_close_notifications = {}  # {symbol: timestamp} - для предотвращения дублирования
        
        # 🔴 ПРИОРИТЕТ 2.2: Система резервного копирования
        self.backup_system = None
        self.backup_counter = 0  # Счетчик для периодического резервного копирования
        self.backup_interval = 10  # Резервное копирование каждые 10 закрытых позиций
        if BACKUP_SYSTEM_AVAILABLE:
            try:
                bot_dir = "/opt/bot" if os.path.exists("/opt/bot") else os.path.dirname(os.path.abspath(__file__))
                self.backup_system = get_backup_system(bot_dir=bot_dir)
                logger.info("✅ Система резервного копирования инициализирована")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка инициализации системы резервного копирования: {e}")
        
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
            
            # Нормализуем символы используя единую функцию
            selected_symbols = [self.normalize_symbol(pair[0]) for pair in sorted_pairs[:top_n]]
            
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
            
            # 📊 РАССЧИТЫВАЕМ ВОЛАТИЛЬНОСТЬ РЫНКА (для адаптивных фильтров)
            try:
                # Получаем данные BTC за последние 24 часа для расчета волатильности
                btc_ohlcv = await self._fetch_ohlcv('BTCUSDT', '1h', limit=24)
                if not btc_ohlcv.empty and len(btc_ohlcv) > 0:
                    btc_prices = btc_ohlcv['close'].values
                    if len(btc_prices) > 0:
                        btc_volatility = (btc_prices.max() - btc_prices.min()) / btc_prices.min() * 100
                        # Сохраняем в кэш
                        self._market_volatility_cache = btc_volatility
                        from datetime import datetime
                        self._market_volatility_cache_time = datetime.now()
                        logger.debug(f"📊 Волатильность рынка (BTC 24h): {btc_volatility:.2f}%")
            except Exception as e:
                logger.debug(f"⚠️ Не удалось рассчитать волатильность рынка: {e}")
            
            # Рассчитываем общий тренд
            avg_change = total_change / analyzed_count if analyzed_count > 0 else 0
            
            # 🎯 V5.0 LIGHTNING: Проверка разворота тренда BTC (EMA50 > EMA200)
            btc_trend_reversal = False
            btc_ema50 = 0
            btc_ema200 = 0
            try:
                btc_4h = await self._fetch_ohlcv('BTCUSDT', '4h', limit=200)
                if not btc_4h.empty and len(btc_4h) >= 200:
                    btc_close = btc_4h['close']
                    btc_ema50 = float(btc_close.ewm(span=50, adjust=False).mean().iloc[-1])
                    btc_ema200 = float(btc_close.ewm(span=200, adjust=False).mean().iloc[-1])
                    
                    # Проверяем разворот: EMA50 пересек EMA200 вверх
                    btc_trend_reversal = btc_ema50 > btc_ema200
                    
                    # Сохраняем предыдущее состояние для отслеживания разворота
                    prev_trend = getattr(self, '_prev_btc_trend', None)
                    if prev_trend is None:
                        prev_trend = 'bearish' if btc_ema50 < btc_ema200 else 'bullish'
                    
                    # Если произошел разворот (было BEAR, стало BULL)
                    if prev_trend == 'bearish' and btc_trend_reversal:
                        logger.info(f"🚨 РАЗВОРОТ ТРЕНДА BTC: EMA50 ({btc_ema50:.2f}) > EMA200 ({btc_ema200:.2f})")
                        logger.info(f"📈 BUY WAVE INCOMING! Переключение на BULL режим")
                        
                        # Отправляем уведомление в Telegram
                        if self.telegram_bot:
                            try:
                                await self.send_telegram_v4(
                                    f"🚨 РАЗВОРОТ ТРЕНДА BTC!\n\n"
                                    f"📈 EMA50 ({btc_ema50:.2f}) > EMA200 ({btc_ema200:.2f})\n"
                                    f"💰 Цена BTC: ${btc_price:.2f}\n\n"
                                    f"✅ BUY WAVE INCOMING!\n"
                                    f"🔄 Переключение на BULL режим\n"
                                    f"🎯 Ожидаем BUY сигналы"
                                )
                            except Exception as e:
                                logger.debug(f"⚠️ Ошибка отправки Telegram уведомления о развороте: {e}")
                    
                    # Сохраняем текущее состояние
                    self._prev_btc_trend = 'bullish' if btc_trend_reversal else 'bearish'
            except Exception as e:
                logger.debug(f"⚠️ Ошибка проверки разворота тренда BTC: {e}")
            
            # Определяем тренд
            if rising > falling * 1.5 and avg_change > 1:
                trend = 'bullish'
            elif falling > rising * 1.5 and avg_change < -1:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Если BTC развернулся вверх, принудительно переключаем на BULL
            if btc_trend_reversal and trend != 'bullish':
                logger.info(f"🔄 Принудительное переключение на BULL (разворот BTC)")
                trend = 'bullish'
            
            # Рассчитываем score рынка
            market_score = (rising - falling) * 10 + avg_change * 2
            
            market_data = {
                'trend': trend,
                'btc_change': btc_change,
                'btc_price': btc_price,
                'btc_ema50': btc_ema50,
                'btc_ema200': btc_ema200,
                'btc_trend_reversal': btc_trend_reversal,
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
            if btc_ema50 > 0 and btc_ema200 > 0:
                logger.info(f"📈 BTC EMA50: {btc_ema50:.2f} | EMA200: {btc_ema200:.2f} | Разворот: {'✅' if btc_trend_reversal else '❌'}")
            
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
            
            # Определяем условие рынка для умного селектора
            condition_for_selector = market_condition.lower()
            if condition_for_selector == 'neutral':
                condition_for_selector = 'normal'
            
            # Определяем целевое количество на основе рыночных условий (100-200 монет)
            if market_condition == 'bullish':
                target_count = 200
            elif market_condition == 'bearish':
                target_count = 150
            elif market_condition == 'volatile':
                target_count = 175
            else:  # neutral
                target_count = 150
            
            if self.smart_selector:
                # Используем умный селектор если доступен
                try:
                    symbols = await self.smart_selector.get_smart_symbols(self.exchange, condition_for_selector)
                    # Требуем минимум 100 монет (а не 50), чтобы соответствовать требованиям
                    if symbols and len(symbols) >= 100:
                        logger.info(f"✅ Умный селектор выбрал {len(symbols)} символов (целевое: {target_count}, топ-50 гарантированы)")
                        # Если умный селектор вернул меньше целевого количества, используем fallback для дополнения
                        if len(symbols) < target_count:
                            logger.info(f"📊 Дополняем список до {target_count} монет через fallback...")
                            try:
                                additional = await self.get_top_symbols_v4(target_count - len(symbols))
                                existing_set = set(symbols)
                                for sym in additional:
                                    if sym not in existing_set:
                                        symbols.append(sym)
                                        existing_set.add(sym)
                                        if len(symbols) >= target_count:
                                            break
                                symbols = symbols[:target_count]
                            except:
                                pass
                        return symbols[:target_count]
                    else:
                        logger.warning(f"⚠️ Умный селектор вернул мало символов ({len(symbols) if symbols else 0} < 100), используем fallback")
                except Exception as e:
                    logger.error(f"❌ Ошибка умного селектора: {e}", exc_info=True)
            
            # Fallback: если селектор недоступен или вернул мало монет
            # Используем топ монеты по объему
            try:
                base_symbols = await self.get_top_symbols_v4(200)
                if not base_symbols:
                    base_symbols = []
            except Exception as e:
                logger.error(f"❌ Ошибка получения топ символов: {e}")
                base_symbols = []
            
            # Используем уже определенное целевое количество (100-200 монет)
            selected_count = target_count
            
            # Если fallback символов недостаточно, дополняем до целевого количества
            if len(base_symbols) < selected_count:
                try:
                    # Пытаемся дополнить через повторный вызов селектора или топ монет
                    additional = await self.get_top_symbols_v4(selected_count - len(base_symbols))
                    base_symbols.extend([s for s in additional if s not in base_symbols])
                except:
                    pass
            
            selected_symbols = base_symbols[:selected_count]
            
            # ✅ Гарантируем топ-50 приоритетных монет в начале списка (если еще нет)
            priority_top50 = [
                'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT','LTCUSDT',
                'ATOMUSDT','ETCUSDT','XLMUSDT','NEARUSDT','ICPUSDT','FILUSDT','APTUSDT','ARBUSDT','OPUSDT','SUIUSDT',
                'TIAUSDT','SEIUSDT','TRXUSDT','TONUSDT','AAVEUSDT','UNIUSDT','HBARUSDT','BCHUSDT','MATICUSDT','INJUSDT',
                'ALGOUSDT','VETUSDT','THETAUSDT','FTMUSDT','EGLDUSDT','AXSUSDT','SANDUSDT','MANAUSDT','GALAUSDT','ENJUSDT',
                'DOGEUSDT','SHIBUSDT','PEPEUSDT','1000FLOKIUSDT','BONKUSDT','WIFUSDT','BOMEUSDT','MYROUSDT','POPCATUSDT','MEWUSDT'
            ]
            
            # Вставляем приоритетные монеты в начало списка (на основе индикаторов и логики, без приоритетных SHORT)
            final_symbols = []
            seen = set()
            
            # Добавляем приоритетные монеты
            for symbol in priority_top50:
                if symbol not in seen:
                    final_symbols.append(symbol)
                    seen.add(symbol)
            
            # Добавляем остальные символы
            for symbol in selected_symbols:
                if symbol not in seen:
                    final_symbols.append(symbol)
                    seen.add(symbol)
            
            # Обрезаем до целевого количества
            final_symbols = final_symbols[:selected_count]
            
            logger.info(f"✅ V4.0: Итоговый список {len(final_symbols)} символов из 100-200 (целевое: {target_count}, топ-50 гарантированы) для рынка {market_condition.upper()}")
            
            return final_symbols
            
        except Exception as e:
            logger.error(f"❌ Ошибка умного выбора символов V4.0: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
    
    async def initialize(self):
        """Инициализация соединений"""
        try:
            # 🔴 КРИТИЧЕСКАЯ ПРОВЕРКА: Проверяем доступность критических модулей
            critical_modules_missing = []
            if not V4_MODULES_AVAILABLE:
                critical_modules_missing.append("V4_MODULES (probability_calculator, strategy_evaluator, realism_validator)")
            if not SMART_SELECTOR_AVAILABLE:
                critical_modules_missing.append("SMART_SELECTOR (smart_coin_selector)")
            
            if critical_modules_missing:
                error_msg = f"🚨 КРИТИЧЕСКАЯ ОШИБКА: Отсутствуют критически важные модули:\n" + "\n".join(f"  - {m}" for m in critical_modules_missing)
                logger.error(error_msg)
                raise ImportError(f"Критические модули недоступны: {', '.join(critical_modules_missing)}")
            
            # Предупреждения о некритических модулях
            warnings = []
            if not ML_AVAILABLE:
                warnings.append("AI+ML система (опционально)")
            if not ADVANCED_ML_AVAILABLE:
                warnings.append("Advanced ML System (LSTM) (опционально)")
            if not ADVANCED_INDICATORS_AVAILABLE:
                warnings.append("Advanced Indicators (опционально)")
            
            if warnings:
                logger.warning(f"⚠️ Некритические модули недоступны (бот будет работать, но с ограниченной функциональностью):\n" + "\n".join(f"  - {w}" for w in warnings))
            
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
                
                # Добавляем обработчик ошибок для предотвращения 409 Conflict
                async def error_handler(update, context):
                    """Обработчик ошибок Telegram"""
                    error = context.error
                    error_msg = str(error) if error else ""
                    
                    # Игнорируем 409 Conflict (не критично)
                    if "409" in error_msg or "Conflict" in error_msg or "terminated by other getUpdates" in error_msg.lower():
                        logger.debug(f"⚠️ Telegram 409 Conflict (игнорируется): {error_msg}")
                        return
                    
                    # Логируем другие ошибки
                    logger.error(f"❌ Telegram error: {error}", exc_info=error)
                
                # Регистрируем обработчик ошибок
                self.application.add_error_handler(error_handler)
                
                # Регистрируем команды
                self.commands_handler = TelegramCommandsHandler(self)
                await self.commands_handler.register_commands(self.application)
                
                logger.info("✅ Telegram бот инициализирован с командами")
            else:
                self.application = None
                self.commands_handler = None
            
            logger.info("✅ Все соединения инициализированы")
            
            # ✅ ЗАДАЧА #1: Загрузка позиций с биржи при старте
            await self._load_positions_from_exchange()
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            raise
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Получить OHLCV данные с повторами и exponential backoff. Поддерживает: 5m, 15m, 30m, 1h."""
        # Нормализация символа используя единую функцию
        normalized_symbol = self.normalize_symbol(symbol)

        # 🔄 RETRY МЕХАНИЗМ С EXPONENTIAL BACKOFF
        max_attempts = 3
        base_delay = 1  # Начальная задержка в секундах
        last_err = None
        
        for attempt in range(max_attempts):
            try:
                if self.api_optimizer:
                    ohlcv = await self.api_optimizer.fetch_with_cache(
                        'fetch_ohlcv', normalized_symbol, timeframe, limit, cache_ttl=30
                    )
                    if ohlcv is None:
                        ohlcv = await self.exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
                else:
                    ohlcv = await self.exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
                
                # Проверка на пустые данные (для всех таймфреймов)
                if not ohlcv or len(ohlcv) == 0:
                    logger.warning(f"⚠️ Пустые данные для {symbol} {timeframe}")
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
            except Exception as e:
                last_err = e
                error_str = str(e).lower()
                
                # Проверяем тип ошибки
                is_rate_limit = 'rate limit' in error_str or '429' in error_str or 'too many requests' in error_str
                is_network_error = 'network' in error_str or 'timeout' in error_str or 'connection' in error_str
                is_symbol_error = 'symbol' in error_str and ('invalid' in error_str or 'not found' in error_str)
                
                # Для ошибок символа не повторяем
                if is_symbol_error:
                    logger.debug(f"⚠️ Символ {normalized_symbol} не существует на бирже: {e}")
                    break
                
                # Для rate limit и network ошибок делаем retry
                if (is_rate_limit or is_network_error) and attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"⚠️ Ошибка запроса {normalized_symbol} {timeframe} (попытка {attempt + 1}/{max_attempts}): {e}. Повтор через {delay}с...")
                    await asyncio.sleep(delay)
                else:
                    logger.debug(f"⚠️ Ошибка получения данных {symbol} {timeframe}: {e}")
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
        
        if last_err:
            logger.debug(f"⚠️ Не удалось получить данные {symbol} {timeframe} после {max_attempts} попыток: {last_err}")
        return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Расчет технических индикаторов"""
        if df.empty or len(df) < 21:
            return {}
        
        # Проверяем доступность TA-Lib (проверено при инициализации)
        if not self._talib_available or self._talib is None:
            logger.debug("⚠️ TA-Lib недоступен, пропускаем расчет индикаторов")
            return {}
        
        try:
            talib = self._talib
            
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
            
            # ADX (Average Directional Index) - сила тренда
            # ADX показывает силу тренда (не направление)
            # ADX+ показывает силу восходящего тренда
            # ADX- показывает силу нисходящего тренда
            adx_period = 14
            if len(close) >= adx_period * 2:  # ADX требует достаточно данных
                adx = talib.ADX(high, low, close, timeperiod=adx_period)[-1]
                adx_plus = talib.PLUS_DI(high, low, close, timeperiod=adx_period)[-1]  # +DI
                adx_minus = talib.MINUS_DI(high, low, close, timeperiod=adx_period)[-1]  # -DI
            else:
                adx = 0.0
                adx_plus = 0.0
                adx_minus = 0.0
            
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
                'adx': adx,
                'adx_plus': adx_plus,
                'adx_minus': adx_minus,
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
        """V4.0: Получить данные по 4 таймфреймам"""
        try:
            timeframes = ['5m', '15m', '30m', '1h']  # ✅ ИЗМЕНЕНО: 4 таймфрейма: 5m ⏩ 15m ⏩ 30m ⏩ 1h
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
            # 🚀 V5.0 LIGHTNING: Упрощенная адаптивная уверенность
            base_confidence = self.MIN_CONFIDENCE_BASE
            
            # Адаптация под рыночные условия и направление сделки
            if trade_direction:
                market_upper = market_condition.upper()
                
                # TRADEGPT ЛОГИКА: Качество > Количество (как в примере KITEUSDT)
                if market_upper == 'BEARISH':
                    if trade_direction.lower() == 'sell':  # SHORT в медвежьем рынке
                        adaptive_min_confidence = 82  # TRADEGPT: Повышено до 82% для качества
                        logger.debug(f"🎯 TRADEGPT BEARISH + SHORT: порог {adaptive_min_confidence}%")
                    else:  # LONG в медвежьем рынке
                        adaptive_min_confidence = 85  # TRADEGPT: Против тренда - очень высокий порог
                        logger.debug(f"🎯 TRADEGPT BEARISH + LONG: порог {adaptive_min_confidence}%")
                elif market_upper == 'BULLISH':
                    if trade_direction.lower() == 'buy':  # LONG в бычьем рынке
                        adaptive_min_confidence = 82  # TRADEGPT: Повышено до 82% для качества
                        logger.debug(f"🎯 TRADEGPT BULLISH + LONG: порог {adaptive_min_confidence}%")
                    else:  # SHORT в бычьем рынке
                        adaptive_min_confidence = 85  # TRADEGPT: Против тренда - очень высокий порог
                        logger.debug(f"🎯 TRADEGPT BULLISH + SHORT: порог {adaptive_min_confidence}%")
                else:  # NEUTRAL
                    if trade_direction == 'buy':
                        adaptive_min_confidence = 83  # TRADEGPT: Нейтральный рынок - высокий порог
                        logger.debug(f"🎯 TRADEGPT NEUTRAL + LONG: порог {adaptive_min_confidence}%")
                    else:
                        adaptive_min_confidence = 83  # TRADEGPT: Нейтральный рынок - высокий порог
                        logger.debug(f"🎯 TRADEGPT NEUTRAL + SHORT: порог {adaptive_min_confidence}%")
            else:
                # Если направление еще не определено, используем базовое
                adaptive_min_confidence = base_confidence
            
            # TRADEGPT ЛОГИКА: Диапазон 82-87% (качество > количество)
            adaptive_min_confidence = max(82, min(87, adaptive_min_confidence))
            
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
            
            # ⚠️ ОТКЛЮЧЕНО: TradingMLSystem (упрощение архитектуры)
            # ML предсказания не используются в торговых решениях
            ml_confidence_bonus = 0
            
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
                'min_confidence': adaptive_params.min_confidence,  # ✅ АДАПТИВНЫЙ (70-75% в зависимости от рынка/монеты)
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
                # 🔴 КРИТИЧНО: Диапазон 75-80% для всех случаев (повышено для улучшения WR)!
                base_confidence = 75  # Базовая уверенность 75% (диапазон 75-80%)
            
            return {
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'min_confidence': max(75, min(80, base_confidence)),  # 🔴 КРИТИЧНО: Диапазон 75-80%!
                'ml_confidence_bonus': 0,
                'bb_adjustment': 0,
                'market_condition': market_condition,
                'trade_direction': trade_direction
            }
    
    def _get_bollinger_signal(self, c_30m: Dict) -> Tuple[str, float, List[str]]:
        """V4.0: Получить сигнал Bollinger Reversion с 30m подтверждением"""
        # Рассчитываем BB позицию (0-100%)
        bb_range = c_30m['bb_upper'] - c_30m['bb_lower']
        if bb_range > 0:
            bb_position = (c_30m['price'] - c_30m['bb_lower']) / bb_range * 100
        else:
            bb_position = 50
        
        # BUY: цена в нижней зоне BB (≤25%) + RSI не перекуплен (≤65)
        if (bb_position <= 25 and c_30m['rsi'] <= 65):
            # Дополнительные бонусы
            rsi_bonus = max(0, 65 - c_30m['rsi']) * 0.5  # бонус за низкий RSI
            bb_bonus = max(0, 25 - bb_position) * 0.8     # бонус за близость к границе
            candle_bonus = 5 if c_30m.get('candle_reversal', 0) > 0 else 0
            
            confidence = 55 + rsi_bonus + bb_bonus + candle_bonus
            reasons = [
                'BUY-BB_REVERSION_V4',
                f"BB={bb_position:.0f}%",
                f"RSI={c_30m['rsi']:.0f}",
                f"30m_confirm"  # V4.0: подтверждение 30m
            ]
            if c_30m.get('candle_reversal', 0) > 0:
                reasons.append(f"Candle↗️{c_30m['candle_reversal']:.1f}%")
            return 'buy', min(90, confidence), reasons

        # SELL: цена в верхней зоне BB (≥75%) + RSI не перепродан (≥35)
        elif (bb_position >= 75 and c_30m['rsi'] >= 35):
            # Дополнительные бонусы
            rsi_bonus = max(0, c_30m['rsi'] - 35) * 0.5   # бонус за высокий RSI
            bb_bonus = max(0, bb_position - 75) * 0.8     # бонус за близость к границе
            candle_bonus = 5 if c_30m.get('candle_reversal', 0) < 0 else 0
            
            confidence = 55 + rsi_bonus + bb_bonus + candle_bonus
            reasons = [
                'SELL-BB_REVERSION_V4',
                f"BB={bb_position:.0f}%",
                f"RSI={c_30m['rsi']:.0f}",
                f"30m_confirm"  # V4.0: подтверждение 30m
            ]
            if c_30m.get('candle_reversal', 0) < 0:
                reasons.append(f"Candle↘️{c_30m['candle_reversal']:.1f}%")
            return 'sell', min(90, confidence), reasons

        return None, 0, []
    
    async def analyze_symbol_v4(self, symbol: str) -> Optional[EnhancedSignal]:
        """V4.0: Расширенный анализ символа с новыми возможностями"""
        try:
            # 🚀 СКАЛЬПЕРСКИЕ ФИЛЬТРЫ: Проверка возраста листинга (минимум 7 дней)
            try:
                df_1d = await self._fetch_ohlcv(symbol, '1d', limit=10)
                if len(df_1d) < 7:
                    logger.debug(f"⏸️ {symbol}: Листинг младше 7 дней - пропуск")
                    return None
            except:
                pass  # Продолжаем если не удалось проверить
            
            # Получаем данные по 4 таймфреймам: 5m, 15m, 30m, 1h
            mtf_data = await self._fetch_multi_timeframe_data(symbol)
            if len(mtf_data) < 4:  # Минимум 4 таймфрейма (5m, 15m, 30m, 1h)
                return None
            
            current_5m = mtf_data.get('5m', {})  # ИЗМЕНЕНО: 5m таймфрейм
            current_15m = mtf_data.get('15m', {})  # ИЗМЕНЕНО: 15m таймфрейм
            current_30m = mtf_data.get('30m', {})  # ОСНОВНОЙ таймфрейм для анализа
            current_1h = mtf_data.get('1h', {})
            
            # ОСНОВНОЙ АНАЛИЗ НА 30m - требует наличие данных
            if not all([current_5m, current_15m, current_30m, current_1h]):
                return None
            
            # 🔴 ПРИОРИТЕТ 1.1: ОБРАБОТКА EDGE CASES (None, NaN, 0)
            # Проверка критических значений перед анализом
            import math
            
            # Проверка цены
            price_30m = current_30m.get('price', 0)
            if price_30m is None or price_30m <= 0 or math.isnan(price_30m) or math.isinf(price_30m):
                logger.warning(f"🚫 {symbol}: Цена 30m = None/NaN/Inf/<=0 ({price_30m}), пропускаем")
                return None
            
            # Проверка объема
            volume_30m = current_30m.get('volume', 0)
            if volume_30m is None or volume_30m == 0 or math.isnan(volume_30m) or math.isinf(volume_30m):
                logger.warning(f"🚫 {symbol}: Объем 30m = None/0/NaN/Inf ({volume_30m}), нет ликвидности")
                return None
            
            # Проверка ATR
            atr_30m = current_30m.get('atr', 0)
            if atr_30m is None or atr_30m == 0 or math.isnan(atr_30m) or math.isinf(atr_30m):
                logger.warning(f"🚫 {symbol}: ATR 30m = None/0/NaN/Inf ({atr_30m}), нет волатильности")
                return None
            
            # Проверка RSI
            rsi_30m = current_30m.get('rsi', 50)
            if rsi_30m is None or math.isnan(rsi_30m) or math.isinf(rsi_30m):
                logger.warning(f"🚫 {symbol}: RSI 30m = None/NaN/Inf ({rsi_30m}), пропускаем")
                return None
            
            # Проверка MACD
            macd_30m = current_30m.get('macd', 0)
            if macd_30m is None or math.isnan(macd_30m) or math.isinf(macd_30m):
                logger.warning(f"🚫 {symbol}: MACD 30m = None/NaN/Inf ({macd_30m}), пропускаем")
                return None
            
            # Проверка EMA (критично для анализа)
            ema_9_30m = current_30m.get('ema_9', 0)
            ema_21_30m = current_30m.get('ema_21', 0)
            if ema_9_30m is None or ema_21_30m is None or \
               math.isnan(ema_9_30m) or math.isnan(ema_21_30m) or \
               math.isinf(ema_9_30m) or math.isinf(ema_21_30m):
                logger.warning(f"🚫 {symbol}: EMA 30m = None/NaN/Inf (9={ema_9_30m}, 21={ema_21_30m}), пропускаем")
                return None
            
            # Проверка данных 1h (тоже критично)
            price_1h = current_1h.get('price', 0)
            if price_1h is None or price_1h <= 0 or math.isnan(price_1h) or math.isinf(price_1h):
                logger.warning(f"🚫 {symbol}: Цена 1h = None/NaN/Inf/<=0 ({price_1h}), пропускаем")
                return None
            
            # 🚀 СКАЛЬПЕРСКИЙ ФИЛЬТР #1: RSI спячка на 1h (RSI < 42 хотя бы 10 из 20 свечей)
            try:
                df_1h_full = await self._fetch_ohlcv(symbol, '1h', limit=20)
                if len(df_1h_full) >= 20:
                    # Рассчитываем RSI
                    rsi_1h_series = df_1h_full['close'].ewm(span=14, adjust=False).mean()
                    delta = df_1h_full['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi_1h_values = 100 - (100 / (1 + rs))
                    rsi_below_42_count = (rsi_1h_values < 42).sum()
                    
                    # ✅ ЗАДАЧА #4: Динамическая RSI-спячка в зависимости от рыночного условия
                    market_condition = getattr(self, '_current_market_condition', 'NEUTRAL')
                    
                    if market_condition == 'BULLISH':
                        rsi_threshold = 1  # BULLISH → 0-1 из 20 (ослаблено для бычьего рынка)
                    elif market_condition in ['SIDEWAYS', 'NEUTRAL']:
                        rsi_threshold = 2  # SIDEWAYS → 1-2 из 20
                    else:  # BEARISH
                        rsi_threshold = 2  # BEARISH → 2+ из 20 (можно оставить 2 или увеличить до 3)
                    
                    if rsi_below_42_count < rsi_threshold:
                        logger.debug(f"⏸️ {symbol}: RSI спячка не подтверждена ({rsi_below_42_count}/20 свечей < 42, требуется {rsi_threshold}, рынок: {market_condition})")
                        return None
                    logger.debug(f"✅ {symbol}: RSI спячка подтверждена ({rsi_below_42_count}/20 свечей < 42, порог: {rsi_threshold}, рынок: {market_condition})")
            except Exception as e:
                logger.debug(f"⚠️ {symbol}: Ошибка проверки RSI спячки: {e}")
            
            # 🚀 СКАЛЬПЕРСКИЙ ФИЛЬТР #2: УДАЛЕН (блокировал 99% сигналов)
            # Объемная аномалия была слишком строгой (350% + цена ±1%)
            
            # 🤖 ПОЛУЧАЕМ АДАПТИВНЫЕ ПАРАМЕТРЫ (AI+ML + 1000+ ПАТТЕРНОВ)
            market_condition = getattr(self, '_current_market_condition', 'NEUTRAL')
            
            # 🔄 СТРАТЕГИЯ ДЛЯ БОКОВОГО РЫНКА (SIDEWAYS/NEUTRAL) - заработок в боковике
            is_sideways = market_condition in ['NEUTRAL', 'SIDEWAYS']
            if is_sideways:
                # В боковике используем более агрессивную стратегию: BB Reversion + Range Trading
                # Фокус на быстрых откатах и отскоках
                logger.debug(f"📊 {symbol}: БОКОВОЙ РЫНОК - активирована стратегия заработка в боковике")
                # Для бокового рынка снижаем порог уверенности и делаем упор на манипуляции и BB
                
            # Предварительно определяем потенциальное направление для адаптации порога
            # (будет уточнено позже)
            potential_direction = None
            
            # ИЗМЕНЕНО: Анализ глобального тренда на 1h (вместо 4h)
            global_trend_bullish = current_1h.get('ema_50', 0) > current_1h.get('ema_200', 0)
            global_trend_bearish = current_1h.get('ema_50', 0) < current_1h.get('ema_200', 0)
            
            # 🔴 КРИТИЧНО: Для бокового рынка НЕ разрешаем обе стороны автоматически
            # Требуем подтверждение на всех таймфреймах для обеих сторон
            if is_sideways:
                # В боковом рынке разрешаем обе стороны, но только при подтверждении на всех таймфреймах
                # Это предотвращает открытие позиций против тренда
                logger.debug(f"📊 {symbol}: БОКОВОЙ РЫНОК - требуем подтверждение на всех таймфреймах для обеих сторон")
                # global_trend_bullish и global_trend_bearish остаются как есть (определены выше)
            
            signal = None
            confidence = 0
            reasons = []
            
            # 🚀 V5.0 LIGHTNING: УПРОЩЕННЫЕ ФИЛЬТРЫ (5 условий вместо 10+MTF)
            # Получаем adaptive_params заранее (без направления для начальной проверки)
            temp_adaptive_params = self._get_adaptive_signal_params(market_condition, current_30m, None)
            
            # 📊 РАСШИРЕННЫЙ АНАЛИЗ (как в TradeGPT боте)
            # 1. Объем за последний час vs 3-дневное среднее (72 свечи 1h = 3 дня)
            volume_ok = False
            volume_ratio = 0
            volume_1h_vs_3d_ratio = 0
            volume_analysis_text = ""
            # 2. Краткосрочные уровни поддержки/сопротивления
            short_term_support = None
            short_term_resistance = None
            # 3. Анализ за последние 5 минут
            price_change_5m = 0
            price_change_5m_text = ""
            # 4. Индекс рыночного настроения
            market_sentiment_index = 50
            market_sentiment_text = ""
            
            try:
                # Анализ объема за последний час vs 3-дневное среднее
                df_1h_vol = await self._fetch_ohlcv(symbol, '1h', limit=72)  # 3 дня = 72 свечи
                if len(df_1h_vol) >= 72:
                    # Объем за последний час (последняя свеча)
                    volume_last_hour = df_1h_vol['volume'].iloc[-1]
                    # Средний объем за 3 дня (72 свечи)
                    avg_volume_3d = df_1h_vol['volume'].tail(72).mean()
                    if avg_volume_3d > 0:
                        volume_1h_vs_3d_ratio = volume_last_hour / avg_volume_3d
                        volume_analysis_text = f"Объем за последний час увеличился в {volume_1h_vs_3d_ratio:.2f} раз по сравнению с 3-дневным средним"
                        logger.debug(f"📊 {symbol}: {volume_analysis_text}")
                
                # Краткосрочные уровни поддержки/сопротивления (на основе последних 20 свечей 1h)
                if len(df_1h_vol) >= 20:
                    recent_20 = df_1h_vol.tail(20)
                    current_price = recent_20['close'].iloc[-1]
                    # Находим локальные минимумы (поддержка) и максимумы (сопротивление)
                    support_levels = recent_20['low'].rolling(window=3, center=True).min().dropna()
                    resistance_levels = recent_20['high'].rolling(window=3, center=True).max().dropna()
                    
                    # Ближайшая поддержка (ниже текущей цены)
                    supports_below = [s for s in support_levels if s < current_price]
                    if supports_below:
                        short_term_support = max(supports_below)
                    
                    # Ближайшее сопротивление (выше текущей цены)
                    resistances_above = [r for r in resistance_levels if r > current_price]
                    if resistances_above:
                        short_term_resistance = min(resistances_above)
                    
                    # Если не нашли, используем дефолтные значения
                    if short_term_support is None:
                        short_term_support = current_price * 0.95  # -5% от текущей цены
                    if short_term_resistance is None:
                        short_term_resistance = current_price * 1.05  # +5% от текущей цены
                    
                    logger.debug(f"📊 {symbol}: Краткосрочные уровни | Поддержка: ${short_term_support:.6f} | Сопротивление: ${short_term_resistance:.6f}")
                
                # Анализ за последние 5 минут
                df_5m_recent = await self._fetch_ohlcv(symbol, '5m', limit=2)
                if len(df_5m_recent) >= 2:
                    price_5m_ago = df_5m_recent['close'].iloc[-2]
                    price_current = df_5m_recent['close'].iloc[-1]
                    if price_5m_ago > 0:
                        price_change_5m = ((price_current - price_5m_ago) / price_5m_ago) * 100
                        if price_change_5m > 0:
                            price_change_5m_text = f"В последние 5 минут цена {symbol} сигнализирует о бычьем тренде с увеличением на {price_change_5m:.2f}%"
                        else:
                            price_change_5m_text = f"В последние 5 минут цена {symbol} снижается на {abs(price_change_5m):.2f}%"
                        logger.debug(f"📊 {symbol}: {price_change_5m_text}")
                
                # Индекс рыночного настроения (Fear/Greed)
                try:
                    if self.adaptive_params_system:
                        # Получаем данные для расчета индекса
                        btc_ticker = await self.exchange.fetch_ticker('BTCUSDT')
                        btc_change_24h = btc_ticker.get('percentage', 0) if btc_ticker else 0
                        
                        market_data = {
                            'btc_change_24h': btc_change_24h,
                            'total_volume_24h': current_30m.get('volume', 0) * 48 if current_30m else 0,  # Примерная оценка
                            'avg_volume_7d': current_30m.get('volume', 0) * 48 * 7 if current_30m else 0
                        }
                        
                        market_sentiment_index = self.adaptive_params_system._calculate_fear_greed_index(market_data)
                        
                        if market_sentiment_index < 25:
                            market_sentiment_text = "Экстремальный страх"
                        elif market_sentiment_index < 45:
                            market_sentiment_text = "Страх"
                        elif market_sentiment_index < 55:
                            market_sentiment_text = "Нейтрально"
                        elif market_sentiment_index < 75:
                            market_sentiment_text = "Жадность"
                        else:
                            market_sentiment_text = "Экстремальная жадность"
                        
                        logger.debug(f"📊 {symbol}: Индекс рыночного настроения: {market_sentiment_index:.0f} ({market_sentiment_text})")
                except Exception as e:
                    logger.debug(f"⚠️ {symbol}: Ошибка расчета индекса настроения: {e}")
                
                # Также проверяем объем на 30m для основной логики
                df_30m_vol = await self._fetch_ohlcv(symbol, '30m', limit=20)
                if len(df_30m_vol) >= 20:
                    avg_volume_20 = df_30m_vol['volume'].tail(20).mean()
                    recent_3_volume = df_30m_vol['volume'].tail(3).mean()
                    if avg_volume_20 > 0:
                        volume_ratio = recent_3_volume / avg_volume_20
                        # Адаптивный порог: BEAR/NEUTRAL = 120%, BULLISH = 130% (ОСЛАБЛЕНО)
                        # 🚀 СКАЛЬПЕРСКИЙ РЕЖИМ: сниженные пороги для большего количества сделок
                        volume_threshold = 0.8 if market_condition in ['BEARISH', 'NEUTRAL'] else 0.9  # СКАЛЬПИНГ: было 1.2/1.3, стало 0.8/0.9
                        volume_ok = volume_ratio >= volume_threshold
                        if volume_ok:
                            logger.debug(f"✅ {symbol}: Объём подтвержден ({volume_ratio:.2f}x среднего, порог={volume_threshold:.2f}x, рынок={market_condition})")
                        else:
                            logger.debug(f"⚠️ {symbol}: Объём недостаточен ({volume_ratio:.2f}x < {volume_threshold:.2f}x)")
            except Exception as e:
                logger.debug(f"⚠️ {symbol}: Ошибка проверки объёма: {e}")
            
            # 🟢 BUY СИГНАЛ V5.0 LIGHTNING (5 условий)
            buy_conditions = {
                # 1. Глобальный тренд BULLISH (EMA50 > EMA200 на 1h)
                'global_trend_bullish': global_trend_bullish and market_condition != 'BEARISH',
                
                # 2. ИЗМЕНЕНО: Хотя бы 2 из 3 младших ТФ в бычьем тренде (5m, 15m, 30m)
                'mtf_trend': sum([
                    current_5m.get('ema_9', 0) > current_5m.get('ema_21', 0),
                    current_15m.get('ema_9', 0) > current_15m.get('ema_21', 0),
                    current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0)
                ]) >= 2,
                
                # 3. Цена > EMA21 на 30m или 1h
                'price_above_ema': (current_30m.get('price', 0) > current_30m.get('ema_21', 0)) or \
                                   (current_1h.get('price', 0) > current_1h.get('ema_21', 0)),
                
                # 4. RSI 30m < 70 (не перекуплен)
                'rsi_ok': current_30m.get('rsi', 50) < 70,
                
                # 5. Объём за последние 3 свечи > 120% (BEAR/NEUTRAL) или 130% (BULLISH) от среднего за 20 (ОСЛАБЛЕНО)
                'volume_ok': volume_ok,
            }
            
            # 🔴 SELL СИГНАЛ V5.0 LIGHTNING (5 условий - зеркальные)
            # Для SHORT в BEARISH достаточно 1/3 ТФ + обязательный объём
            mtf_sell_count = sum([
                current_5m.get('ema_9', 0) < current_5m.get('ema_21', 0),
                current_15m.get('ema_9', 0) < current_15m.get('ema_21', 0),
                current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0)
            ])
            # ОСЛАБЛЕНО: для SHORT в BEARISH достаточно 1/3, иначе 2/3
            mtf_sell_required = 1 if (market_condition == 'BEARISH' and volume_ok) else 2
            
            sell_conditions = {
                # 1. ИЗМЕНЕНО: Глобальный тренд BEARISH (EMA50 < EMA200 на 1h)
                'global_trend_bearish': global_trend_bearish and market_condition != 'BULLISH',
                
                # 2. MTF тренд: для SHORT в BEARISH достаточно 1/3 (если объём OK), иначе 2/3
                'mtf_trend': mtf_sell_count >= mtf_sell_required,
                
                # 3. Цена < EMA21 на 30m или 1h
                'price_below_ema': (current_30m.get('price', 0) < current_30m.get('ema_21', 0)) or \
                                   (current_1h.get('price', 0) < current_1h.get('ema_21', 0)),
                
                # 4. RSI 30m > 30 (не перепродан)
                'rsi_ok': current_30m.get('rsi', 50) > 30,
                
                # 5. Объём за последние 3 свечи > 120% (BEAR/NEUTRAL) или 130% (BULLISH) от среднего за 20 (ОСЛАБЛЕНО)
                'volume_ok': volume_ok,
            }
            
            # V5.0 LIGHTNING: buy_normal и sell_conditions теперь упрощены
            buy_normal = buy_conditions
            sell_conditions_old = sell_conditions
            
            # 🎯 ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ: Advanced Indicators (Ichimoku, Fibonacci, S/R)
            advanced_bonus = 0
            advanced_reasons = []
            
            if self.advanced_indicators:
                try:
                    # Получаем данные для расчета Advanced Indicators (используем 30m - ОСНОВНОЙ)
                    df_30m = await self._fetch_ohlcv(symbol, '30m', 100)
                    if not df_30m.empty and len(df_30m) >= 52:
                        advanced_data = self.advanced_indicators.get_all_indicators(df_30m)
                        
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
            
            # 🎭 ПРИОРИТЕТ #1: ДЕТЕКТОР МАНИПУЛЯЦИЙ (для быстрого заработка)
            manipulation_signal = None
            try:
                # Используем 30m таймфрейм (Bybit поддерживает: 30m, 1h, 4h, 1D)
                df_30m = await self._fetch_ohlcv(symbol, '30m', 100)
                if not df_30m.empty and len(df_30m) > 20:
                    manipulation_result = ManipulationDetector.detect_manipulation(df_30m, current_30m)
                    if manipulation_result:
                        manipulation_signal = manipulation_result.get('signal')
                        manipulation_type = manipulation_result.get('type', 'UNKNOWN')
                        manipulation_confidence = manipulation_result.get('confidence', 70)
                        manipulation_reason = manipulation_result.get('reason', f'Манипуляция {manipulation_type}')
                        tp_multiplier = manipulation_result.get('tp_multiplier', 0.7)
                        
                        logger.info(f"🎭 {symbol}: ДЕТЕКТ МАНИПУЛЯЦИИ! {manipulation_type} → {manipulation_signal.upper()} | Уверенность: {manipulation_confidence}%")
                        logger.info(f"   Причина: {manipulation_reason}")
                        
                        # ИЗМЕНЕНО: Манипуляции имеют приоритет, НО ВСЕ РАВНО требуют MTF подтверждения 5m+15m+30m+1h
                        # Проверяем MTF подтверждение ДО установки сигнала
                        potential_signal = manipulation_signal
                        potential_direction = potential_signal
                        
                        # КРИТИЧНО: Манипуляции тоже должны проходить MTF проверку
                        # ИЗМЕНЕНО: Временная проверка MTF для манипуляций (5m, 15m, 30m, 1h)
                        if potential_signal == 'buy':
                            mtf_ok = (current_5m.get('ema_9', 0) > current_5m.get('ema_21', 0) and
                                     current_15m.get('ema_9', 0) > current_15m.get('ema_21', 0) and
                                     current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0))
                        elif potential_signal == 'sell':
                            mtf_ok = (current_5m.get('ema_9', 0) < current_5m.get('ema_21', 0) and
                                     current_15m.get('ema_9', 0) < current_15m.get('ema_21', 0) and
                                     current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0))
                        else:
                            mtf_ok = False
                        
                        if mtf_ok:
                            # Только если MTF подтверждено - используем сигнал манипуляции
                            signal = manipulation_signal
                            confidence = manipulation_confidence
                            reasons = [f'🎭{manipulation_type}', manipulation_reason]
                            # 🔴 КРИТИЧНО: Для манипуляций тоже требуем минимум 85%!
                            # Получаем адаптивные параметры для проверки min_confidence
                            market_condition = getattr(self, '_current_market_condition', 'NEUTRAL')
                            adaptive_params_dict = self._get_adaptive_signal_params(market_condition, current_30m, potential_signal)
                            adaptive_min_confidence = max(adaptive_params_dict.get('min_confidence', 70), 70)
                            logger.info(f"🎭 {symbol}: ДЕТЕКТ МАНИПУЛЯЦИИ + MTF ПОДТВЕРЖДЕНО → сигнал принят | Уверенность: {manipulation_confidence}%")
                        else:
                            logger.warning(f"🎭 {symbol}: ДЕТЕКТ МАНИПУЛЯЦИИ, НО MTF НЕ ПОДТВЕРЖДЕНО → сигнал ОТКЛОНЕН")
                            # НЕ устанавливаем signal - он останется None и будет отклонен
                            potential_direction = potential_signal  # Сохраняем для адаптации параметров
            except Exception as e:
                logger.debug(f"⚠️ Ошибка детекции манипуляций для {symbol}: {e}")
            
            # Проверяем Bollinger Reversion с 30m подтверждением (если манипуляций нет)
            if not signal and current_30m:
                bb_signal, bb_confidence, bb_reasons = self._get_bollinger_signal(current_30m)
                if bb_signal:
                    signal = bb_signal
                    confidence = bb_confidence
                    reasons = bb_reasons
                    potential_direction = signal  # Определили направление
            
            # 💡 TRADEGPT SIGNALS: Анализ сигналов TradeGPT (разворот формы, рост объема, MACD разворот)
            if not signal and current_30m:
                try:
                    # Получаем данные за последние 6 часов (12 свечей 30m) для анализа объема
                    df_30m_6h = await self._fetch_ohlcv(symbol, '30m', 12)
                    df_30m_full = await self._fetch_ohlcv(symbol, '30m', 50)  # Для MACD разворота нужны предыдущие значения
                    
                    if not df_30m_6h.empty and len(df_30m_6h) >= 12 and not df_30m_full.empty and len(df_30m_full) >= 26:
                        # 1. TRADEGPT ЛОГИКА: Анализ объема за 6 часов
                        # - Снижение объема = накопление (для BUY)
                        # - Рост объема = импульс (для SELL)
                        volumes_6h = df_30m_6h['volume'].values
                        volume_first_3h = volumes_6h[:6].mean() if len(volumes_6h) >= 6 else 0
                        volume_last_3h = volumes_6h[-6:].mean() if len(volumes_6h) >= 6 else 0
                        volume_avg_6h = volumes_6h.mean()
                        volume_recent = volumes_6h[-3:].mean()  # Последние 3 свечи
                        
                        # Рост объема: вторая половина > первой на 10% (импульс для SELL)
                        volume_increasing = volume_last_3h > volume_first_3h * 1.1
                        
                        # Снижение объема: последние 3 свечи < среднего на 20% (накопление для BUY)
                        volume_decreasing = volume_recent < volume_avg_6h * 0.8
                        
                        # 2. Проверка разворота MACD
                        macd_current = current_30m.get('macd', 0)
                        macd_signal_current = current_30m.get('macd_signal', 0)
                        macd_histogram_current = current_30m.get('macd_histogram', 0)
                        
                        # Получаем предыдущие значения MACD для детекции разворота
                        if len(df_30m_full) >= 26:
                            close_full = df_30m_full['close'].values
                            if self._talib_available and self._talib:
                                talib = self._talib
                                macd_full, macd_signal_full, macd_histogram_full = talib.MACD(close_full, fastperiod=12, slowperiod=26, signalperiod=9)
                                
                                # MACD разворот: MACD пересекает signal или histogram меняет знак
                                macd_reversal_bearish = False
                                macd_reversal_bullish = False
                                
                                if len(macd_full) >= 2 and len(macd_signal_full) >= 2:
                                    # Медвежий разворот: MACD был выше signal, стал ниже (или histogram был положительным, стал отрицательным)
                                    if macd_full[-2] > macd_signal_full[-2] and macd_full[-1] < macd_signal_full[-1]:
                                        macd_reversal_bearish = True
                                    elif len(macd_histogram_full) >= 2 and macd_histogram_full[-2] > 0 and macd_histogram_full[-1] < 0:
                                        macd_reversal_bearish = True
                                    
                                    # Бычий разворот: MACD был ниже signal, стал выше (или histogram был отрицательным, стал положительным)
                                    if macd_full[-2] < macd_signal_full[-2] and macd_full[-1] > macd_signal_full[-1]:
                                        macd_reversal_bullish = True
                                    elif len(macd_histogram_full) >= 2 and macd_histogram_full[-2] < 0 and macd_histogram_full[-1] > 0:
                                        macd_reversal_bullish = True
                        
                        # 3. Проверка разворота формы (pattern reversal)
                        # Разворот формы: цена делает разворот (был рост → падение или наоборот)
                        price_reversal_bearish = False
                        price_reversal_bullish = False
                        
                        if len(df_30m_6h) >= 6:
                            prices_6h = df_30m_6h['close'].values
                            # Медвежий разворот: цена росла, затем начала падать
                            if len(prices_6h) >= 6:
                                price_first_half = prices_6h[:3].mean()
                                price_second_half = prices_6h[3:6].mean()
                                price_current = prices_6h[-1]
                                
                                # Медвежий разворот: цена была выше, затем упала
                                if price_first_half > price_second_half * 1.01 and price_current < price_second_half:
                                    price_reversal_bearish = True
                                
                                # Бычий разворот: цена была ниже, затем выросла
                                if price_first_half < price_second_half * 0.99 and price_current > price_second_half:
                                    price_reversal_bullish = True
                        
                        # 4. Формируем TradeGPT сигнал (АДАПТИРОВАНО ПОД ПРИМЕР KITEUSDT)
                        # TRADEGPT ЛОГИКА:
                        # - BUY: разворот формы + СНИЖЕНИЕ объема (накопление) + MACD разворот вверх
                        # - SELL: разворот формы + РОСТ объема (импульс) + MACD разворот вниз
                        
                        # SHORT сигнал: разворот формы + рост объема + MACD разворот вниз
                        if price_reversal_bearish and volume_increasing and macd_reversal_bearish:
                            signal = 'sell'
                            confidence = 90  # Высокая уверенность для полного TradeGPT сигнала
                            reasons = ['💡TradeGPT-SELL', 'Разворот формы', 'Объем↑6ч (импульс)', 'MACD разворот↓']
                            potential_direction = 'sell'
                            logger.info(f"💡 {symbol}: TRADEGPT SELL сигнал! Разворот формы + Объем↑ (импульс) + MACD разворот↓ | Уверенность: {confidence}%")
                        
                        # LONG сигнал: разворот формы + СНИЖЕНИЕ объема (накопление) + MACD разворот вверх
                        elif price_reversal_bullish and volume_decreasing and macd_reversal_bullish:
                            signal = 'buy'
                            confidence = 90  # Высокая уверенность для полного TradeGPT сигнала
                            reasons = ['💡TradeGPT-BUY', 'Разворот формы', 'Объем↓6ч (накопление)', 'MACD разворот↑']
                            potential_direction = 'buy'
                            logger.info(f"💡 {symbol}: TRADEGPT BUY сигнал! Разворот формы + Объем↓ (накопление) + MACD разворот↑ | Уверенность: {confidence}%")
                        
                        # Частичные сигналы (2 из 3 условий) - только для высокого качества
                        elif price_reversal_bearish and (volume_increasing or macd_reversal_bearish):
                            signal = 'sell'
                            confidence = 85  # Высокая уверенность для частичного сигнала
                            reasons = ['💡TradeGPT-SELL', 'Разворот формы', 'Объем↑' if volume_increasing else 'MACD разворот↓']
                            potential_direction = 'sell'
                            logger.info(f"💡 {symbol}: TRADEGPT SELL (частичный): Разворот формы + {'Объем↑' if volume_increasing else 'MACD разворот↓'} | Уверенность: {confidence}%")
                        
                        elif price_reversal_bullish and (volume_decreasing or macd_reversal_bullish):
                            signal = 'buy'
                            confidence = 85  # Высокая уверенность для частичного сигнала
                            reasons = ['💡TradeGPT-BUY', 'Разворот формы', 'Объем↓' if volume_decreasing else 'MACD разворот↑']
                            potential_direction = 'buy'
                            logger.info(f"💡 {symbol}: TRADEGPT BUY (частичный): Разворот формы + {'Объем↓' if volume_decreasing else 'MACD разворот↑'} | Уверенность: {confidence}%")
                
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка анализа TradeGPT сигналов для {symbol}: {e}")
            
            # 🚀 V5.0 LIGHTNING: Упрощенная проверка условий (5 вместо 10)
            if not signal:
                buy_count = sum(buy_conditions.values())
                sell_count = sum(sell_conditions.values())
                
                # V5.0 LIGHTNING: Система оценки (5 условий)
                if buy_count == 5:
                    signal = 'buy'
                    potential_direction = 'buy'
                    confidence = 90  # Максимальная уверенность для 5/5
                    reasons = ['BUY-V5_LIGHTNING', f'Conditions:{buy_count}/5']
                elif buy_count == 4:
                    signal = 'buy'
                    potential_direction = 'buy'
                    confidence = 85  # Высокая уверенность для 4/5
                    reasons = ['BUY-V5_LIGHTNING', f'Conditions:{buy_count}/5']
                elif buy_count == 3:
                    # 3/5 - разрешено (ослаблено с 4/5 для увеличения количества сигналов)
                    signal = 'buy'
                    potential_direction = 'buy'
                    confidence = 80  # Средняя уверенность для 3/5
                    reasons = ['BUY-V5_LIGHTNING', f'Conditions:{buy_count}/5']
                elif sell_count == 5:
                    signal = 'sell'
                    potential_direction = 'sell'
                    confidence = 90  # Максимальная уверенность для 5/5
                    reasons = ['SELL-V5_LIGHTNING', f'Conditions:{sell_count}/5']
                elif sell_count == 4:
                    signal = 'sell'
                    potential_direction = 'sell'
                    confidence = 85  # Высокая уверенность для 4/5
                    reasons = ['SELL-V5_LIGHTNING', f'Conditions:{sell_count}/5']
                elif sell_count == 3:
                    # 3/5 - разрешено (ослаблено с 4/5 для увеличения количества сигналов)
                    signal = 'sell'
                    potential_direction = 'sell'
                    confidence = 80  # Средняя уверенность для 3/5
                    reasons = ['SELL-V5_LIGHTNING', f'Conditions:{sell_count}/5']
            
            # ⚠️ ОТКЛЮЧЕНО: HIGH PERFORMANCE и AdvancedMLSystem
            # ПРИЧИНА: Упрощение архитектуры, снижение потребления памяти
            high_potential_data = None
            ml_big_movement_data = None
            
            # 🤖 ПОЛУЧАЕМ АДАПТИВНЫЕ ПАРАМЕТРЫ С УЧЕТОМ НАПРАВЛЕНИЯ СДЕЛКИ
            # (делаем это после определения направления для правильной адаптации)
            # ОСНОВНОЙ АНАЛИЗ НА 30m
            adaptive_params = self._get_adaptive_signal_params(market_condition, current_30m, potential_direction)
            
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
            
            # 📊 СОБИРАЕМ КАНДИДАТОВ ДЛЯ ДЕТАЛЬНОГО ОТЧЕТА
            # Сохраняем информацию о символах, которые близки к порогу (даже если не прошли)
            if not hasattr(self, 'candidates_list'):
                self.candidates_list = []
            
            candidate_info = {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence if signal else 0,
                'adaptive_min_confidence': adaptive_min_confidence,
                'strategy_score': 0,  # Будет рассчитан позже
                'rsi': current_30m.get('rsi', 0) if current_30m else 0,
                'bb_position': current_30m.get('bb_position', 50) if current_30m else 50,
                'volume_ratio': current_30m.get('volume_ratio', 0) if current_30m else 0,
                'market_condition': market_condition,
                'reasons': reasons if signal else [],
                'entry_price': current_30m.get('price', 0) if current_30m else 0
            }
            
            # Сохраняем кандидатов с уверенностью >= 70% (близкие к порогу)
            if candidate_info['confidence'] >= 70:
                self.candidates_list.append(candidate_info)
            
            # Логируем адаптивный порог для отладки
            logger.debug(
                f"🎯 {symbol}: Адаптивный MIN_CONFIDENCE={adaptive_min_confidence}% | "
                f"Рынок={market_condition} | Направление={signal if signal else 'n/a'}"
            )

            # Детальный срез индикаторов по MTF для отладки
            try:
                logger.debug(
                    f"🔎 {symbol} 30m: EMA9={current_30m.get('ema_9')} EMA21={current_30m.get('ema_21')} "
                    f"RSI={current_30m.get('rsi')} MACD={current_30m.get('macd')} MACDsig={current_30m.get('macd_signal')} "
                    f"BBpos={current_30m.get('bb_position')} ATR={current_30m.get('atr')} VolRatio={current_30m.get('volume_ratio')}"
                )
                logger.debug(
                    f"🔎 {symbol} 1h:  EMA9={current_1h.get('ema_9')} EMA21={current_1h.get('ema_21')} RSI={current_1h.get('rsi')}"
                )
                # ИЗМЕНЕНО: 4h и 1D не используются в текущей версии (только 5m, 15m, 30m, 1h)
                # logger.debug(
                #     f"🔎 {symbol} 4h:  EMA9={current_4h.get('ema_9')} EMA21={current_4h.get('ema_21')} RSI={current_4h.get('rsi')}"
                # )
            except Exception:
                pass
            
            # ИЗМЕНЕНО: V5.0 LIGHTNING: Упрощенная MTF проверка (2 из 3 младших ТФ: 5m, 15m, 30m)
            def _mtf_confirm(dir_: str) -> bool:
                if dir_ == 'buy':
                    # ИЗМЕНЕНО: Проверяем 3 младших ТФ (5m, 15m, 30m) - требуется минимум 2 из 3
                    c5m = current_5m.get('ema_9', 0) > current_5m.get('ema_21', 0)
                    c15m = current_15m.get('ema_9', 0) > current_15m.get('ema_21', 0)
                    c30m = current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0)
                    
                    mtf_count = sum([c5m, c15m, c30m])
                    
                    # ИЗМЕНЕНО: Глобальный тренд на 1h (обязательно)
                    ema50_1h = current_1h.get('ema_50', 0)
                    ema200_1h = current_1h.get('ema_200', 0)
                    global_trend_ok = ema50_1h > ema200_1h
                    
                    # Рыночное условие
                    market_ok = market_condition != 'BEARISH'
                    
                    # V5.0 LIGHTNING: 2 из 3 младших ТФ достаточно
                    result = mtf_count >= 2 and global_trend_ok and market_ok
                    logger.debug(f"✅ MTF V5.0 {symbol} LONG: 5m={c5m} 15m={c15m} 30m={c30m} ({mtf_count}/3) GlobalTrend={global_trend_ok} Market={market_ok}")
                    if not result:
                        logger.debug(f"🚫 {symbol}: MTF V5.0 LONG не пройден: {mtf_count}/3 ТФ, GlobalTrend={global_trend_ok}, Market={market_ok}")
                    return result
                if dir_ == 'sell':
                    # ИЗМЕНЕНО: Проверяем 3 младших ТФ (5m, 15m, 30m) - требуется минимум 2 из 3
                    c5m = current_5m.get('ema_9', 0) < current_5m.get('ema_21', 0)
                    c15m = current_15m.get('ema_9', 0) < current_15m.get('ema_21', 0)
                    c30m = current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0)
                    
                    mtf_count = sum([c5m, c15m, c30m])
                    
                    # ИЗМЕНЕНО: Глобальный тренд на 1h (обязательно)
                    ema50_1h = current_1h.get('ema_50', 0)
                    ema200_1h = current_1h.get('ema_200', 0)
                    global_trend_ok = ema50_1h < ema200_1h
                    
                    # Рыночное условие
                    market_ok = market_condition != 'BULLISH'
                    
                    # V5.0 LIGHTNING: 2 из 3 младших ТФ достаточно
                    result = mtf_count >= 2 and global_trend_ok and market_ok
                    logger.debug(f"✅ MTF V5.0 {symbol} SHORT: 5m={c5m} 15m={c15m} 30m={c30m} ({mtf_count}/3) GlobalTrend={global_trend_ok} Market={market_ok}")
                    if not result:
                        logger.debug(f"🚫 {symbol}: MTF V5.0 SHORT не пройден: {mtf_count}/3 ТФ, GlobalTrend={global_trend_ok}, Market={market_ok}")
                    return result
                return False

            # 🚀 V5.0 LIGHTNING: УДАЛЕНА повторная MTF проверка (уже проверено в buy_conditions/sell_conditions)
            # Дублирование было избыточным и блокировало сигналы

            # 📊 ФИЛЬТР ВОЛАТИЛЬНОСТИ (ATR): исключаем слишком волатильные монеты
            if signal and current_30m:
                atr = current_30m.get('atr', 0)
                price = current_30m.get('price', 0)
                
                if price > 0 and atr > 0:
                    # ATR в процентах от цены
                    atr_percent = (atr / price) * 100
                    
                    # Исключаем слишком волатильные монеты (ATR > 5% от цены)
                    # И слишком мало волатильные (ATR < 0.1% - нет движения)
                    if atr_percent > 5.0:
                        logger.warning(f"🚫 {symbol}: ОТКЛОНЕНО — слишком высокая волатильность (ATR={atr_percent:.2f}% > 5%)")
                        signal = None
                        confidence = 0
                    elif atr_percent < 0.1:
                        logger.debug(f"⚠️ {symbol}: Низкая волатильность (ATR={atr_percent:.2f}% < 0.1%), но продолжаем анализ")
                    else:
                        logger.debug(f"✅ {symbol}: Волатильность в норме (ATR={atr_percent:.2f}%)")
            
            # ДОП. ФИЛЬТРЫ ДЛЯ НОВЫХ СДЕЛОК: УПРОЩЕННАЯ ВЕРСИЯ (только ATR и Volume)
            def _has_min_potential(direction: str) -> bool:
                try:
                    price = float(current_30m.get('price', 0) or 0)
                    atr = float(current_30m.get('atr', 0) or 0)
                    vol_ratio = float(current_30m.get('volume_ratio', 0) or 0)
                    if price <= 0:
                        return False
                    atr_pct = (atr / price) * 100.0
                    
                    # 🚀 СКАЛЬПЕРСКИЙ РЕЖИМ: сниженные пороги для большего количества сделок
                    atr_threshold = 0.5  # СКАЛЬПИНГ: было 0.9%, стало 0.5%
                    vol_threshold = 0.8  # СКАЛЬПИНГ: было 1.1x, стало 0.8x
                    if atr_pct < atr_threshold:
                        logger.debug(f"🚫 {symbol}: Отклонено | ATR30m={atr_pct:.2f}% < {atr_threshold}%")
                        return False
                    if vol_ratio < vol_threshold:
                        logger.debug(f"🚫 {symbol}: Отклонено | VolumeRatio30m={vol_ratio:.2f} < {vol_threshold}x")
                        return False
                    
                    logger.debug(f"✅ {symbol}: Потенциал OK | ATR%={atr_pct:.2f} VolRatio={vol_ratio:.2f} Dir={direction}")
                    return True
                except Exception:
                    return False

            # ОТКЛЮЧЕНО: High Performance фильтр удален по требованию
            
            # ИЗМЕНЕНО: КРИТИЧЕСКАЯ ПРОВЕРКА: Даже если все условия выполнены, повторно проверяем MTF
            # Это гарантирует, что НИ ОДИН сигнал не пройдет без 5m+15m+30m+1h подтверждения
            if signal and confidence >= adaptive_min_confidence and _has_min_potential(signal):
                # 🔍 УЛУЧШЕНИЕ #1: ПОДТВЕРЖДЕНИЕ СИГНАЛА НА СЛЕДУЮЩЕЙ СВЕЧЕ
                # Ждем следующую свечу и проверяем, что цена движется в нужном направлении
                try:
                    # Получаем последние 2 свечи для проверки подтверждения
                    df_30m_confirm = await self._fetch_ohlcv(symbol, '30m', limit=5)
                    if len(df_30m_confirm) >= 2:
                        current_candle = df_30m_confirm.iloc[-1]
                        previous_candle = df_30m_confirm.iloc[-2]
                        
                        # Проверяем подтверждение сигнала
                        signal_confirmed = False
                        if signal == 'buy':
                            # Для BUY: цена должна быть выше предыдущей свечи или закрытие выше открытия
                            price_confirmed = current_candle['close'] > previous_candle['close']
                            candle_confirmed = current_candle['close'] > current_candle['open']  # Бычья свеча
                            signal_confirmed = price_confirmed or candle_confirmed
                        else:  # sell
                            # Для SELL: цена должна быть ниже предыдущей свечи или закрытие ниже открытия
                            price_confirmed = current_candle['close'] < previous_candle['close']
                            candle_confirmed = current_candle['close'] < current_candle['open']  # Медвежья свеча
                            signal_confirmed = price_confirmed or candle_confirmed
                        
                        if not signal_confirmed:
                            logger.warning(f"⏸️ {symbol} {signal.upper()}: Сигнал НЕ подтвержден на следующей свече | "
                                         f"Предыдущая: {previous_candle['close']:.6f}, Текущая: {current_candle['close']:.6f}")
                            return None
                        else:
                            logger.info(f"✅ {symbol} {signal.upper()}: Сигнал подтвержден на следующей свече | "
                                      f"Цена движется в нужном направлении")
                    else:
                        logger.debug(f"⚠️ {symbol}: Недостаточно данных для подтверждения сигнала, пропускаем проверку")
                except Exception as e:
                    logger.debug(f"⚠️ {symbol}: Ошибка подтверждения сигнала: {e}, продолжаем без подтверждения")
                
                # ФИНАЛЬНАЯ ПРОВЕРКА MTF: УДАЛЕНА (уже проверено в buy_conditions/sell_conditions)
                # Дублирование было избыточным
                
                # 🔍 Advanced Trend Detector: ИСПОЛЬЗУЕТСЯ КАК БОНУС (+3-5% к уверенности), а не блокировка
                if self.advanced_trend_detector:
                    try:
                        trend_analysis = self.advanced_trend_detector.analyze_trend(
                            symbol=symbol,
                            mtf_data=mtf_data,
                            timeframe='1h'  # ИЗМЕНЕНО: с 4h на 1h
                        )
                        
                        # Проверяем согласованность направления сигнала с анализом тренда
                        signal_direction_normalized = 'bullish' if signal == 'buy' else 'bearish'
                        
                        if trend_analysis.direction == signal_direction_normalized:
                            # Сигнал совпадает с трендом - добавляем бонус
                            trend_bonus = min(5, max(3, trend_analysis.confidence / 20))  # +3-5% в зависимости от уверенности
                            confidence += trend_bonus
                            logger.info(f"✅ {symbol} {signal.upper()}: Advanced Trend Detector БОНУС +{trend_bonus:.1f}% | "
                                      f"Тренд: {trend_analysis.direction.upper()} | "
                                      f"Уверенность тренда: {trend_analysis.confidence:.1f}%")
                        else:
                            # Сигнал не совпадает - не блокируем, но не даем бонус
                            logger.debug(f"⚠️ {symbol} {signal.upper()}: Advanced Trend Detector не совпадает | "
                                       f"Сигнал: {signal_direction_normalized.upper()}, Тренд: {trend_analysis.direction.upper()} | "
                                       f"Бонус не применен, но сигнал не заблокирован")
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка Advanced Trend Detector для {symbol}: {e}")
                        # В случае ошибки не блокируем сигнал
                
                # 🔍 Market Trend Validator: УДАЛЕН (дублировал проверку глобального тренда в buy_conditions/sell_conditions)
                        # В случае ошибки не блокируем сигнал, но логируем
                
                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Убеждаемся, что таймфреймы согласованы
                # ИЗМЕНЕНО: Ослаблено требование - теперь 3 из 3 таймфреймов (15m, 30m, 1h) вместо всех 4
                if signal == 'buy':
                    # 🔴 КРИТИЧНО: Для LONG проверяем, что 3 из 3 таймфреймов (15m, 30m, 1h) бычьи
                    # И рынок НЕ медвежий!
                    # ИЗМЕНЕНО: Ослаблено требование - теперь 3 из 3 таймфреймов вместо всех 4
                    mtf_count = sum([
                        current_15m.get('ema_9', 0) > current_15m.get('ema_21', 0),
                        current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0),
                        current_1h.get('ema_9', 0) > current_1h.get('ema_21', 0)
                    ])
                    mtf_alignment = (
                        mtf_count >= 3 and  # 3 из 3 таймфреймов (15m, 30m, 1h) должны быть бычьими
                        market_condition != 'BEARISH'  # 🔴 КРИТИЧНО: НЕ разрешаем Buy в медвежьем рынке!
                    )
                else:
                    # 🔴 КРИТИЧНО: Для SHORT проверяем, что 3 из 4 таймфреймов (15m, 30m, 1h) медвежьи
                    # И рынок НЕ бычий!
                    # ИЗМЕНЕНО: Ослаблено требование - теперь 3 из 4 таймфреймов вместо всех 4
                    mtf_count = sum([
                        current_15m.get('ema_9', 0) < current_15m.get('ema_21', 0),
                        current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0),
                        current_1h.get('ema_9', 0) < current_1h.get('ema_21', 0)
                    ])
                    mtf_alignment = (
                        mtf_count >= 3 and  # 3 из 3 таймфреймов (15m, 30m, 1h) должны быть медвежьими
                        market_condition != 'BULLISH'  # 🔴 КРИТИЧНО: НЕ разрешаем Sell в бычьем рынке!
                    )
                
                if not mtf_alignment:
                    logger.warning(f"🚫 {symbol}: Отклонено | Неполное MTF согласование таймфреймов (15m/30m/1h - требуется 3 из 3)")
                    logger.warning(f"   Это предотвращает входы при нестабильном рынке")
                    return None
                
                # 🎯 КРИТИЧЕСКАЯ ПРОВЕРКА: Уверенность достижения TP1 (+1.15%)
                # Бот входит в сделку ТОЛЬКО если уверен, что достигнем минимум TP1
                tp1_confidence = self._calculate_tp1_confidence(
                    symbol, signal, current_30m, mtf_data, market_condition, confidence
                )
                
                # 🎯 ОПТИМИЗАЦИЯ: Баланс между качеством и количеством сигналов
                # TRADEGPT ЛОГИКА: Качество > Количество
                # Целевой Win Rate: 70%+ (как в примере KITEUSDT +7.08%)
                tp1_threshold = 75.0  # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Повышено до 75% для качества сигналов
                if tp1_confidence < tp1_threshold:
                    logger.warning(f"🚫 {symbol} {signal.upper()}: Отклонено | "
                                 f"Недостаточная уверенность достижения TP1 (+1.15%): {tp1_confidence:.1f}% < {tp1_threshold}%")
                    logger.warning(f"   Бот входит в сделку ТОЛЬКО если уверен, что достигнем минимум TP1")
                    return None
                
                logger.info(f"✅ {symbol} {signal.upper()}: Уверенность достижения TP1 (+1.15%): {tp1_confidence:.1f}% >= {tp1_threshold}% | "
                           f"Вход разрешен")
                
                # V5.0: Создаем расширенный сигнал (ОСНОВНОЙ АНАЛИЗ НА 30m)
                # 🚀 АРБИТРАЖ СИГНАЛОВ: Четкая иерархия принятия решений
                # ПРИОРИТЕТ: MTF (основной) → ML (бонус к confidence) → PPO (через Disco57, если доступен)
                # 
                # ЛОГИКА:
                # 1. MTF сигнал - ОСНОВНОЙ (уже проверен выше)
                # 2. ML вероятность - добавляется как бонус к confidence (не блокирует сигнал)
                # 3. PPO сигнал - используется через Disco57 для финальной проверки (если доступен)
                # 4. Финальное решение: MTF + ML бонус + PPO проверка (если все доступны)
                
                # 🚀 Передаем информацию о высоком потенциале (объединяем High Performance и ML данные)
                combined_potential_data = high_potential_data or ml_big_movement_data
                if high_potential_data and ml_big_movement_data:
                    # Объединяем данные от обеих систем
                    combined_potential_data = {
                        'has_potential': True,
                        'potential_percent': (high_potential_data.get('potential_percent', 0) + ml_big_movement_data.get('potential_percent', 0)) / 2,
                        'confidence': (high_potential_data.get('confidence', 0) + ml_big_movement_data.get('confidence', 0)) / 2,
                        'source': 'combined'
                    }
                
                # 🔍 ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: PPO Agent через Disco57 (если доступен)
                # PPO используется как дополнительный фильтр, НЕ как основной сигнал
                if self.disco57 and hasattr(self.disco57, 'get_rl_signal'):
                    try:
                        # Получаем RL сигнал от PPO агента
                        rl_signal_data = self.disco57.get_rl_signal(
                            symbol=symbol,
                            mtf_data=mtf_data,
                            current_price=current_30m['price'],
                            market_condition=market_condition
                        )
                        
                        if rl_signal_data:
                            rl_action = rl_signal_data.get('action', 'HOLD')
                            rl_confidence = rl_signal_data.get('confidence', 0)
                            
                            # Проверяем согласованность RL сигнала с MTF сигналом
                            if rl_action == 'HOLD':
                                logger.debug(f"🤖 {symbol}: PPO Agent рекомендует HOLD (не открывать позицию)")
                                # PPO HOLD не блокирует MTF сигнал, но снижает confidence
                                confidence = max(confidence - 5, self.MIN_CONFIDENCE_BASE)
                                reasons.append('PPO:HOLD')
                            elif (rl_action == 'LONG' and signal == 'buy') or (rl_action == 'SHORT' and signal == 'sell'):
                                # RL сигнал совпадает с MTF - добавляем бонус
                                confidence = min(confidence + 3, 95)
                                reasons.append(f'PPO:{rl_action}')
                                logger.info(f"🤖 {symbol}: PPO Agent подтверждает {signal.upper()} сигнал | Confidence: {rl_confidence:.1f}%")
                            elif (rl_action == 'LONG' and signal == 'sell') or (rl_action == 'SHORT' and signal == 'buy'):
                                # RL сигнал противоречит MTF - снижаем confidence
                                confidence = max(confidence - 10, self.MIN_CONFIDENCE_BASE)
                                reasons.append(f'PPO:CONFLICT({rl_action})')
                                logger.warning(f"⚠️ {symbol}: PPO Agent противоречит {signal.upper()} сигналу | RL: {rl_action} | Confidence снижена")
                    except Exception as e:
                        logger.debug(f"⚠️ {symbol}: Ошибка получения RL сигнала от PPO: {e}")
                        # В случае ошибки не блокируем MTF сигнал
                
                # 📊 ГЕНЕРАЦИЯ КОНКРЕТНЫХ СТРАТЕГИЙ (как в TradeGPT боте)
                strategies = []
                try:
                    strategies = self._generate_trading_strategies(
                        symbol, signal, current_30m['price'], 
                        short_term_support if short_term_support else current_30m['price'] * 0.95,
                        short_term_resistance if short_term_resistance else current_30m['price'] * 1.05,
                        current_30m, market_condition, market_sentiment_index
                    )
                    
                    # Логируем сгенерированные стратегии
                    if strategies:
                        logger.info(f"📊 {symbol}: Сгенерировано {len(strategies)} стратегий:")
                        for i, strategy in enumerate(strategies, 1):
                            logger.info(f"   Стратегия {i}: {strategy.get('direction', 'N/A')} | "
                                      f"Вход: ${strategy.get('entry_price', 0):.6f} | "
                                      f"Цель: ${strategy.get('target_price', 0):.6f} | "
                                      f"SL: ${strategy.get('stop_loss', 0):.6f} | "
                                      f"Обоснование: {strategy.get('rationale', 'N/A')[:50]}")
                except Exception as e:
                    logger.debug(f"⚠️ {symbol}: Ошибка генерации стратегий: {e}")
                
                enhanced_signal = await self._create_enhanced_signal_v4(
                    symbol, signal, current_30m['price'], confidence, reasons,
                    mtf_data, market_condition, high_potential_data=combined_potential_data
                )
                
                # Добавляем информацию о стратегиях в сигнал
                if hasattr(enhanced_signal, '__dict__'):
                    enhanced_signal.strategies = strategies
                    enhanced_signal.volume_1h_vs_3d_ratio = volume_1h_vs_3d_ratio
                    enhanced_signal.volume_analysis_text = volume_analysis_text
                    enhanced_signal.price_change_5m = price_change_5m
                    enhanced_signal.price_change_5m_text = price_change_5m_text
                    enhanced_signal.market_sentiment_index = market_sentiment_index
                    enhanced_signal.market_sentiment_text = market_sentiment_text
                    enhanced_signal.short_term_support = short_term_support
                    enhanced_signal.short_term_resistance = short_term_resistance
                
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
                # Логируем отсутствие сигнала (ОСНОВНОЙ АНАЛИЗ НА 30m)
                logger.debug(f"⚪ {symbol}: Нет сигнала | "
                           f"RSI={current_30m.get('rsi', 0):.0f} | "
                           f"BB={current_30m.get('bb_position', 50):.0f}% | "
                           f"Vol={current_30m.get('volume_ratio', 0):.1f}x | "
                           f"Рынок={market_condition}")
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка анализа {symbol}: {e}")
            return None
    
    def _calculate_tp1_confidence(self, symbol: str, signal: str, current_30m: Dict, 
                                   mtf_data: Dict, market_condition: str, base_confidence: float) -> float:
        """
        🎯 Расчет уверенности достижения TP1 (+1.15%)
        
        Бот входит в сделку ТОЛЬКО если уверен, что достигнем минимум TP1
        
        Args:
            symbol: Торговая пара
            signal: Направление сигнала (buy/sell)
            current_30m: Данные 30m таймфрейма
            mtf_data: Multi-timeframe данные
            market_condition: Состояние рынка
            base_confidence: Базовая уверенность сигнала
            
        Returns:
            float: Уверенность достижения TP1 в процентах (0-100)
        """
        try:
            # Базовая уверенность на основе общей уверенности сигнала
            tp1_confidence = base_confidence * 0.8  # TP1 легче достичь, чем общая уверенность
            
            # 1. Проверка тренда и импульса (30m)
            rsi = current_30m.get('rsi', 50)
            bb_position = current_30m.get('bb_position', 50)
            volume_ratio = current_30m.get('volume_ratio', 1.0)
            atr_percent = current_30m.get('atr_percent', 0)
            
            # 2. Проверка MTF согласования (чем больше таймфреймов согласовано, тем выше уверенность)
            current_5m = mtf_data.get('5m', {})
            current_15m = mtf_data.get('15m', {})
            current_1h = mtf_data.get('1h', {})
            
            mtf_aligned_count = 0
            if signal == 'buy':
                if current_5m.get('ema_9', 0) > current_5m.get('ema_21', 0):
                    mtf_aligned_count += 1
                if current_15m.get('ema_9', 0) > current_15m.get('ema_21', 0):
                    mtf_aligned_count += 1
                if current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0):
                    mtf_aligned_count += 1
                if current_1h.get('ema_9', 0) > current_1h.get('ema_21', 0):
                    mtf_aligned_count += 1
            else:  # sell
                if current_5m.get('ema_9', 0) < current_5m.get('ema_21', 0):
                    mtf_aligned_count += 1
                if current_15m.get('ema_9', 0) < current_15m.get('ema_21', 0):
                    mtf_aligned_count += 1
                if current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0):
                    mtf_aligned_count += 1
                if current_1h.get('ema_9', 0) < current_1h.get('ema_21', 0):
                    mtf_aligned_count += 1
            
            # Бонус за MTF согласование (каждый таймфрейм +5%)
            mtf_bonus = mtf_aligned_count * 5
            tp1_confidence += mtf_bonus
            
            # 3. Проверка силы тренда
            if signal == 'buy':
                # Для BUY: RSI должен быть выше 50, но не перекуплен (>80)
                if 50 < rsi < 80:
                    tp1_confidence += 10
                elif rsi >= 80:
                    tp1_confidence -= 15  # Перекупленность снижает вероятность
                # BB Position должна быть в нижней части (возможность роста)
                if bb_position < 30:
                    tp1_confidence += 10
                elif bb_position > 70:
                    tp1_confidence -= 10  # Перекупленность
            else:  # sell
                # Для SELL: RSI должен быть ниже 50, но не перепродан (<20)
                if 20 < rsi < 50:
                    tp1_confidence += 10
                elif rsi <= 20:
                    tp1_confidence -= 15  # Перепроданность снижает вероятность
                # BB Position должна быть в верхней части (возможность падения)
                if bb_position > 70:
                    tp1_confidence += 10
                elif bb_position < 30:
                    tp1_confidence -= 10  # Перепроданность
            
            # 4. Проверка объема (высокий объем = больше вероятность движения)
            if volume_ratio >= 1.5:
                tp1_confidence += 8
            elif volume_ratio >= 1.2:
                tp1_confidence += 5
            elif volume_ratio < 0.8:
                tp1_confidence -= 10  # Низкий объем снижает вероятность
            
            # 5. Проверка волатильности (ATR)
            # Если ATR достаточен для движения 1%, это хорошо
            if atr_percent >= 1.0:
                tp1_confidence += 5
            elif atr_percent < 0.5:
                tp1_confidence -= 10  # Слишком низкая волатильность
            
            # 6. Проверка состояния рынка
            if signal == 'buy' and market_condition == 'BULLISH':
                tp1_confidence += 8
            elif signal == 'sell' and market_condition == 'BEARISH':
                tp1_confidence += 8
            elif signal == 'buy' and market_condition == 'BEARISH':
                tp1_confidence -= 15  # Против тренда
            elif signal == 'sell' and market_condition == 'BULLISH':
                tp1_confidence -= 15  # Против тренда
            
            # Ограничиваем уверенность в разумных пределах
            tp1_confidence = max(0, min(100, tp1_confidence))
            
            logger.debug(f"🎯 {symbol} {signal.upper()}: Уверенность TP1 (+1.15%) = {tp1_confidence:.1f}% | "
                        f"MTF: {mtf_aligned_count}/4, RSI: {rsi:.0f}, BB: {bb_position:.0f}%, Vol: {volume_ratio:.1f}x, ATR: {atr_percent:.2f}%")
            
            return tp1_confidence
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета уверенности TP1 для {symbol}: {e}")
            # В случае ошибки возвращаем консервативное значение
            return 50.0
    
    def _generate_trading_strategies(self, symbol: str, direction: str, current_price: float,
                                     support: float, resistance: float, indicators: Dict,
                                     market_condition: str, sentiment_index: float) -> List[Dict]:
        """
        📊 Генерация конкретных торговых стратегий (как в TradeGPT боте)
        
        Генерирует несколько стратегий с конкретными ценами входа/выхода/SL
        
        Args:
            symbol: Торговая пара
            direction: Направление сигнала (buy/sell)
            current_price: Текущая цена
            support: Уровень поддержки
            resistance: Уровень сопротивления
            indicators: Технические индикаторы
            market_condition: Состояние рынка
            sentiment_index: Индекс рыночного настроения (0-100)
            
        Returns:
            List[Dict]: Список стратегий с ценами входа/выхода/SL
        """
        strategies = []
        
        try:
            rsi = indicators.get('rsi', 50)
            bb_position = indicators.get('bb_position', 50)
            atr = indicators.get('atr', 0)
            atr_percent = (atr / current_price * 100) if current_price > 0 and atr > 0 else 0
            
            # Стратегия 1: Короткая позиция на уровне сопротивления (для SHORT)
            if direction == 'sell' and resistance and resistance > current_price:
                # Вход на уровне сопротивления или немного выше
                entry_price = resistance * 1.001  # +0.1% от сопротивления для гарантии входа
                # Цель: уровень поддержки
                target_price = support if support and support < current_price else current_price * 0.98  # -2% если поддержки нет
                # SL: выше уровня сопротивления
                stop_loss = resistance * 1.005  # +0.5% от сопротивления
                
                rationale = f"Учитывая текущий медвежий тренд и перекупленность по RSI (RSI={rsi:.0f}), открытие короткой позиции на уровне сопротивления может быть выгодным. Если цена достигнет уровня сопротивления (${resistance:.6f}), это может стать хорошей возможностью для входа в SHORT."
                
                strategies.append({
                    'strategy_number': 1,
                    'name': 'Короткая позиция на уровне сопротивления',
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'rationale': rationale,
                    'leverage': 3,  # Как в примере TradeGPT
                    'risk_reward': abs((target_price - entry_price) / (entry_price - stop_loss)) if entry_price > stop_loss else 0
                })
            
            # Стратегия 2: Долгосрочная позиция с учетом коррекции (для LONG)
            if direction == 'buy' and support and support < current_price:
                # Вход ниже уровня поддержки (при откате)
                entry_price = support * 0.998  # -0.2% от поддержки
                # Цель: текущая цена или сопротивление
                target_price = resistance if resistance and resistance > current_price else current_price * 1.02  # +2% если сопротивления нет
                # SL: ниже уровня поддержки
                stop_loss = support * 0.995  # -0.5% от поддержки
                
                rationale = f"Если цена откатится к ${entry_price:.6f}, это может быть хорошей возможностью для входа в LONG, учитывая, что уровень поддержки находится на ${support:.6f}. Это может быть выгодной стратегией для тех, кто верит в восстановление цены."
                
                strategies.append({
                    'strategy_number': 2,
                    'name': 'Долгосрочная позиция с учетом коррекции',
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'rationale': rationale,
                    'leverage': 3,
                    'risk_reward': abs((target_price - entry_price) / (entry_price - stop_loss)) if entry_price > stop_loss else 0
                })
            
            # Стратегия 3: Дневная торговля на основе волатильности
            # Вход при пробое текущей цены
            if direction == 'buy':
                entry_price = current_price * 1.001  # +0.1% для пробоя вверх
                target_price = current_price * 1.02  # +2% краткосрочная цель
                stop_loss = current_price * 0.995  # -0.5% SL
            else:  # sell
                entry_price = current_price * 0.999  # -0.1% для пробоя вниз
                target_price = current_price * 0.98  # -2% краткосрочная цель
                stop_loss = current_price * 1.005  # +0.5% SL
            
            rationale = f"Дневная торговля может быть эффективной в условиях высокой волатильности (ATR={atr_percent:.2f}%). Если цена пробьет уровень ${current_price:.6f}, это может сигнализировать о продолжении {'роста' if direction == 'buy' else 'падения'}, и трейдер может воспользоваться этим движением."
            
            strategies.append({
                'strategy_number': 3,
                'name': 'Дневная торговля на основе волатильности',
                'direction': direction.upper(),
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'rationale': rationale,
                'leverage': 3,
                'risk_reward': abs((target_price - entry_price) / abs(entry_price - stop_loss)) if abs(entry_price - stop_loss) > 0 else 0
            })
            
            # Добавляем информацию о рыночном настроении и технических индикаторах
            for strategy in strategies:
                strategy['market_sentiment'] = {
                    'index': sentiment_index,
                    'condition': market_condition,
                    'rsi': rsi,
                    'rsi_status': 'Перекуплен' if rsi > 70 else 'Перепродан' if rsi < 30 else 'Норма',
                    'bb_position': bb_position,
                    'atr_percent': atr_percent
                }
                
                # Анализ технических индикаторов
                tech_indicators = []
                if rsi > 70:
                    tech_indicators.append('RSI: Перекуплен (может указывать на возможный откат)')
                elif rsi < 30:
                    tech_indicators.append('RSI: Перепродан (может указывать на возможный отскок)')
                
                # KDJ, MACD, EMA BREAK, BOLL - проверяем наличие четких паттернов
                macd = indicators.get('macd', 0)
                macd_signal = indicators.get('macd_signal', 0)
                if abs(macd - macd_signal) < 0.0001:  # MACD близок к signal
                    tech_indicators.append('MACD: Нет четких паттернов')
                else:
                    if macd > macd_signal:
                        tech_indicators.append('MACD: Бычий сигнал')
                    else:
                        tech_indicators.append('MACD: Медвежий сигнал')
                
                strategy['technical_indicators'] = tech_indicators if tech_indicators else ['KDJ, MACD, EMA BREAK, BOLL: Нет четких паттернов']
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка генерации стратегий для {symbol}: {e}")
        
        return strategies
    
    async def _create_enhanced_signal_v4(self, symbol: str, direction: str, entry_price: float,
                                       confidence: float, reasons: List[str], mtf_data: Dict,
                                       market_condition: str, high_potential_data: Dict = None) -> EnhancedSignal:
        """V4.0: Создать расширенный сигнал с новыми возможностями"""
        try:
            # 1. Рассчитываем вероятности TP уровней
            tp_probabilities = []
            if self.probability_calculator:
                market_data = mtf_data.get('30m', {})  # ОСНОВНОЙ АНАЛИЗ НА 30m
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
            market_30m = mtf_data.get('30m', {}) or {}  # ОСНОВНОЙ АНАЛИЗ НА 30m
            price_30m = float(market_30m.get('price', entry_price) or entry_price)
            atr_30m = float(market_30m.get('atr', 0) or 0)
            atr_percent = (atr_30m / price_30m * 100) if price_30m > 0 else 0.0
            # Для крупных активов: лимит = min(фикс.лимит, max(6%, ATR% * 2.5))
            # Для прочих активов: лимит = min(фикс.лимит, max(12%, ATR% * 3.0))
            if symbol.upper() in major_assets_limits:
                dynamic_tp_limit = max(6.0, atr_percent * 2.5)
            else:
                dynamic_tp_limit = max(12.0, atr_percent * 3.0)
            effective_tp_limit = min(max_tp_for_symbol, dynamic_tp_limit)
            logger.info(f"📏 {symbol}: ATR={atr_percent:.2f}% → динамический лимит TP={effective_tp_limit:.1f}% (жесткий={max_tp_for_symbol}%)")
            
            # ИЗМЕНЕНО: TP уровни - каждый закрывает 100% позиции
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
                
                # ИЗМЕНЕНО: Каждый TP закрывает 100% позиции (portion=1.0)
                enhanced_tp = EnhancedTakeProfitLevel(
                    level=tp_config['level'],
                    price=tp_price,
                    percent=tp_config['percent'],
                    probability=probability,
                    confidence_interval=confidence_interval,
                    pnl_percent=tp_config['percent'],
                    close_percent=1.0,  # ИЗМЕНЕНО: 100% позиции закрывается на каждом TP
                    market_condition_factor=1.0
                )
                enhanced_tp_levels.append(enhanced_tp)
            
            # ИЗМЕНЕНО: TP1 переводит в без убыток (break-even)
            # Это означает, что при достижении TP1 SL перемещается на уровень входа
            if enhanced_tp_levels and len(enhanced_tp_levels) > 0:
                tp1 = enhanced_tp_levels[0]
                logger.info(f"✅ {symbol}: TP1={tp1.percent}% закрывает 100% позиции и переводит в без убыток")
            
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
            
            # ОТКЛЮЧЕНО: AdvancedMLSystem удалена по требованию
            ml_probability = confidence / 100.0
            lstm_prediction = None
            
            # 🔗 DISCO57: Сбор признаков через Feature Bus
            disco57_features = None
            if self.disco57:
                try:
                    current_30m = mtf_data.get('30m', {})
                    # Подготовка данных для Feature Bus
                    market_data = {
                        'price': entry_price,
                        'volume_24h': current_30m.get('volume_24h', 0),
                        'atr_percent': current_30m.get('atr_percent', 0),
                        'volume_ratio': current_30m.get('volume_ratio', 0),
                        'rsi': current_30m.get('rsi', 50)
                    }
                    
                    # Disco57 сигнал (если есть Disco57Bot)
                    disco57_signal = None  # Будет заполнено при интеграции с Disco57Bot
                    
                    # Сбор признаков
                    disco57_features = self.disco57.collect_features(
                        symbol, market_data, mtf_data, lstm_prediction, disco57_signal
                    )
                    logger.debug(f"🔗 {symbol}: Признаки собраны через Feature Bus")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка сбора признаков Disco57: {e}")
            
            # ИЗМЕНЕНО: АДАПТИВНЫЙ SL НА ОСНОВЕ ATR (~1%)
            # Получаем ATR для расчета адаптивного SL
            adaptive_sl_percent = 0.7  # ИЗМЕНЕНО: Fallback значение 0.7% (было 1%)
            try:
                market_30m = mtf_data.get('30m', {}) or {}
                atr_30m = float(market_30m.get('atr', 0) or 0)
                if atr_30m > 0 and entry_price > 0:
                    atr_percent = (atr_30m / entry_price * 100)
                    # ИЗМЕНЕНО: Адаптивный SL: 0.8-1.2x ATR (минимум 0.8%, максимум 1.5%)
                    atr_multiplier = 1.0  # Базовый множитель
                    adaptive_sl_percent = max(0.8, min(1.5, atr_percent * atr_multiplier))
                    logger.info(f"📏 {symbol}: Адаптивный SL на основе ATR={atr_percent:.2f}% → SL={adaptive_sl_percent:.2f}%")
                else:
                    # Fallback: используем фиксированный SL 1%
                    position_notional = self.POSITION_SIZE * self.LEVERAGE  # $20
                    adaptive_sl_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100
                    logger.debug(f"⚠️ {symbol}: ATR недоступен, используем фиксированный SL={adaptive_sl_percent:.2f}%")
            except Exception as e:
                # Fallback: используем фиксированный SL 1%
                position_notional = self.POSITION_SIZE * self.LEVERAGE  # $20
                adaptive_sl_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100
                logger.debug(f"⚠️ {symbol}: Ошибка расчета ATR SL: {e}, используем фиксированный SL={adaptive_sl_percent:.2f}%")
            
            stop_loss = entry_price * (1 - adaptive_sl_percent / 100) if direction == 'buy' else entry_price * (1 + adaptive_sl_percent / 100)
            
            # 📊 Обновляем счетчик сгенерированных сигналов
            self.performance_stats['signals_generated'] = self.performance_stats.get('signals_generated', 0) + 1
            
            # ОТКЛЮЧЕНО: HIGH PERFORMANCE система удалена по требованию
            
            # 🔗 DISCO57: Сбор признаков через Feature Bus (перед созданием сигнала)
            disco57_features = None
            if self.disco57:
                try:
                    current_30m = mtf_data.get('30m', {})
                    # Подготовка данных для Feature Bus
                    market_data = {
                        'price': entry_price,
                        'volume_24h': current_30m.get('volume_24h', 0),
                        'atr_percent': current_30m.get('atr_percent', 0),
                        'volume_ratio': current_30m.get('volume_ratio', 0),
                        'rsi': current_30m.get('rsi', 50)
                    }
                    
                    # Disco57 сигнал (если есть Disco57Bot)
                    disco57_signal = None  # Будет заполнено при интеграции с Disco57Bot
                    
                    # Сбор признаков
                    disco57_features = self.disco57.collect_features(
                        symbol, market_data, mtf_data, lstm_prediction, disco57_signal
                    )
                    logger.debug(f"🔗 {symbol}: Признаки собраны через Feature Bus")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка сбора признаков Disco57: {e}")
            
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
            
            # 🚀 Сохраняем информацию о высоком потенциале в сигнале
            if high_potential_data:
                enhanced_signal.high_potential_data = high_potential_data
                enhanced_signal.potential_percent = high_potential_data.get('potential_percent', 0)
            
            # 🔗 DISCO57: Сохраняем признаки для Shadow Learning
            if self.disco57 and disco57_features:
                if not hasattr(enhanced_signal, 'disco57_features'):
                    enhanced_signal.disco57_features = {}
                enhanced_signal.disco57_features = disco57_features
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания расширенного сигнала: {e}")
            # Возвращаем базовый сигнал
            # Fallback: используем фиксированный SL при ошибке
            position_notional = self.POSITION_SIZE * self.LEVERAGE
            fallback_sl_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100
            fallback_stop_loss = entry_price * (1 - fallback_sl_percent / 100) if direction == 'buy' else entry_price * (1 + fallback_sl_percent / 100)
            
            return EnhancedSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                confidence=confidence,
                strategy_score=10.0,
                timeframe_analysis=mtf_data,
                tp_levels=[],
                stop_loss=fallback_stop_loss,
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
            
            # Формируем сообщение (упрощенное)
            message = f"""📥 #{signal.symbol} | {direction_text}
Текущая цена: {entry_price_str}

🎯 ТП: +1.15% (100% позиции)
🛑 SL: -${self.MAX_STOP_LOSS_USD:.2f} максимум

📈 Торговля
⚡ Сделка: ${self.POSITION_SIZE:.1f} x{self.LEVERAGE} = ${self.POSITION_NOTIONAL:.0f}
📌 Позиции: {current_positions}/{self.MAX_POSITIONS}"""
            
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
            # ИСПРАВЛЕНИЕ: Используем нормализацию для предотвращения дубликатов
            active_symbols = set()
            active_symbols_normalized = set()
            for pos in open_positions:
                symbol = pos.get('symbol', '')
                if symbol:
                    active_symbols.add(symbol)
                    symbol_norm = self.normalize_symbol(symbol)
                    active_symbols_normalized.add(symbol_norm)
                    
                    # ИСПРАВЛЕНИЕ: Проверяем с нормализацией, чтобы избежать дубликатов
                    # Ищем существующий ключ с нормализованным символом
                    existing_key = None
                    for key in self.active_positions.keys():
                        if self.normalize_symbol(key) == symbol_norm:
                            existing_key = key
                            break
                    
                    if existing_key:
                        # Обновляем существующую позицию
                        self.active_positions[existing_key].update({
                            'side': pos.get('side', ''),
                            'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                            'size': pos.get('contracts', pos.get('size', 0)),
                            'pnl_percent': pos.get('percentage', 0),
                        })
                    else:
                        # Добавляем новую позицию (используем оригинальный символ с биржи)
                        self.active_positions[symbol] = {
                            'side': pos.get('side', ''),
                            'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                            'size': pos.get('contracts', pos.get('size', 0)),
                            'pnl_percent': pos.get('percentage', 0),
                            'confidence': 0  # Будет обновлено
                        }
            
            # Удаляем закрытые позиции из словаря (с нормализацией)
            closed_symbols = []
            for key in self.active_positions.keys():
                key_norm = self.normalize_symbol(key)
                if key_norm not in active_symbols_normalized:
                    closed_symbols.append(key)
            
            for symbol in closed_symbols:
                del self.active_positions[symbol]
            
            return current_count
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка получения позиций с биржи: {e}")
            # Fallback на словарь
            return len(self.active_positions)
    
    async def _load_positions_from_exchange(self):
        """✅ ЗАДАЧА #1: Загрузка позиций с биржи при старте"""
        try:
            if not self.exchange:
                logger.warning("⚠️ Биржа не инициализирована, пропускаем загрузку позиций")
                return
            
            logger.info("🔄 Загрузка позиций с биржи при старте...")
            
            # Получаем все позиции с биржи
            try:
                positions = await self.exchange.fetch_positions(params={'category': 'linear', 'accountType': 'UNIFIED'})
            except Exception as e1:
                try:
                    positions = await self.exchange.fetch_positions(params={'category': 'linear'})
                except Exception as e2:
                    logger.warning(f"⚠️ Ошибка получения позиций: {e1} / {e2}")
                    positions = []
            
            if not positions:
                logger.info("📊 На бирже нет открытых позиций")
                return
            
            # Фильтруем только позиции с ненулевым размером
            open_positions = [p for p in positions if float(p.get('contracts', 0) or p.get('size', 0)) > 0]
            
            if not open_positions:
                logger.info("📊 На бирже нет открытых позиций с ненулевым размером")
                return
            
            # Загружаем позиции в active_positions
            loaded_count = 0
            for pos in open_positions:
                symbol = pos.get('symbol', '')
                if not symbol:
                    continue
                
                size = float(pos.get('contracts', 0) or pos.get('size', 0))
                if size <= 0:
                    continue
                
                # Нормализуем символ
                symbol_norm = self.normalize_symbol(symbol)
                
                # Проверяем, нет ли уже такой позиции (с нормализацией)
                existing_key = None
                for key in self.active_positions.keys():
                    if self.normalize_symbol(key) == symbol_norm:
                        existing_key = key
                        break
                
                if existing_key:
                    # Обновляем существующую позицию
                    self.active_positions[existing_key].update({
                        'side': pos.get('side', ''),
                        'entry_price': float(pos.get('entryPrice', 0) or pos.get('avgPrice', 0) or pos.get('markPrice', 0)),
                        'size': size,
                        'pnl_percent': float(pos.get('percentage', 0)),
                    })
                    logger.info(f"🔄 Обновлена позиция {symbol} из биржи: размер {size}, цена входа ${self.active_positions[existing_key]['entry_price']:.4f}")
                else:
                    # Добавляем новую позицию
                    entry_price = float(pos.get('entryPrice', 0) or pos.get('avgPrice', 0) or pos.get('markPrice', 0))
                    created_time = pos.get('createdTime') or pos.get('updatedTime')
                    opened_at = datetime.now(WARSAW_TZ)
                    if created_time:
                        try:
                            if isinstance(created_time, (int, float)):
                                opened_at = datetime.fromtimestamp(int(created_time) / 1000, tz=WARSAW_TZ)
                        except Exception:
                            pass
                    
                    self.active_positions[symbol] = {
                        'side': pos.get('side', ''),
                        'entry_price': entry_price,
                        'size': size,
                        'pnl_percent': float(pos.get('percentage', 0)),
                        'opened_at': opened_at,
                    }
                    logger.info(f"✅ Загружена позиция {symbol} с биржи: размер {size}, цена входа ${entry_price:.4f}")
                    loaded_count += 1
            
            logger.info(f"✅ Загружено позиций с биржи: {loaded_count} (всего в active_positions: {len(self.active_positions)})")
            
            # 🔴 ПРИОРИТЕТ 1.2: Проверка и восстановление SL/TP для загруженных позиций
            if loaded_count > 0:
                logger.info("🔄 Проверка SL/TP для загруженных позиций...")
                sl_tp_restored = 0
                for symbol, pos_info in list(self.active_positions.items()):
                    try:
                        entry_price = pos_info.get('entry_price', 0)
                        side = pos_info.get('side', '')
                        size = pos_info.get('size', 0)
                        
                        if entry_price <= 0 or not side or size <= 0:
                            continue
                        
                        # Проверяем SL/TP на бирже
                        sl_verified = await self._verify_sl_tp_on_exchange(symbol)
                        
                        if not sl_verified:
                            logger.warning(f"⚠️ {symbol}: SL/TP не установлены на бирже, восстанавливаем...")
                            
                            # Рассчитываем SL/TP на основе entry_price
                            direction = 'buy' if side.lower() in ['buy', 'long'] else 'sell'
                            
                            # Получаем ATR для расчета адаптивного SL
                            try:
                                mtf_data = await self._fetch_multi_timeframe_data(symbol)
                                current_30m = mtf_data.get('30m', {})
                                atr = current_30m.get('atr', 0)
                                atr_percent = current_30m.get('atr_percent', 0.7)
                                
                                if atr_percent > 0:
                                    adaptive_sl_percent = atr_percent
                                else:
                                    adaptive_sl_percent = 0.7  # Fallback
                            except:
                                adaptive_sl_percent = 0.7
                            
                            # Рассчитываем SL
                            if direction == 'buy':
                                stop_loss_price = entry_price * (1 - adaptive_sl_percent / 100)
                            else:
                                stop_loss_price = entry_price * (1 + adaptive_sl_percent / 100)
                            
                            # Рассчитываем TP1 (+1.15%)
                            if direction == 'buy':
                                tp_price = entry_price * (1 + 1.15 / 100)
                            else:
                                tp_price = entry_price * (1 - 1.15 / 100)
                            
                            # Устанавливаем SL/TP
                            success = await self._set_position_sl_tp_bybit(
                                symbol, side, size, stop_loss_price, [tp_price]
                            )
                            
                            if success:
                                logger.info(f"✅ {symbol}: SL/TP восстановлены (SL: ${stop_loss_price:.4f}, TP: ${tp_price:.4f})")
                                sl_tp_restored += 1
                                
                                # Обновляем информацию в active_positions
                                pos_info['stop_loss'] = stop_loss_price
                                pos_info['take_profit'] = tp_price
                            else:
                                logger.error(f"❌ {symbol}: Не удалось восстановить SL/TP")
                        else:
                            logger.debug(f"✅ {symbol}: SL/TP уже установлены на бирже")
                    except Exception as e:
                        logger.error(f"❌ Ошибка проверки SL/TP для {symbol}: {e}")
                
                if sl_tp_restored > 0:
                    logger.info(f"✅ Восстановлено SL/TP для {sl_tp_restored} позиций")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки позиций с биржи: {e}")
    
    async def _set_position_sl_tp_bybit(self, symbol: str, side: str, size: float, 
                                        stop_loss_price: float = None, take_profit_prices: list = None) -> bool:
        """
        Устанавливает Stop Loss и Take Profit для позиции на Bybit
        ИСПРАВЛЕНО: Использует только trading-stop API (pybit), так как conditional orders не работают
        
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
            
            # ИСПРАВЛЕНО: Проверяем, что позиция открыта перед установкой SL/TP
            try:
                positions = await self.exchange.fetch_positions(params={'category': 'linear'})
                position = next((p for p in positions if p.get('symbol') == symbol and 
                               (p.get('contracts', 0) or p.get('size', 0)) > 0), None)
                if not position:
                    logger.warning(f"⚠️ {symbol}: Позиция не найдена, пропускаем установку SL/TP")
                    return False
            except Exception as e_check:
                logger.debug(f"⚠️ {symbol}: Не удалось проверить позицию: {e_check}")
            
            # ИСПРАВЛЕНО: Округляем цены до правильного формата (Bybit требует определенную точность)
            def round_price(price: float, symbol: str) -> float:
                """Округляет цену до нужной точности для Bybit"""
                if price <= 0:
                    return price
                # Определяем количество знаков после запятой в зависимости от цены
                if price >= 1000:
                    return round(price, 2)  # Для дорогих активов (BTC и т.д.)
                elif price >= 100:
                    return round(price, 3)
                elif price >= 10:
                    return round(price, 4)
                elif price >= 1:
                    return round(price, 5)
                else:
                    return round(price, 6)  # Для дешевых активов
            
            # ИСПРАВЛЕНО: Используем только pybit (trading-stop API) - самый надежный способ
            # 1. Устанавливаем Stop Loss
            if stop_loss_price:
                rounded_sl = round_price(stop_loss_price, bybit_symbol)
                pybit_success = await self._set_sl_tp_pybit(symbol, rounded_sl, None)
                if pybit_success:
                    logger.info(f"🛑 {symbol}: Stop Loss установлен: ${rounded_sl:.6f}")
                    success = True
                else:
                    # Проверяем, может быть SL уже установлен (ошибка 34040)
                    try:
                        positions = await self.exchange.fetch_positions(params={'category': 'linear'})
                        position = next((p for p in positions if p.get('symbol') == symbol and 
                                       (p.get('contracts', 0) or p.get('size', 0)) > 0), None)
                        if position:
                            existing_sl = position.get('stopLoss') or position.get('stop_loss')
                            if existing_sl:
                                existing_sl_float = float(existing_sl)
                                # Если существующий SL близок к нашему (в пределах 1%), считаем успехом
                                if abs(existing_sl_float - rounded_sl) / rounded_sl < 0.01:
                                    logger.info(f"✅ {symbol}: Stop Loss уже установлен на бирже: ${existing_sl_float:.6f}")
                                    success = True
                                else:
                                    logger.warning(f"⚠️ {symbol}: SL на бирже отличается. Ожидался: ${rounded_sl:.6f}, на бирже: ${existing_sl_float:.6f}")
                            else:
                                logger.warning(f"⚠️ {symbol}: Не удалось установить SL. Будет контролироваться через мониторинг на ${rounded_sl:.6f}")
                    except Exception as e_verify:
                        logger.warning(f"⚠️ {symbol}: Не удалось проверить SL на бирже: {e_verify}")
                        logger.info(f"📝 {symbol}: Stop Loss будет контролироваться через мониторинг на ${rounded_sl:.6f}")
            
            # 2. Устанавливаем Take Profit (только первый уровень через trading-stop API)
            # Остальные уровни контролируются через мониторинг
            if take_profit_prices and len(take_profit_prices) > 0:
                # Устанавливаем только первый TP через API
                tp_price = take_profit_prices[0]
                rounded_tp = round_price(tp_price, bybit_symbol)
                pybit_success = await self._set_sl_tp_pybit(symbol, None, rounded_tp)
                if pybit_success:
                    logger.info(f"🎯 {symbol}: TP1 установлен: ${rounded_tp:.6f}")
                    success = True
                else:
                    # Проверяем, может быть TP уже установлен
                    try:
                        positions = await self.exchange.fetch_positions(params={'category': 'linear'})
                        position = next((p for p in positions if p.get('symbol') == symbol and 
                                       (p.get('contracts', 0) or p.get('size', 0)) > 0), None)
                        if position:
                            existing_tp = position.get('takeProfit') or position.get('take_profit')
                            if existing_tp:
                                existing_tp_float = float(existing_tp)
                                if abs(existing_tp_float - rounded_tp) / rounded_tp < 0.01:
                                    logger.info(f"✅ {symbol}: TP1 уже установлен на бирже: ${existing_tp_float:.6f}")
                                    success = True
                                else:
                                    logger.debug(f"⚠️ {symbol}: TP на бирже отличается. Ожидался: ${rounded_tp:.6f}, на бирже: ${existing_tp_float:.6f}")
                            else:
                                logger.info(f"📝 {symbol}: TP1 будет контролироваться через мониторинг на ${rounded_tp:.6f}")
                    except Exception as e_verify:
                        logger.debug(f"⚠️ {symbol}: Не удалось проверить TP на бирже: {e_verify}")
                        logger.info(f"📝 {symbol}: TP1 будет контролироваться через мониторинг на ${rounded_tp:.6f}")
                
                # Остальные уровни TP контролируются через мониторинг
                for i, tp_price_extra in enumerate(take_profit_prices[1:], 2):
                    rounded_tp_extra = round_price(tp_price_extra, bybit_symbol)
                    logger.info(f"📝 {symbol}: TP{i} будет контролироваться через мониторинг на ${rounded_tp_extra:.6f}")
        
        except Exception as e:
            logger.error(f"❌ {symbol}: Ошибка установки SL/TP: {e}")
        
        return success
    
    async def _set_sl_tp_pybit(self, symbol: str, stop_loss_price: float = None, take_profit_price: float = None) -> bool:
        """
        Устанавливает SL/TP используя официальную библиотеку pybit
        Работает для существующих позиций
        ИСПРАВЛЕНО: Улучшена обработка ошибок, включая 34040 (not modified)
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
            ret_code = response.get('retCode', -1)
            ret_msg = response.get('retMsg', '')
            
            if ret_code == 0:
                return True
            elif ret_code == 34040:
                # Ошибка 34040 означает "not modified" - возможно, значение уже установлено
                # Это не критично, считаем успехом
                logger.debug(f"ℹ️ {symbol}: SL/TP не изменены (уже установлены или значения совпадают): {ret_msg}")
                return True
            else:
                logger.debug(f"⚠️ {symbol}: Ошибка pybit set_trading_stop (retCode={ret_code}): {ret_msg}")
                return False
                
        except Exception as e:
            logger.debug(f"⚠️ {symbol}: Ошибка установки SL/TP через pybit: {e}")
            return False
    
    async def _retry_set_sl_tp(self, symbol: str, side: str, entry_price: float, 
                               stop_loss_price: float, tp_prices: list, size: float):
        """Повторная попытка установки SL/TP через 5 секунд после открытия позиции"""
        try:
            await asyncio.sleep(5)
            logger.info(f"🔄 {symbol}: Повторная попытка установки SL/TP...")
            success = await self._set_position_sl_tp_bybit(
                symbol=symbol,
                side=side,
                size=size,
                stop_loss_price=stop_loss_price,
                take_profit_prices=tp_prices
            )
            if success:
                logger.info(f"✅ {symbol}: SL/TP успешно установлены при повторной попытке!")
            else:
                logger.error(f"🚨 {symbol}: SL/TP НЕ УСТАНОВЛЕНЫ даже при повторной попытке! КРИТИЧНО!")
        except Exception as e:
            logger.error(f"❌ {symbol}: Ошибка повторной установки SL/TP: {e}")
    
    async def _verify_sl_tp_on_exchange(self, symbol: str, expected_sl: float = None, expected_tp: float = None) -> bool:
        """Проверяет, что SL/TP действительно установлены на бирже"""
        try:
            positions = await self.exchange.fetch_positions(params={'category': 'linear'})
            position = next((p for p in positions if p.get('symbol') == symbol and 
                           (p.get('contracts', 0) or p.get('size', 0)) > 0), None)
            
            if not position:
                logger.warning(f"⚠️ {symbol}: Позиция не найдена для проверки SL/TP")
                return False
            
            actual_sl = position.get('stopLoss') or position.get('stop_loss')
            actual_tp = position.get('takeProfit') or position.get('take_profit')
            
            if expected_sl:
                if not actual_sl or abs(float(actual_sl) - expected_sl) > expected_sl * 0.01:  # 1% допуск
                    logger.error(f"🚨 {symbol}: SL НЕ УСТАНОВЛЕН НА БИРЖЕ! Ожидался: ${expected_sl:.4f}, на бирже: {actual_sl}")
                    return False
                else:
                    logger.info(f"✅ {symbol}: SL подтвержден на бирже: ${actual_sl}")
            
            if expected_tp:
                if not actual_tp or abs(float(actual_tp) - expected_tp) > expected_tp * 0.01:  # 1% допуск
                    logger.warning(f"⚠️ {symbol}: TP может быть не установлен. Ожидался: ${expected_tp:.4f}, на бирже: {actual_tp}")
                else:
                    logger.info(f"✅ {symbol}: TP подтвержден на бирже: ${actual_tp}")
            
            return True
        except Exception as e:
            logger.error(f"❌ {symbol}: Ошибка проверки SL/TP на бирже: {e}")
            return False
    
    async def _retry_critical_operation(self, func, operation_name: str, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
        """
        🔴 ПРИОРИТЕТ 1.3: Универсальная retry логика для критических операций
        
        Args:
            func: Асинхронная функция для выполнения
            operation_name: Название операции (для логирования)
            max_retries: Максимальное количество попыток
            delay: Базовая задержка между попытками (секунды)
            *args, **kwargs: Аргументы для функции
        
        Returns:
            Результат выполнения функции или None при ошибке
        """
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str
                is_network_error = 'network' in error_str or 'timeout' in error_str or 'connection' in error_str or 'timeout' in error_str
                
                if attempt == max_retries - 1:
                    logger.error(f"❌ {operation_name}: Не удалось после {max_retries} попыток: {e}")
                    raise
                
                if is_rate_limit:
                    wait_time = delay * (2 ** attempt) * 2  # Удваиваем для rate limit
                    logger.warning(f"⚠️ {operation_name}: Rate limit (попытка {attempt + 1}/{max_retries}), ждем {wait_time:.1f}с...")
                    await asyncio.sleep(wait_time)
                elif is_network_error:
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"⚠️ {operation_name}: Network error (попытка {attempt + 1}/{max_retries}), ждем {wait_time:.1f}с...")
                    await asyncio.sleep(wait_time)
                else:
                    wait_time = delay * (attempt + 1)
                    logger.warning(f"⚠️ {operation_name}: Ошибка (попытка {attempt + 1}/{max_retries}): {e}, ждем {wait_time:.1f}с...")
                    await asyncio.sleep(wait_time)
        
        return None
    
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
            # 🚀 УЛУЧШЕНИЕ #2: АДАПТИВНЫЙ SL НА ОСНОВЕ ATR
            # Получаем ATR для расчета адаптивного SL
            try:
                df_30m = await self._fetch_ohlcv(symbol, '30m', limit=50)
                if not df_30m.empty and len(df_30m) >= 14:
                    # Рассчитываем ATR
                    high = df_30m['high'].values
                    low = df_30m['low'].values
                    close = df_30m['close'].values
                    
                    if self._talib_available and self._talib:
                        atr = self._talib.ATR(high, low, close, timeperiod=14)[-1]
                        atr_percent = (atr / entry_price * 100) if entry_price > 0 else 0
                        
                        # Адаптивный SL: 1.5-2.5x ATR (минимум 2%, максимум 5%)
                        atr_multiplier = 2.0  # Базовый множитель
                        adaptive_sl_percent = max(2.0, min(5.0, atr_percent * atr_multiplier))
                        
                        logger.info(f"📏 {symbol}: ATR={atr_percent:.2f}% → Адаптивный SL={adaptive_sl_percent:.2f}% (вместо фиксированного 3%)")
                    else:
                        # Fallback: используем фиксированный SL
                        position_notional = self.POSITION_SIZE * self.LEVERAGE
                        adaptive_sl_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100
                        logger.debug(f"⚠️ {symbol}: TA-Lib недоступен, используем фиксированный SL={adaptive_sl_percent:.2f}%")
                else:
                    # Fallback: используем фиксированный SL
                    position_notional = self.POSITION_SIZE * self.LEVERAGE
                    adaptive_sl_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100
                    logger.debug(f"⚠️ {symbol}: Недостаточно данных для ATR, используем фиксированный SL={adaptive_sl_percent:.2f}%")
            except Exception as e:
                # Fallback: используем фиксированный SL
                position_notional = self.POSITION_SIZE * self.LEVERAGE
                adaptive_sl_percent = (self.MAX_STOP_LOSS_USD / position_notional) * 100
                logger.debug(f"⚠️ {symbol}: Ошибка расчета ATR SL: {e}, используем фиксированный SL={adaptive_sl_percent:.2f}%")
            
            # 🔴 ИЗМЕНЕНО: TP1 = +1.15% (компенсация комиссии, сразу в без убыток)
            tp_percent = 1.15  # TP1: +1.15% (закрывает 100% позиции)
            # Дополнительные TP: TP2: 2.5% (ROE 50%), TP3: 4% (ROE 80%), TP4: 5% (ROE 100%), TP5: 6% (ROE 120%)
            
            if side == 'buy':
                stop_loss_price = entry_price * (1 - adaptive_sl_percent / 100.0)
                tp_price = entry_price * (1 + tp_percent / 100.0)
            else:
                stop_loss_price = entry_price * (1 + adaptive_sl_percent / 100.0)
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
    
    def _update_performance_metrics(self, pnl_usd: float, pnl_percent: float, 
                                    symbol: str, duration_seconds: float = None, 
                                    timeframe: str = None):
        """
        📊 Обновляет метрики производительности при закрытии позиции
        
        Args:
            pnl_usd: Прибыль/убыток в USD
            pnl_percent: Прибыль/убыток в процентах
            symbol: Символ торговой пары
            duration_seconds: Длительность позиции в секундах (опционально)
            timeframe: Таймфрейм сигнала (опционально)
        """
        try:
            stats = self.performance_stats
            
            # Обновляем счетчики сделок
            stats['total_trades'] = stats.get('total_trades', 0) + 1
            stats['positions_closed'] = stats.get('positions_closed', 0) + 1
            
            if pnl_usd > 0:
                stats['winning_trades'] = stats.get('winning_trades', 0) + 1
                stats['total_profit'] = stats.get('total_profit', 0.0) + pnl_usd
                # Обновляем максимальную прибыль
                if pnl_usd > stats.get('max_win', 0.0):
                    stats['max_win'] = pnl_usd
                
                # 🚨 ПСИХОЛОГИЧЕСКИЙ СТОП-КОНТУР: Сбрасываем счетчик убытков при прибыли
                self.consecutive_losses = 0
                if self._trading_paused_due_to_losses:
                    self._trading_paused_due_to_losses = False
                    logger.info(f"✅ Торговля возобновлена после прибыльной сделки (сброс consecutive_losses)")
            else:
                stats['losing_trades'] = stats.get('losing_trades', 0) + 1
                stats['total_loss'] = stats.get('total_loss', 0.0) + abs(pnl_usd)
                # Обновляем максимальный убыток
                if abs(pnl_usd) > stats.get('max_loss', 0.0):
                    stats['max_loss'] = abs(pnl_usd)
                
                # 🚨 ПСИХОЛОГИЧЕСКИЙ СТОП-КОНТУР: Увеличиваем счетчик последовательных убытков
                self.consecutive_losses += 1
                self.last_loss_time = datetime.now(WARSAW_TZ)
                
                if self.consecutive_losses >= self.max_consecutive_losses:
                    if not self._trading_paused_due_to_losses:
                        self._trading_paused_due_to_losses = True
                        logger.error(f"🚨 КРИТИЧНО: {self.consecutive_losses} убытков подряд! ТОРГОВЛЯ ПРИОСТАНОВЛЕНА!")
                        logger.error(f"🚨 Пауза торговли до следующей прибыльной сделки или ручного сброса")
                        if self.telegram_bot:
                            try:
                                asyncio.create_task(self.send_telegram_v4(
                                    f"🚨 КРИТИЧНО: ТОРГОВЛЯ ПРИОСТАНОВЛЕНА\n"
                                    f"Последовательных убытков: {self.consecutive_losses}\n"
                                    f"Лимит: {self.max_consecutive_losses}\n"
                                    f"Пауза до следующей прибыльной сделки"
                                ))
                            except:
                                pass
            
            # Общий PnL
            stats['total_pnl'] = stats.get('total_pnl', 0.0) + pnl_usd
            
            # 🚨 ОТСЛЕЖИВАНИЕ ДНЕВНОЙ ПРОСАДКИ (MAX_DAILY_DRAWDOWN)
            # ИСПРАВЛЕНО: Добавлена защита от деления на ноль и корректный расчет
            today = datetime.now(WARSAW_TZ).date().isoformat()
            
            # Сброс трекера при новом дне (удаляем старые записи старше 1 дня)
            yesterday = (datetime.now(WARSAW_TZ) - timedelta(days=1)).date().isoformat()
            if yesterday in self.daily_pnl_tracker:
                del self.daily_pnl_tracker[yesterday]
            
            if today not in self.daily_pnl_tracker:
                self.daily_pnl_tracker[today] = {
                    'pnl': 0.0,
                    'peak': 0.0,
                    'drawdown': 0.0,
                    'initial_balance': self.current_balance if hasattr(self, 'current_balance') else 0.0
                }
            
            daily_tracker = self.daily_pnl_tracker[today]
            daily_tracker['pnl'] += pnl_usd
            
            # Обновляем пик (максимальный PnL за день)
            if daily_tracker['pnl'] > daily_tracker['peak']:
                daily_tracker['peak'] = daily_tracker['pnl']
            
            # Рассчитываем просадку от пика
            # ИСПРАВЛЕНО: Защита от деления на ноль и корректный расчет просадки
            if daily_tracker['peak'] > 0:
                # Просадка от пика (в процентах)
                daily_tracker['drawdown'] = ((daily_tracker['peak'] - daily_tracker['pnl']) / daily_tracker['peak']) * 100
            elif daily_tracker['peak'] < 0:
                # Если пик отрицательный (все сделки убыточные), считаем просадку от нуля
                daily_tracker['drawdown'] = abs(daily_tracker['pnl']) if daily_tracker['pnl'] < 0 else 0.0
            else:
                # Если peak = 0 (нет прибыли, но и нет убытков), просадка = 0
                daily_tracker['drawdown'] = abs(daily_tracker['pnl']) if daily_tracker['pnl'] < 0 else 0.0
            
            # Дополнительная проверка: просадка не может быть отрицательной
            daily_tracker['drawdown'] = max(0.0, daily_tracker['drawdown'])
            
            # Проверяем превышение лимита MAX_DAILY_DRAWDOWN
            # ИСПРАВЛЕНО: Дополнительная проверка на валидность drawdown
            if daily_tracker['drawdown'] >= self.max_daily_drawdown_percent and daily_tracker['drawdown'] > 0:
                if not self._trading_paused_due_to_drawdown:
                    self._trading_paused_due_to_drawdown = True
                    logger.error(f"🚨 КРИТИЧНО: Дневная просадка {daily_tracker['drawdown']:.2f}% >= {self.max_daily_drawdown_percent}%")
                    logger.error(f"🚨 ТОРГОВЛЯ ПРИОСТАНОВЛЕНА из-за превышения MAX_DAILY_DRAWDOWN!")
                    logger.error(f"   Дневной PnL: ${daily_tracker['pnl']:.2f} | Пик: ${daily_tracker['peak']:.2f}")
                    if self.telegram_bot:
                        try:
                            asyncio.create_task(self.send_telegram_v4(
                                f"🚨 КРИТИЧНО: ТОРГОВЛЯ ПРИОСТАНОВЛЕНА\n"
                                f"Дневная просадка: {daily_tracker['drawdown']:.2f}%\n"
                                f"Лимит: {self.max_daily_drawdown_percent}%\n"
                                f"Дневной PnL: ${daily_tracker['pnl']:.2f}"
                            ))
                        except:
                            pass
            else:
                # Если просадка снизилась ниже лимита, возобновляем торговлю
                if self._trading_paused_due_to_drawdown:
                    self._trading_paused_due_to_drawdown = False
                    logger.info(f"✅ Дневная просадка снизилась до {daily_tracker['drawdown']:.2f}% < {self.max_daily_drawdown_percent}% - торговля возобновлена")
            
            # Обновляем средние значения
            winning_count = stats.get('winning_trades', 0)
            losing_count = stats.get('losing_trades', 0)
            
            if winning_count > 0:
                stats['avg_win'] = stats.get('total_profit', 0.0) / winning_count
            if losing_count > 0:
                stats['avg_loss'] = stats.get('total_loss', 0.0) / losing_count
            
            # Profit Factor
            total_profit = stats.get('total_profit', 0.0)
            total_loss = stats.get('total_loss', 0.0)
            if total_loss > 0:
                stats['profit_factor'] = total_profit / total_loss
            elif total_profit > 0:
                stats['profit_factor'] = float('inf')
            else:
                stats['profit_factor'] = 0.0
            
            # Win Rate
            total_trades = stats.get('total_trades', 0)
            if total_trades > 0:
                stats['win_rate'] = (winning_count / total_trades) * 100
            
            # 🔴 ПРИОРИТЕТ 2.1: Мониторинг производительности в реальном времени
            # Проверяем метрики и отправляем алерты при отклонении от нормы
            self._check_performance_alerts()
            
            # 🔴 ПРИОРИТЕТ 3.3: Улучшенное логирование метрик
            # Логируем метрики каждые 10 сделок в структурированном формате
            if total_trades > 0 and total_trades % 10 == 0:
                try:
                    metrics_log = {
                        'timestamp': datetime.now(WARSAW_TZ).isoformat(),
                        'total_trades': total_trades,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'total_pnl': total_pnl,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'risk_reward': avg_win / avg_loss if avg_loss > 0 else 0.0
                    }
                    logger.info(f"📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ (каждые 10 сделок): {metrics_log}")
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка логирования метрик: {e}")
            
            # Максимальная просадка (упрощенный расчет)
            if pnl_usd < 0:
                current_drawdown = abs(pnl_usd)
                if current_drawdown > stats.get('max_drawdown', 0.0):
                    stats['max_drawdown'] = current_drawdown
            
            # Длительность позиции
            if duration_seconds is not None:
                current_avg = stats.get('avg_trade_duration', 0.0)
                total_closed = stats.get('positions_closed', 1)
                # Взвешенное среднее
                stats['avg_trade_duration'] = (current_avg * (total_closed - 1) + duration_seconds) / total_closed
            
            # Обновляем время последней сделки
            stats['last_trade_time'] = datetime.now()
            
            # Добавляем символ в список торгуемых
            if 'symbols_traded' in stats:
                stats['symbols_traded'].add(symbol)
            
            # Обновляем использование таймфреймов
            if timeframe and 'timeframe_usage' in stats:
                if timeframe in stats['timeframe_usage']:
                    stats['timeframe_usage'][timeframe] += 1
            
            # Обновляем счетчик API вызовов (если есть)
            if hasattr(self, 'api_optimizer') and self.api_optimizer:
                # Можно добавить счетчик из api_optimizer если он есть
                pass
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления метрик производительности: {e}")
    
    def _check_performance_alerts(self):
        """
        🔴 ПРИОРИТЕТ 2.1: Проверка метрик производительности и отправка алертов
        
        Отслеживает:
        - Win Rate (норма: 60-80%)
        - Profit Factor (норма: > 1.2)
        - Средний PnL (норма: > 0)
        - Risk/Reward (норма: > 1.0)
        """
        try:
            stats = self.performance_stats
            total_trades = stats.get('total_trades', 0)
            
            # Проверяем только если есть достаточно сделок (минимум 5)
            if total_trades < 5:
                return
            
            win_rate = stats.get('win_rate', 0)
            profit_factor = stats.get('profit_factor', 0)
            avg_win = stats.get('avg_win', 0)
            avg_loss = stats.get('avg_loss', 0)
            total_pnl = stats.get('total_pnl', 0)
            
            # Рассчитываем Risk/Reward
            risk_reward = 0.0
            if avg_loss > 0:
                risk_reward = avg_win / avg_loss
            
            alerts = []
            
            # Алерт 1: Низкий Win Rate (< 50%)
            if win_rate < 50:
                alerts.append(f"⚠️ Низкий Win Rate: {win_rate:.1f}% (норма: 60-80%)")
            
            # Алерт 2: Низкий Profit Factor (< 1.0)
            if profit_factor < 1.0 and profit_factor > 0:
                alerts.append(f"⚠️ Низкий Profit Factor: {profit_factor:.2f} (норма: > 1.2)")
            
            # Алерт 3: Плохой Risk/Reward (< 1.0)
            if risk_reward < 1.0 and risk_reward > 0:
                alerts.append(f"⚠️ Плохой Risk/Reward: {risk_reward:.2f} (норма: > 1.0)")
            
            # Алерт 4: Отрицательный общий PnL
            if total_pnl < 0:
                alerts.append(f"⚠️ Отрицательный PnL: ${total_pnl:.2f}")
            
            # Алерт 5: Слишком высокий Win Rate (> 90%) - возможное переобучение
            if win_rate > 90:
                alerts.append(f"⚠️ Слишком высокий Win Rate: {win_rate:.1f}% (возможно переобучение)")
            
            # Отправляем алерты в Telegram (не чаще раза в час)
            if alerts and hasattr(self, 'last_performance_alert_time'):
                now = datetime.now(WARSAW_TZ)
                time_since_last = (now - self.last_performance_alert_time).total_seconds()
                
                if time_since_last > 3600:  # Раз в час
                    alert_message = "📊 АЛЕРТЫ ПРОИЗВОДИТЕЛЬНОСТИ:\n\n" + "\n".join(alerts)
                    alert_message += f"\n\n📈 Статистика:\n"
                    alert_message += f"• Сделок: {total_trades}\n"
                    alert_message += f"• Win Rate: {win_rate:.1f}%\n"
                    alert_message += f"• Profit Factor: {profit_factor:.2f}\n"
                    alert_message += f"• R/R: {risk_reward:.2f}\n"
                    alert_message += f"• Общий PnL: ${total_pnl:.2f}"
                    
                    if self.telegram_bot:
                        try:
                            asyncio.create_task(self.send_telegram_v4(alert_message))
                            self.last_performance_alert_time = now
                        except:
                            pass
            elif alerts:
                # Первый раз - инициализируем время
                self.last_performance_alert_time = datetime.now(WARSAW_TZ)
                
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки алертов производительности: {e}")
    
    async def _perform_backup(self):
        """🔴 ПРИОРИТЕТ 2.2: Выполнение резервного копирования в фоновом режиме"""
        try:
            if self.backup_system:
                logger.info("📦 Запуск автоматического резервного копирования...")
                # Резервное копирование конфигурации и базы данных (модели реже)
                self.backup_system.backup_config()
                self.backup_system.backup_database()
                logger.info("✅ Резервное копирование завершено")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка выполнения резервного копирования: {e}")
    
    async def get_performance_metrics(self) -> dict:
        """
        🔴 ПРИОРИТЕТ 2.1: Получение метрик производительности для команды /metrics
        
        Returns:
            dict с метриками производительности
        """
        try:
            stats = self.performance_stats
            total_trades = stats.get('total_trades', 0)
            winning_trades = stats.get('winning_trades', 0)
            losing_trades = stats.get('losing_trades', 0)
            win_rate = stats.get('win_rate', 0)
            profit_factor = stats.get('profit_factor', 0)
            avg_win = stats.get('avg_win', 0)
            avg_loss = stats.get('avg_loss', 0)
            total_pnl = stats.get('total_pnl', 0)
            
            # Рассчитываем Risk/Reward
            risk_reward = 0.0
            if avg_loss > 0:
                risk_reward = avg_win / avg_loss
            
            # Получаем статистику за последние 24 часа
            stats_24h = await self._get_trade_stats_24h()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'risk_reward': risk_reward,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'stats_24h': stats_24h
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения метрик производительности: {e}")
            return {}
    
    async def open_position_automatically(self, signal: EnhancedSignal) -> bool:
        """
        🚀 Автоматически открывает позицию на бирже
        
        Args:
            signal: Торговый сигнал
        
        Returns:
            True если позиция успешно открыта, False в противном случае
        """
        try:
            # Проверяем флаг паузы торговли (дополнительная защита)
            if hasattr(self, '_trading_paused') and self._trading_paused:
                logger.debug(f"⏸️ {signal.symbol}: Торговля на паузе, пропускаем открытие позиции")
                return False
            
            if not self.exchange:
                logger.error("❌ Биржа не инициализирована")
                return False
            
            symbol = signal.symbol
            side = 'Buy' if signal.direction == 'buy' else 'Sell'
            
            # 🚨 КРИТИЧЕСКАЯ ПРОВЕРКА #1: Проверка направления сигнала против рынка
            # БЛОКИРУЕМ открытие позиций против тренда
            try:
                market_condition = getattr(self, '_current_market_condition', 'NEUTRAL')
                
                # Проверяем глобальный тренд на 1h
                try:
                    df_1h = await self._fetch_ohlcv(symbol, '1h', limit=200)
                    if not df_1h.empty and len(df_1h) >= 200:
                        close_1h = df_1h['close']
                        ema50_1h = float(close_1h.ewm(span=50, adjust=False).mean().iloc[-1])
                        ema200_1h = float(close_1h.ewm(span=200, adjust=False).mean().iloc[-1])
                        global_trend_bullish = ema50_1h > ema200_1h
                        global_trend_bearish = ema50_1h < ema200_1h
                        
                        # КРИТИЧЕСКАЯ ПРОВЕРКА: Блокируем открытие против тренда
                        if signal.direction == 'buy':
                            # BUY запрещен в медвежьем рынке
                            if market_condition == 'BEARISH' or (global_trend_bearish and not global_trend_bullish):
                                logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: {symbol} BUY ЗАПРЕЩЕН! Рынок BEARISH или глобальный тренд медвежий (EMA50={ema50_1h:.6f} < EMA200={ema200_1h:.6f})")
                                logger.error(f"   Позиция НЕ будет открыта - защита от торговли против тренда!")
                                return False
                        elif signal.direction == 'sell':
                            # SELL запрещен в бычьем рынке
                            if market_condition == 'BULLISH' or (global_trend_bullish and not global_trend_bearish):
                                logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: {symbol} SELL ЗАПРЕЩЕН! Рынок BULLISH или глобальный тренд бычий (EMA50={ema50_1h:.6f} > EMA200={ema200_1h:.6f})")
                                logger.error(f"   Позиция НЕ будет открыта - защита от торговли против тренда!")
                                return False
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Ошибка проверки глобального тренда: {e}. Продолжаем с проверкой market_condition...")
                    # Fallback: проверяем только market_condition
                    if signal.direction == 'buy' and market_condition == 'BEARISH':
                        logger.error(f"🚨 {symbol} BUY ЗАПРЕЩЕН! Рынок BEARISH")
                        return False
                    elif signal.direction == 'sell' and market_condition == 'BULLISH':
                        logger.error(f"🚨 {symbol} SELL ЗАПРЕЩЕН! Рынок BULLISH")
                        return False
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: Ошибка проверки направления сигнала: {e}. Продолжаем...")
            
            # 🚨 КРИТИЧЕСКАЯ ПРОВЕРКА: Нет ли уже открытой позиции по этому символу (с нормализацией)
            # Используем единую функцию нормализации из __init__
            symbol_norm = self.normalize_symbol(symbol)
            
            # 1. Проверяем в active_positions с нормализацией
            for active_symbol, pos_info in list(self.active_positions.items()):
                active_symbol_norm = self.normalize_symbol(active_symbol)
                if active_symbol_norm == symbol_norm:
                    logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: {symbol} уже в active_positions как {active_symbol}")
                    logger.error(f"   Позиция уже отслеживается! Пропускаем открытие.")
                    return False
            
            # 2. КРИТИЧЕСКАЯ ПРОВЕРКА: Получаем ВСЕ позиции с биржи и проверяем с нормализацией
            try:
                all_positions = await self.exchange.fetch_positions(params={'category': 'linear'})
                # Также проверяем через UNIFIED аккаунт
                try:
                    unified_positions = await self.exchange.fetch_positions(params={'category': 'linear', 'accountType': 'UNIFIED'})
                    if unified_positions:
                        all_positions.extend(unified_positions)
                except:
                    pass
                
                for pos in all_positions:
                    pos_symbol = pos.get('symbol', '')
                    pos_symbol_norm = self.normalize_symbol(pos_symbol)
                    
                    if pos_symbol_norm == symbol_norm:
                        size = float(pos.get('contracts', 0) or pos.get('size', 0) or 0)
                        if size > 0:
                            logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: {symbol} УЖЕ ОТКРЫТ НА БИРЖЕ как {pos_symbol} (размер: {size})")
                            logger.error(f"   Позиция уже существует на бирже! Пропускаем открытие.")
                            # Добавляем в активные позиции для синхронизации
                            self.active_positions[pos_symbol] = {
                                'side': pos.get('side', ''),
                                'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                                'size': size,
                                'pnl_percent': pos.get('percentage', 0),
                                'opened_at': datetime.now(WARSAW_TZ)
                            }
                            # Получаем createdTime
                            created_time = pos.get('createdTime') or pos.get('updatedTime')
                            if created_time:
                                try:
                                    if isinstance(created_time, (int, float)):
                                        self.active_positions[pos_symbol]['opened_at'] = datetime.fromtimestamp(int(created_time) / 1000, tz=WARSAW_TZ)
                                except Exception:
                                    pass
                            return False
            except Exception as e:
                logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА проверки позиций для {symbol}: {e}")
                # В случае ошибки не открываем позицию
                return False
            
            # 3. ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Проверяем через pybit API (более надежно)
            try:
                from pybit.unified_trading import HTTP
                session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False)
                bybit_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                bybit_symbol = bybit_symbol.replace('/', '').replace(':', '')
                
                positions_response = session.get_position_info(category='linear', symbol=bybit_symbol)
                positions_list = positions_response.get('result', {}).get('list', []) or []
                
                for pos in positions_list:
                    pos_size = float(pos.get('size', 0) or 0)
                    if pos_size > 0:
                        logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: {symbol} УЖЕ ОТКРЫТ НА БИРЖЕ (проверка через pybit, размер: {pos_size})")
                        logger.error(f"   Позиция уже существует! Пропускаем открытие.")
                        return False
            except Exception as e:
                logger.debug(f"⚠️ Не удалось проверить позицию через pybit для {symbol}: {e}. Продолжаем...")
            
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
                # Для позиции $20 с плечом 20x нужна маржа $1
                required_margin = self.POSITION_SIZE  # $1
                
                # Проверяем, сколько уже используется в открытых позициях
                current_positions_count = await self._get_current_open_positions_count()
                used_margin = current_positions_count * self.POSITION_SIZE  # Маржа для каждой позиции $1
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
            
            # 🚨 КРИТИЧЕСКАЯ ПРОВЕРКА: Аварийные условия перед открытием позиции
            # 🔴 ПРИОРИТЕТ 2.4: Расширенная валидация данных перед открытием позиции
            
            # Проверка 1: entry_price не None и > 0
            entry_price = signal.entry_price
            if entry_price is None or entry_price <= 0:
                logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА - entry_price is None или <= 0: {entry_price}")
                return False
            
            # Проверка 1.1: entry_price в разумных пределах (не слишком мал/велик)
            if entry_price < 0.0001 or entry_price > 1000000:
                logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА - entry_price вне разумных пределов: {entry_price}")
                return False
            
            # Проверка 1.2: Сравнение entry_price с текущей рыночной ценой (допуск 5%)
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = float(ticker.get('last') or ticker.get('close') or 0)
                if current_price > 0:
                    price_diff_pct = abs(entry_price - current_price) / current_price * 100
                    if price_diff_pct > 5:
                        logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА - entry_price отличается от рыночной цены на {price_diff_pct:.2f}% (допуск: 5%)")
                        logger.error(f"   Entry: ${entry_price:.4f}, Market: ${current_price:.4f}")
                        return False
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: Не удалось проверить рыночную цену: {e}")
            
            # Проверка 2: Проверяем индикаторы на валидность
            try:
                import math
                import numpy as np
                
                # Получаем данные индикаторов из сигнала
                if hasattr(signal, 'disco57_features') and signal.disco57_features:
                    features = signal.disco57_features
                    # Проверяем ATR
                    atr = features.get('atr', 0) or features.get('atr_percent', 0)
                    if atr is None or (isinstance(atr, float) and (math.isnan(atr) or math.isinf(atr) or atr == 0)):
                        logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА - ATR is None, NaN, Inf или 0: {atr}")
                        return False
                    
                    # Проверяем Volume
                    volume = features.get('volume', 0) or features.get('volume_24h', 0) or features.get('volume_ratio', 0)
                    if volume is None or (isinstance(volume, float) and (math.isnan(volume) or math.isinf(volume) or volume == 0)):
                        logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА - Volume is None, NaN, Inf или 0: {volume}")
                        return False
                    
                    # Проверяем RSI
                    rsi = features.get('rsi', 50)
                    if rsi is None or (isinstance(rsi, float) and (math.isnan(rsi) or math.isinf(rsi))):
                        logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА - RSI is None, NaN или Inf: {rsi}")
                        return False
            except Exception as e:
                logger.error(f"🚨 {symbol}: Ошибка проверки индикаторов: {e}")
                return False
            
            # 3. Рассчитываем размер позиции
            position_size_usdt = self.POSITION_SIZE  # $1
            position_notional = position_size_usdt * self.LEVERAGE  # $20 с плечом 20x
            
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
                    max_allowed_margin = self.POSITION_SIZE * 1.02  # Максимум $1.02 (2% запас на округление)
                    if actual_margin > max_allowed_margin:
                        logger.error(f"❌ {symbol}: Фактическая маржа (${actual_margin:.2f}) превышает допустимую (${max_allowed_margin:.2f})!")
                        logger.error(f"   Рассчитываем меньший размер позиции...")
                        # Пересчитываем с точным контролем маржи - максимальная нотиональная стоимость $25
                        max_allowed_notional = self.POSITION_SIZE * self.LEVERAGE  # Максимум $20
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
                    
                    # 🔴 ПРИОРИТЕТ 2.4: Дополнительная валидация размера позиции
                    # Проверка: размер позиции не должен быть слишком мал (минимум $0.10)
                    if actual_notional < 0.10:
                        logger.error(f"❌ {symbol}: Размер позиции слишком мал: ${actual_notional:.2f} (минимум: $0.10)")
                        return False
                    
                    # Проверка: количество контрактов должно быть положительным
                    if qty <= 0:
                        logger.error(f"❌ {symbol}: Количество контрактов <= 0: {qty:.6f}")
                        return False
                    
                    # Проверка: количество контрактов должно быть >= минимального размера
                    if qty < min_amount:
                        logger.error(f"❌ {symbol}: Количество контрактов ({qty:.6f}) < минимального размера ({min_amount:.6f})")
                        return False
                    
                    # Логируем информацию о размере позиции
                    logger.info(f"📊 {symbol}: Размер позиции | Нотиональная: ${actual_notional:.2f} | Маржа: ${actual_margin:.2f} (контроль: максимум ${self.POSITION_SIZE:.2f})")
                else:
                    logger.error(f"❌ {symbol}: Поддерживаются только линейные контракты")
                    return False
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Ошибка расчета размера позиции: {e}")
                return False
            
            # 3. Открываем позицию (LIMIT ORDER по лучшей цене с автообновлением)
            try:
                logger.info(f"🚀 {symbol}: Открываю позицию {side} | Размер: {qty:.6f} | Цена входа: ${entry_price:.4f}")
                
                # 🎯 НОВОЕ: Лимитный ордер по лучшей цене спроса/предложения (или с отступом)
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    expected_price = float(ticker.get('last') or ticker.get('close') or entry_price)
                    bid_price = float(ticker.get('bid', expected_price))
                    ask_price = float(ticker.get('ask', expected_price))
                    
                    # Для BUY используем ask (лучшая цена предложения), для SELL используем bid (лучшая цена спроса)
                    # Можно добавить небольшой отступ (например, 0.01%) для гарантии исполнения
                    price_offset_pct = 0.01  # 0.01% отступ для гарантии исполнения
                    
                    if signal.direction == 'buy':
                        # Для BUY: лимитный ордер по ask цене (или с небольшим отступом вверх)
                        limit_price = ask_price * (1 + price_offset_pct / 100.0)
                        best_price = ask_price
                    else:
                        # Для SELL: лимитный ордер по bid цене (или с небольшим отступом вниз)
                        limit_price = bid_price * (1 - price_offset_pct / 100.0)
                        best_price = bid_price
                    
                    logger.info(f"📊 {symbol}: Лимитный ордер | Лучшая цена: ${best_price:.6f} | Лимит: ${limit_price:.6f} | Отступ: {price_offset_pct}%")
                    
                    # Размещаем лимитный ордер
                    order = await self.exchange.create_limit_order(
                        symbol=symbol,
                        side='buy' if signal.direction == 'buy' else 'sell',
                        amount=qty,
                        price=limit_price,
                        params={
                            'category': 'linear',
                            'reduceOnly': False,
                            'timeInForce': 'GTC'  # Good Till Cancel - ордер активен до отмены
                        }
                    )
                    
                    order_id = order.get('id')
                    logger.info(f"✅ {symbol}: Лимитный ордер размещен | ID: {order_id} | Цена: ${limit_price:.6f}")
                    
                    # 🎯 АВТООБНОВЛЕНИЕ: Обновляем цену лимитного ордера до исполнения или удаления
                    max_update_attempts = 10  # Максимум 10 попыток обновления
                    update_interval = 2  # Обновляем каждые 2 секунды
                    order_filled = False
                    
                    for attempt in range(max_update_attempts):
                        await asyncio.sleep(update_interval)
                        
                        # Проверяем статус ордера
                        try:
                            order_status = await self.exchange.fetch_order(order_id, symbol)
                            order_filled = order_status.get('status') == 'closed' or order_status.get('filled', 0) > 0
                            
                            if order_filled:
                                logger.info(f"✅ {symbol}: Лимитный ордер исполнен | ID: {order_id}")
                                break
                            
                            # Получаем актуальную лучшую цену
                            ticker = await self.exchange.fetch_ticker(symbol)
                            if signal.direction == 'buy':
                                new_ask = float(ticker.get('ask', limit_price))
                                new_limit_price = new_ask * (1 + price_offset_pct / 100.0)
                            else:
                                new_bid = float(ticker.get('bid', limit_price))
                                new_limit_price = new_bid * (1 - price_offset_pct / 100.0)
                            
                            # Обновляем цену ордера только если она изменилась более чем на 0.01%
                            price_diff_pct = abs(new_limit_price - limit_price) / limit_price * 100 if limit_price > 0 else 0
                            if price_diff_pct > 0.01:
                                try:
                                    # Отменяем старый ордер
                                    await self.exchange.cancel_order(order_id, symbol)
                                    # Размещаем новый ордер с обновленной ценой
                                    order = await self.exchange.create_limit_order(
                                        symbol=symbol,
                                        side='buy' if signal.direction == 'buy' else 'sell',
                                        amount=qty,
                                        price=new_limit_price,
                                        params={
                                            'category': 'linear',
                                            'reduceOnly': False,
                                            'timeInForce': 'GTC'
                                        }
                                    )
                                    order_id = order.get('id')
                                    limit_price = new_limit_price
                                    logger.debug(f"🔄 {symbol}: Лимитный ордер обновлен | Новая цена: ${limit_price:.6f}")
                                except Exception as e:
                                    logger.debug(f"⚠️ {symbol}: Ошибка обновления лимитного ордера: {e}")
                                    # Продолжаем с текущим ордером
                        except Exception as e:
                            logger.debug(f"⚠️ {symbol}: Ошибка проверки статуса ордера: {e}")
                    
                    # Если ордер не исполнился за отведенное время, отменяем и используем market order
                    if not order_filled:
                        try:
                            await self.exchange.cancel_order(order_id, symbol)
                            logger.warning(f"⚠️ {symbol}: Лимитный ордер не исполнился, используем market order")
                            # Fallback на market order
                            order = await self.exchange.create_market_order(
                                symbol=symbol,
                                side='buy' if signal.direction == 'buy' else 'sell',
                                amount=qty,
                                params={
                                    'category': 'linear',
                                    'reduceOnly': False
                                }
                            )
                        except Exception as e:
                            logger.error(f"❌ {symbol}: Ошибка отмены лимитного ордера: {e}")
                            # Продолжаем с существующим ордером
                    
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Ошибка размещения лимитного ордера, используем market order: {e}")
                    # Fallback на market order
                    order = await self.exchange.create_market_order(
                        symbol=symbol,
                        side='buy' if signal.direction == 'buy' else 'sell',
                        amount=qty,
                        params={
                            'category': 'linear',
                            'reduceOnly': False
                        }
                    )
                
                # 🔍 ОТСЛЕЖИВАНИЕ ФАКТИЧЕСКОЙ ЦЕНЫ ИСПОЛНЕНИЯ
                # После размещения ордера получаем фактическую цену входа
                await asyncio.sleep(1)  # Ждём исполнения
                try:
                    positions = await self.exchange.fetch_positions(params={'category': 'linear', 'symbol': symbol})
                    for pos in positions:
                        if float(pos.get('size', 0) or pos.get('contracts', 0)) > 0:
                            actual_entry = float(pos.get('entryPrice', 0) or pos.get('avgPrice', 0))
                            if actual_entry > 0:
                                actual_slippage = abs(actual_entry - entry_price) / entry_price * 100 if entry_price > 0 else 0
                                if actual_slippage > 0.05:  # Проскальзывание > 0.05%
                                    logger.warning(f"⚠️ {symbol}: ФАКТИЧЕСКОЕ ПРОСКАЛЬЗЫВАНИЕ {actual_slippage:.3f}% (ожидалось ${entry_price:.5f}, фактически ${actual_entry:.5f})")
                                entry_price = actual_entry  # Используем фактическую цену
                                break
                except Exception as e:
                    logger.debug(f"⚠️ {symbol}: Не удалось получить фактическую цену входа: {e}")
                
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
                
                # 🔴 ПРИОРИТЕТ 1.4: Проверка размера позиции после открытия
                try:
                    await asyncio.sleep(2)  # Ждем полного исполнения
                    
                    # Получаем фактический размер позиции с биржи
                    position_check = await self._retry_critical_operation(
                        lambda: self.exchange.fetch_position(symbol, params={'category': 'linear'}),
                        f"Проверка размера позиции {symbol}",
                        max_retries=3,
                        delay=1.0
                    )
                    
                    if position_check:
                        actual_size = float(position_check.get('contracts', 0) or position_check.get('size', 0) or 0)
                        expected_size = qty
                        
                        # Проверяем, что позиция открыта полностью (допускаем 5% отклонение)
                        if actual_size < expected_size * 0.95:
                            logger.warning(f"⚠️ {symbol}: Частичное исполнение! Ожидалось: {expected_size:.6f}, получено: {actual_size:.6f} ({actual_size/expected_size*100:.1f}%)")
                            
                            # Дозаполняем позицию
                            remaining = expected_size - actual_size
                            if remaining > 0:
                                logger.info(f"🔄 {symbol}: Дозаполняем позицию: {remaining:.6f}")
                                
                                # Используем market order для дозаполнения
                                try:
                                    fill_order = await self._retry_critical_operation(
                                        lambda: self.exchange.create_market_order(
                                            symbol=symbol,
                                            side='buy' if signal.direction == 'buy' else 'sell',
                                            amount=remaining,
                                            params={
                                                'category': 'linear',
                                                'reduceOnly': False
                                            }
                                        ),
                                        f"Дозаполнение позиции {symbol}",
                                        max_retries=3,
                                        delay=2.0
                                    )
                                    
                                    if fill_order:
                                        logger.info(f"✅ {symbol}: Позиция дозаполнена до {expected_size:.6f}")
                                        
                                        # Проверяем финальный размер
                                        await asyncio.sleep(1)
                                        final_check = await self.exchange.fetch_position(symbol, params={'category': 'linear'})
                                        if final_check:
                                            final_size = float(final_check.get('contracts', 0) or final_check.get('size', 0) or 0)
                                            logger.info(f"✅ {symbol}: Финальный размер позиции: {final_size:.6f} (ожидалось: {expected_size:.6f})")
                                    else:
                                        logger.warning(f"⚠️ {symbol}: Не удалось дозаполнить позицию, продолжаем с текущим размером")
                                except Exception as e:
                                    logger.warning(f"⚠️ {symbol}: Ошибка дозаполнения позиции: {e}, продолжаем с текущим размером")
                        else:
                            logger.info(f"✅ {symbol}: Позиция открыта полностью | Размер: {actual_size:.6f} (ожидалось: {expected_size:.6f})")
                    else:
                        logger.warning(f"⚠️ {symbol}: Не удалось проверить размер позиции, продолжаем...")
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Ошибка проверки размера позиции: {e}, продолжаем...")
                
                # 4. Устанавливаем Stop Loss и Take Profit через правильный метод
                # ⚠️ ОТКЛЮЧЕНО: HighPerformanceTradingSystem (упрощение архитектуры)
                # Используем стандартные TP/SL параметры для всех сделок
                high_potential = getattr(signal, 'high_potential_data', None) or getattr(signal, 'potential_percent', None)
                
                # Обычные параметры (стандартные для всех сделок)
                stop_loss_price = signal.stop_loss
                
                # 🔴 ИЗМЕНЕНО: ТОЛЬКО TP1 = +1.15% (закрывает 100% позиции, компенсация комиссии)
                # Пользователь подтвердил: достаточно одного TP
                tp_percent = 1.15  # TP1: +1.15% (закрывает 100% позиции)
                if signal.direction == 'buy':
                    tp_price = entry_price * (1 + tp_percent / 100.0)
                else:
                    tp_price = entry_price * (1 - tp_percent / 100.0)
                tp_prices = [tp_price]  # Только TP1 = +1.15%
                
                # Используем новую функцию для установки SL/TP
                sl_tp_set = await self._set_position_sl_tp_bybit(
                    symbol=symbol,
                    side=signal.direction,
                    size=qty,
                    stop_loss_price=stop_loss_price,
                    take_profit_prices=tp_prices
                )
                
                # 🚨 КРИТИЧЕСКАЯ ПРОВЕРКА: Проверяем, что SL действительно установлен на бирже
                if not sl_tp_set:
                    logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА! SL/TP НЕ УСТАНОВЛЕНЫ НА БИРЖЕ!")
                    logger.error(f"   Позиция открыта БЕЗ ЗАЩИТЫ! Мониторинг будет проверять SL каждые 10 секунд.")
                    # ✅ ЗАДАЧА #3: Добавляем в список заблокированных символов
                    self.sl_tp_failed_symbols.add(symbol)
                    logger.error(f"🚨 {symbol}: ВХОД В НОВЫЕ ПОЗИЦИИ ЗАБЛОКИРОВАН до успешной установки SL/TP!")
                    # Пытаемся установить SL/TP еще раз через 5 секунд
                    asyncio.create_task(self._retry_set_sl_tp(symbol, signal.direction, entry_price, stop_loss_price, tp_prices, qty))
                else:
                    logger.info(f"✅ {symbol}: SL/TP установлены на бирже (стартовые: TP 2.5% ROE 50%, SL -$0.6). Монитор будет обновлять TP до 4%, 5%, 6%.")
                    # ✅ ЗАДАЧА #3: Убираем из списка заблокированных символов
                    self.sl_tp_failed_symbols.discard(symbol)
                    # Проверяем, что SL действительно установлен на бирже (через 3 секунды после открытия)
                    await asyncio.sleep(3)
                    sl_verified = await self._verify_sl_tp_on_exchange(symbol, stop_loss_price, tp_prices[0] if tp_prices else None)
                    if not sl_verified:
                        logger.error(f"🚨 {symbol}: SL НЕ ПОДТВЕРЖДЕН НА БИРЖЕ! Повторная попытка установки...")
                        # ✅ ЗАДАЧА #3: Добавляем в список заблокированных символов
                        self.sl_tp_failed_symbols.add(symbol)
                        asyncio.create_task(self._retry_set_sl_tp(symbol, signal.direction, entry_price, stop_loss_price, tp_prices, qty))
                    else:
                        # ✅ ЗАДАЧА #3: Убираем из списка заблокированных символов при успешной верификации
                        self.sl_tp_failed_symbols.discard(symbol)
                
                # 5. Сохраняем информацию о позиции
                # 🚀 Сохраняем информацию о высоком потенциале для больших движений
                high_potential = getattr(signal, 'high_potential_data', None) or getattr(signal, 'potential_percent', None)
                potential_percent = 0
                if high_potential:
                    if isinstance(high_potential, dict):
                        potential_percent = high_potential.get('potential_percent', 0)
                    elif isinstance(high_potential, (int, float)):
                        potential_percent = high_potential
                
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
                    'position_notional': position_notional,  # $20
                    'max_loss_usd': self.MAX_STOP_LOSS_USD,  # $0.15 (~0.75% от позиции)
                    'high_potential': potential_percent >= 30,  # 🚀 Флаг большого движения
                    'potential_percent': potential_percent,  # 🚀 Потенциал роста
                    # 🎯 TRAILING STOP ORDER: Поля для скользящего стоп-ордера (как на фото)
                    'trailing_stop_active': False,  # Активирован ли trailing stop
                    'activation_price': None,  # Цена активации (когда цена достигает этого уровня, trailing stop активируется)
                    'highest_price': entry_price if signal.direction == 'buy' else None,  # Максимальная цена для LONG
                    'lowest_price': entry_price if signal.direction == 'sell' else None,  # Минимальная цена для SHORT
                    'correction_level_pct': 0.5,  # Уровень коррекции в % (по умолчанию 0.5%)
                    'limit_order_id': None,  # ID лимитного ордера (если используется)
                    'limit_order_price': None  # Цена лимитного ордера
                }
                
                # 6. Обновляем статистику открытия позиций
                self.performance_stats['positions_opened'] = self.performance_stats.get('positions_opened', 0) + 1
                self.performance_stats['signals_executed'] = self.performance_stats.get('signals_executed', 0) + 1
                # Добавляем символ в список торгуемых
                if 'symbols_traded' in self.performance_stats:
                    self.performance_stats['symbols_traded'].add(symbol)
                
                # ИЗМЕНЕНО: Сохраняем сделку в БД
                # ⚠️ ОТКЛЮЧЕНО: DataStorageSystem (упрощение архитектуры)
                if False and self.data_storage:  # Отключено
                    try:
                        from data_storage_system import TradeDecision
                        trade_decision = TradeDecision(
                            timestamp=datetime.now(WARSAW_TZ).isoformat(),
                            symbol=symbol,
                            decision=signal.direction,
                            confidence=signal.confidence,
                            strategy_score=signal.strategy_score,
                            reasons=signal.reasons,
                            market_data={
                                'entry_price': entry_price,
                                'stop_loss': stop_loss_price,
                                'tp_levels': [{'level': tp.level, 'price': tp.price, 'percent': tp.percent} for tp in signal.tp_levels],
                                'leverage': self.LEVERAGE,
                                'position_size': qty,
                                'market_condition': signal.market_condition
                            },
                            result='pending',
                            pnl_percent=None,
                            entry_price=entry_price,
                            exit_price=None
                        )
                        self.data_storage.store_trade_decision(trade_decision)
                        logger.info(f"💾 {symbol}: Сделка сохранена в БД")
                    except Exception as e:
                        logger.error(f"❌ {symbol}: Ошибка сохранения сделки в БД: {e}")
                
                # ИЗМЕНЕНО: Отправляем Telegram уведомление об открытии позиции
                if self.telegram_bot:
                    try:
                        await self.send_position_opened_v4(
                            symbol=symbol,
                            side=signal.direction,
                            entry_price=entry_price,
                            amount_usdt=position_notional,
                            confidence=signal.confidence,
                            strategy_score=signal.strategy_score
                        )
                        logger.info(f"✅ {symbol}: Telegram уведомление об открытии отправлено успешно")
                    except Exception as e:
                        logger.error(f"❌ {symbol}: Ошибка отправки Telegram уведомления об открытии: {e}")
                        import traceback
                        logger.error(f"   Traceback: {traceback.format_exc()}")
                else:
                    logger.warning(f"⚠️ {symbol}: Telegram бот не инициализирован, уведомление не отправлено")
                
                logger.info(f"✅ {symbol}: Позиция успешно открыта! | Размер: {qty:.6f} | SL: ${stop_loss_price:.4f}")
                return True
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Ошибка открытия позиции: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка автоматического открытия позиции для {signal.symbol}: {e}", exc_info=True)
            return False
    
    async def check_telegram_exchange_sync(self) -> Dict[str, Any]:
        """
        🔍 Проверка соответствия позиций в Telegram и на бирже
        
        Returns:
            Dict с результатами проверки
        """
        result = {
            'status': 'ok',
            'issues': [],
            'details': []
        }
        
        try:
            # Получаем позиции с биржи
            exchange_positions = await self.exchange.fetch_positions(params={'category': 'linear'})
            exchange_open = [p for p in exchange_positions if float(p.get('size', 0) or p.get('contracts', 0)) > 0]
            
            # Получаем позиции из active_positions
            bot_positions = list(self.active_positions.keys())
            
            result['details'].append(f'Позиций на бирже: {len(exchange_open)}')
            result['details'].append(f'Позиций в active_positions: {len(bot_positions)}')
            
            # Проверяем расхождения
            exchange_symbols = {self.normalize_symbol(p.get('symbol', '')) for p in exchange_open}
            bot_symbols = {self.normalize_symbol(s) for s in bot_positions}
            
            # Позиции на бирже, но не в active_positions
            missing_in_bot = exchange_symbols - bot_symbols
            if missing_in_bot:
                result['status'] = 'warning'
                result['issues'].append(f'Позиции на бирже, но не в active_positions: {missing_in_bot}')
                # Синхронизируем
                for pos in exchange_open:
                    symbol = pos.get('symbol', '')
                    if self.normalize_symbol(symbol) in missing_in_bot:
                        self.active_positions[symbol] = {
                            'side': pos.get('side', ''),
                            'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                            'size': pos.get('size', 0) or pos.get('contracts', 0),
                            'pnl_percent': pos.get('percentage', 0),
                            'opened_at': datetime.now(WARSAW_TZ)
                        }
                        logger.warning(f"🔄 Синхронизирована позиция {symbol} с биржи")
            
            # Позиции в active_positions, но не на бирже
            missing_on_exchange = bot_symbols - exchange_symbols
            if missing_on_exchange:
                result['status'] = 'warning'
                result['issues'].append(f'Позиции в active_positions, но не на бирже: {missing_on_exchange}')
                # Удаляем из active_positions
                for symbol in list(self.active_positions.keys()):
                    if self.normalize_symbol(symbol) in missing_on_exchange:
                        del self.active_positions[symbol]
                        logger.warning(f"🔄 Удалена позиция {symbol} из active_positions (не найдена на бирже)")
            
            if not missing_in_bot and not missing_on_exchange:
                result['details'].append('✅ Синхронизация: OK')
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'Ошибка проверки: {e}')
        
        return result
    
    async def monitor_positions(self):
        """
        📊 Мониторинг открытых позиций и автоматическое закрытие по TP/SL
        ✅ НОВОЕ: Проверка closed PnL для обнаружения закрытых позиций
        """
        try:
            if not self.exchange:
                return
            
            # 🔍 ПРОВЕРКА СИНХРОНИЗАЦИИ TELEGRAM ↔ БИРЖА (каждые 10 циклов)
            if not hasattr(self, '_sync_check_counter'):
                self._sync_check_counter = 0
            self._sync_check_counter += 1
            if self._sync_check_counter >= 10:
                sync_result = await self.check_telegram_exchange_sync()
                if sync_result['status'] != 'ok':
                    logger.warning(f"⚠️ Проблемы синхронизации: {sync_result['issues']}")
                self._sync_check_counter = 0
            
            # === НОВОЕ: Проверка закрытых позиций через closed PnL ===
            try:
                from pybit.unified_trading import HTTP
                session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False, recv_window=5000, timeout=10)
                
                # Проверяем все отслеживаемые позиции на закрытие
                for symbol, pos_info in list(self.active_positions.items()):
                    try:
                        # Нормализуем символ для Bybit API
                        bybit_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                        bybit_symbol = bybit_symbol.replace('/', '').replace(':', '')
                        
                        # Получаем последние закрытые позиции
                        cp = session.get_closed_pnl(category='linear', symbol=bybit_symbol, limit=5)
                        items = cp.get('result', {}).get('list', []) or []
                        
                        if items:
                            # Берем самую свежую закрытую позицию
                            latest_closed = items[0]
                            updated_time = latest_closed.get('updatedTime', 0)
                            
                            # Проверяем, была ли позиция закрыта недавно (за последние 5 минут)
                            if updated_time:
                                try:
                                    # Преобразуем timestamp в datetime
                                    # Исправление: updated_time может быть int (миллисекунды) или строкой
                                    try:
                                        if isinstance(updated_time, (int, float)):
                                            # Если это число, это миллисекунды
                                            closed_dt = datetime.fromtimestamp(int(updated_time) / 1000, tz=WARSAW_TZ)
                                        elif isinstance(updated_time, str):
                                            # Если строка, пробуем распарсить
                                            try:
                                                # Пробуем как ISO формат
                                                closed_dt = datetime.fromisoformat(updated_time.replace('Z', '+00:00'))
                                            except:
                                                # Если не получилось, пробуем как число в строке
                                                closed_dt = datetime.fromtimestamp(int(float(updated_time)) / 1000, tz=WARSAW_TZ)
                                        else:
                                            continue
                                    except Exception as e:
                                        logger.debug(f"⚠️ Ошибка преобразования updated_time для {symbol}: {e}, тип: {type(updated_time)}")
                                        continue
                                    
                                    now = datetime.now(WARSAW_TZ)
                                    time_diff = (now - closed_dt).total_seconds()
                                    
                                    # 🔴 КРИТИЧЕСКАЯ ЗАЩИТА: Проверяем, что позиция действительно была открыта нами
                                    # Сравниваем время закрытия с временем открытия из active_positions
                                    pos_info_check = self.active_positions.get(symbol)
                                    if not pos_info_check:
                                        # Если позиции нет в active_positions, это не наша позиция
                                        logger.debug(f"⚠️ {symbol}: Пропускаем закрытие (позиция не найдена в active_positions) - это не наша позиция")
                                        continue
                                    
                                    opened_at = pos_info_check.get('opened_at')
                                    if not opened_at:
                                        # Если нет времени открытия, пропускаем (защита от ложных срабатываний)
                                        logger.debug(f"⚠️ {symbol}: Пропускаем закрытие (нет времени открытия в active_positions)")
                                        continue
                                    
                                    # Преобразуем opened_at в datetime если нужно
                                    if isinstance(opened_at, str):
                                        try:
                                            opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                                        except:
                                            logger.debug(f"⚠️ {symbol}: Не удалось распарсить opened_at: {opened_at}")
                                            continue
                                    
                                    if isinstance(opened_at, datetime):
                                        if opened_at.tzinfo is None:
                                            opened_at = WARSAW_TZ.localize(opened_at)
                                        
                                        # КРИТИЧЕСКАЯ ПРОВЕРКА: Позиция должна быть открыта ДО закрытия
                                        if closed_dt < opened_at:
                                            logger.debug(f"⚠️ {symbol}: Пропускаем закрытие (время закрытия {closed_dt} раньше времени открытия {opened_at}) - это старая позиция")
                                            continue
                                        
                                        # КРИТИЧЕСКАЯ ПРОВЕРКА: Позиция должна быть открыта недавно (не более 15 минут назад)
                                        open_time_diff = (now - opened_at).total_seconds()
                                        if open_time_diff > 900:  # Более 15 минут
                                            logger.debug(f"⚠️ {symbol}: Пропускаем закрытие (позиция открыта {open_time_diff:.0f} сек назад) - возможно, это не наша позиция")
                                            continue
                                        
                                        # КРИТИЧЕСКАЯ ПРОВЕРКА: Позиция должна быть открыта минимум 30 секунд
                                        # Это предотвращает обработку закрытий сразу после открытия (ликвидации, ошибки)
                                        if open_time_diff < 30:  # Меньше 30 секунд
                                            logger.warning(f"🚨 {symbol}: ПРОПУСКАЕМ закрытие - позиция открыта всего {open_time_diff:.0f} сек назад (возможно ликвидация или ошибка)")
                                            continue
                                        
                                        # КРИТИЧЕСКАЯ ПРОВЕРКА: Время между открытием и закрытием должно быть разумным
                                        position_duration = (closed_dt - opened_at).total_seconds()
                                        if position_duration < 30:  # Позиция закрылась менее чем через 30 секунд после открытия
                                            logger.warning(f"🚨 {symbol}: ПРОПУСКАЕМ закрытие - позиция закрылась через {position_duration:.0f} сек после открытия (вероятно ликвидация или ошибка)")
                                            continue
                                    
                                    # ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: Не обрабатываем закрытия, которые произошли менее 3 минут назад
                                    # Это предотвращает ложные срабатывания сразу после открытия позиции
                                    if time_diff < 180:  # Меньше 3 минут - пропускаем (было 2 минуты)
                                        logger.warning(f"🚨 {symbol}: ПРОПУСКАЕМ закрытие (слишком свежее: {time_diff:.0f} сек) - защита от ложных срабатываний")
                                        continue
                                    
                                    # Если позиция закрыта недавно (за последние 5 минут) и мы еще не обработали
                                    if time_diff < 300:  # 5 минут
                                        closed_pnl = float(latest_closed.get('closedPnl', 0))
                                        avg_entry_raw = latest_closed.get('avgEntryPrice')
                                        avg_exit_raw = latest_closed.get('avgExitPrice')
                                        ex_side = latest_closed.get('side', 'Buy')
                                        
                                        # Получаем данные из active_positions для fallback
                                        pos_info_check = self.active_positions.get(symbol, {})
                                        entry_price = pos_info_check.get('entry_price', 0)
                                        side = pos_info_check.get('side', 'Buy')
                                        qty = pos_info_check.get('qty', 0)
                                        
                                        # Используем данные из API только если они валидны
                                        avg_entry = float(avg_entry_raw) if avg_entry_raw and float(avg_entry_raw) > 0 else entry_price
                                        avg_exit = float(avg_exit_raw) if avg_exit_raw and float(avg_exit_raw) > 0 else None
                                        
                                        # Если цена выхода не получена из API, используем текущую цену
                                        if avg_exit is None or avg_exit == 0 or avg_exit == avg_entry:
                                            try:
                                                ticker = await self.exchange.fetch_ticker(symbol)
                                                current_mark = float(ticker.get('last') or ticker.get('close') or 0)
                                                if current_mark > 0 and current_mark != avg_entry:
                                                    avg_exit = current_mark
                                                    logger.info(f"✅ {symbol}: Использована текущая цена как цена выхода: {avg_exit:.5f} (вместо невалидной из API)")
                                                else:
                                                    logger.error(f"🚨 {symbol}: Текущая цена тоже невалидна ({current_mark}) или равна entry ({avg_entry:.5f})")
                                                    # НЕ используем fallback на entry - это скрывает проблему
                                                    # Используем closedPnl для расчета exit_price
                                                    if closed_pnl != 0 and avg_entry > 0:
                                                        # Рассчитываем exit_price из closedPnl
                                                        if side == 'buy':
                                                            avg_exit = avg_entry * (1 + closed_pnl / (avg_entry * qty)) if qty > 0 else avg_entry
                                                        else:
                                                            avg_exit = avg_entry * (1 - closed_pnl / (avg_entry * qty)) if qty > 0 else avg_entry
                                                        logger.info(f"✅ {symbol}: Exit цена рассчитана из closedPnl: {avg_exit:.5f}")
                                            except Exception as e:
                                                logger.error(f"❌ {symbol}: Не удалось получить текущую цену: {e}")
                                                # Используем closedPnl для расчета exit_price
                                                if closed_pnl != 0 and avg_entry > 0 and qty > 0:
                                                    if ex_side == 'Buy':
                                                        avg_exit = avg_entry * (1 + closed_pnl / (avg_entry * qty))
                                                    else:
                                                        avg_exit = avg_entry * (1 - closed_pnl / (avg_entry * qty))
                                                    logger.info(f"✅ {symbol}: Exit цена рассчитана из closedPnl (fallback): {avg_exit:.5f}")
                                                else:
                                                    logger.error(f"🚨 {symbol}: Невозможно рассчитать exit_price! Entry: {avg_entry}, PnL: {closed_pnl}, Qty: {qty}")
                                                    # Критическая ошибка - не отправляем сообщение с невалидными данными
                                                    continue
                                        
                                        side = 'buy' if ex_side == 'Buy' else 'sell'
                                        position_notional = float(latest_closed.get('qty', 0)) * avg_entry if avg_entry > 0 else self.POSITION_NOTIONAL
                                        pnl_percent = (closed_pnl / position_notional) * 100 if position_notional > 0 else 0
                                        
                                        logger.info(f"✅ {symbol}: Обнаружено закрытие через closed PnL | Entry: {avg_entry:.5f}, Exit: {avg_exit:.5f}, PnL=${closed_pnl:.2f} ({pnl_percent:.2f}%)")
                                        
                                        # Отправляем уведомление
                                        if self.telegram_bot:
                                            try:
                                                await self.send_position_closed_v4(
                                                    symbol=symbol,
                                                    side=side,
                                                    entry_price=avg_entry,
                                                    exit_price=avg_exit,
                                                    pnl_percent=pnl_percent,
                                                    profit_usd=closed_pnl,
                                                    reason="Закрыта на бирже (обнаружено через closed PnL)"
                                                )
                                            except Exception as e:
                                                logger.error(f"⚠️ Ошибка отправки Telegram для {symbol}: {e}")
                                        
                                        # 📊 Обновляем метрики производительности
                                        try:
                                            duration_seconds = position_duration if 'position_duration' in locals() else None
                                            timeframe = pos_info_check.get('timeframe') if pos_info_check else None
                                            self._update_performance_metrics(
                                                pnl_usd=closed_pnl,
                                                pnl_percent=pnl_percent,
                                                symbol=symbol,
                                                duration_seconds=duration_seconds,
                                                timeframe=timeframe
                                            )
                                        except Exception as e:
                                            logger.error(f"⚠️ Ошибка обновления метрик для {symbol}: {e}")
                                        
                                        # Удаляем из активных позиций (с нормализацией)
                                        symbol_norm = self.normalize_symbol(symbol)
                                        for key in list(self.active_positions.keys()):
                                            if self.normalize_symbol(key) == symbol_norm:
                                                del self.active_positions[key]
                                                break
                                        
                                        # Отслеживаем убыточные монеты
                                        if closed_pnl < -0.5:
                                            symbol_norm_loss = self.normalize_symbol(symbol)
                                            self.losing_symbols[symbol_norm_loss] = (abs(closed_pnl), datetime.now(WARSAW_TZ))
                                            logger.warning(f"⚠️ {symbol}: Добавлена в список убыточных монет (cooldown 12ч). Потеря: {closed_pnl:.2f} USDT")
                                        
                                        # 🧠 АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ: Обучение сразу после каждой сделки
                                        # ⚠️ ОТКЛЮЧЕНО: UniversalLearningSystem
                                        if False and self.universal_learning:  # Отключено
                                            try:
                                                # Получаем данные рынка на момент закрытия
                                                market_data = {
                                                    'symbol': symbol,
                                                    'side': side,
                                                    'entry_price': avg_entry,
                                                    'exit_price': avg_exit,
                                                    'pnl': closed_pnl,
                                                    'pnl_percent': pnl_percent,
                                                    'market_condition': getattr(self, '_current_market_condition', 'NEUTRAL'),
                                                    'confidence': pos_info_check.get('confidence', 0) if pos_info_check else 0
                                                }
                                                decision = 'buy' if side.lower() == 'buy' or side.lower() == 'long' else 'sell'
                                                result = 'success' if closed_pnl > 0 else 'failure'
                                                
                                                # ⚠️ ОТКЛЮЧЕНО: UniversalLearningSystem (упрощение архитектуры)
                                                # Обучение происходит только через Disco57 (PPO Agent)
                                                # if self.universal_learning:
                                                #     self.universal_learning.learn_from_decision(market_data, decision, result)
                                                logger.debug(f"ℹ️ {symbol}: Обучение через Disco57 (PPO Agent) | Решение: {decision.upper()}, Результат: {result}, PnL: ${closed_pnl:.2f}")
                                            except Exception as e:
                                                logger.error(f"⚠️ {symbol}: Ошибка обучения на сделке: {e}")
                                    # Конец блока if time_diff < 300
                                except Exception as e:
                                    logger.debug(f"⚠️ Ошибка обработки closed PnL для {symbol}: {e}")
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка проверки closed PnL для {symbol}: {e}")
            except Exception as e:
                logger.debug(f"⚠️ Ошибка инициализации проверки closed PnL: {e}")
            
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
                    
                    # 🚨 КРИТИЧЕСКАЯ ПРОВЕРКА: Если позиция есть на бирже, но нет SL/TP - устанавливаем!
                    if pos_info:
                        stop_loss_on_exchange = position.get('stopLoss') or position.get('stop_loss')
                        take_profit_on_exchange = position.get('takeProfit') or position.get('take_profit')
                        
                        # Проверяем, установлены ли SL/TP на бирже
                        if not stop_loss_on_exchange or stop_loss_on_exchange == '' or stop_loss_on_exchange == '0':
                            logger.warning(f"🚨 {symbol}: SL НЕ УСТАНОВЛЕН НА БИРЖЕ! Устанавливаем...")
                            # Используем функцию добавления SL/TP
                            side_str = 'sell' if side.lower() in ['sell', 'short'] else 'buy'
                            sl_tp_added = await self.add_sl_tp_to_existing_position(symbol, side_str, float(entry_price))
                            if sl_tp_added:
                                logger.info(f"✅ {symbol}: SL/TP успешно установлены через add_sl_tp_to_existing_position")
                            else:
                                logger.error(f"❌ {symbol}: Не удалось установить SL/TP! Позиция БЕЗ ЗАЩИТЫ!")
                    
                    # ⏰ ОГРАНИЧЕНИЕ ВРЕМЕНИ УДЕРЖАНИЯ: 24 часа максимум
                    # Проверяем для ВСЕХ позиций, даже если их нет в active_positions
                    opened_at = None
                    if pos_info:
                        opened_at = pos_info.get('opened_at')
                    else:
                        # Если позиция не в active_positions, получаем время открытия из данных биржи
                        created_time = position.get('createdTime') or position.get('updatedTime')
                        if created_time:
                            try:
                                from datetime import datetime
                                import pytz
                                if isinstance(created_time, (int, float)):
                                    opened_at = datetime.fromtimestamp(int(created_time) / 1000, tz=pytz.timezone('Europe/Warsaw'))
                                elif isinstance(created_time, str):
                                    opened_at = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                            except Exception as e:
                                logger.debug(f"⚠️ {symbol}: Не удалось распарсить createdTime: {e}")
                    
                    # Если позиция есть на бирже, но не в нашем словаре - всё равно проверяем 24ч лимит
                    if not pos_info:
                        # Проверяем только время удержания для таких позиций
                        if opened_at:
                            from datetime import datetime, timedelta
                            import pytz
                            if isinstance(opened_at, str):
                                try:
                                    opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                                except:
                                    opened_at = datetime.now(pytz.timezone('Europe/Warsaw'))
                            if isinstance(opened_at, datetime):
                                if opened_at.tzinfo is None:
                                    opened_at = pytz.timezone('Europe/Warsaw').localize(opened_at)
                                
                                now = datetime.now(pytz.timezone('Europe/Warsaw'))
                                hold_duration = now - opened_at
                                max_hold_time = timedelta(hours=24)
                                
                                if hold_duration >= max_hold_time:
                                    logger.warning(f"⏰ {symbol}: Позиция удерживается {hold_duration} (максимум 24ч) — закрываем автоматически (не отслеживалась ботом)")
                                    try:
                                        close_side = 'sell' if side.lower() == 'long' else 'buy'
                                        await self.exchange.create_market_order(
                                            symbol=symbol,
                                            side=close_side,
                                            amount=size,
                                            params={'category': 'linear', 'reduceOnly': True}
                                        )
                                        # Рассчитываем PnL
                                        if side.lower() == 'long':
                                            pnl_percent_temp = ((current_price - entry_price) / entry_price) * 100
                                        else:
                                            pnl_percent_temp = ((entry_price - current_price) / entry_price) * 100
                                        
                                        if self.telegram_bot:
                                            await self.send_position_closed_v4(
                                                symbol=symbol,
                                                side='buy' if side.lower() == 'long' else 'sell',
                                                entry_price=float(entry_price),
                                                exit_price=float(current_price),
                                                pnl_percent=pnl_percent_temp,
                                                profit_usd=pnl_percent_temp / 100 * self.POSITION_NOTIONAL,
                                                reason=f'Авто-закрытие: превышено время удержания ({hold_duration})'
                                            )
                                        # 📊 Обновляем метрики производительности
                                        try:
                                            duration_seconds = hold_duration.total_seconds() if isinstance(hold_duration, timedelta) else None
                                            self._update_performance_metrics(
                                                pnl_usd=pnl_percent_temp / 100 * self.POSITION_NOTIONAL,
                                                pnl_percent=pnl_percent_temp,
                                                symbol=symbol,
                                                duration_seconds=duration_seconds
                                            )
                                        except Exception as e:
                                            logger.error(f"⚠️ Ошибка обновления метрик для {symbol}: {e}")
                                        logger.info(f"✅ {symbol}: Позиция закрыта по времени удержания")
                                    except Exception as e:
                                        logger.error(f"❌ {symbol}: Ошибка закрытия по времени: {e}")
                                    continue
                        # Если не нужно закрывать, пропускаем позицию, которая не отслеживается
                        continue
                    
                    signal = pos_info.get('signal')
                    if not signal:
                        continue
                    
                    # ⏰ ОГРАНИЧЕНИЕ ВРЕМЕНИ УДЕРЖАНИЯ: 24 часа максимум (для отслеживаемых позиций)
                    if opened_at:
                        from datetime import datetime, timedelta
                        import pytz
                        if isinstance(opened_at, str):
                            # Если это строка, парсим её
                            try:
                                opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                            except:
                                opened_at = datetime.now(pytz.timezone('Europe/Warsaw'))
                        if isinstance(opened_at, datetime):
                            # Преобразуем в aware datetime если нужно
                            if opened_at.tzinfo is None:
                                opened_at = pytz.timezone('Europe/Warsaw').localize(opened_at)
                            
                            now = datetime.now(pytz.timezone('Europe/Warsaw'))
                            hold_duration = now - opened_at
                            max_hold_time = timedelta(hours=24)
                            
                            if hold_duration >= max_hold_time:
                                logger.warning(f"⏰ {symbol}: Позиция удерживается {hold_duration} (максимум 24ч) — закрываем автоматически")
                                try:
                                    await self.exchange.create_market_order(
                                        symbol=symbol,
                                        side='sell' if (side.lower() == 'long' or signal.direction == 'buy') else 'buy',
                                        amount=size,
                                        params={'category': 'linear', 'reduceOnly': True}
                                    )
                                    # Рассчитываем PnL перед отправкой сообщения
                                    if side.lower() == 'long' or signal.direction == 'buy':
                                        pnl_percent_temp = ((current_price - entry_price) / entry_price) * 100
                                    else:
                                        pnl_percent_temp = ((entry_price - current_price) / entry_price) * 100
                                    
                                    if self.telegram_bot:
                                        await self.send_position_closed_v4(
                                            symbol=symbol,
                                            side=signal.direction,
                                            entry_price=float(entry_price),
                                            exit_price=float(current_price),
                                            pnl_percent=pnl_percent_temp,
                                            profit_usd=pnl_percent_temp / 100 * (pos_info.get('position_notional', self.POSITION_NOTIONAL)),
                                            reason=f'Авто-закрытие: превышено время удержания ({hold_duration})'
                                        )
                                        # 📊 Обновляем метрики производительности
                                        try:
                                            duration_seconds = hold_duration.total_seconds() if isinstance(hold_duration, timedelta) else None
                                            timeframe = pos_info.get('timeframe') if pos_info else None
                                            self._update_performance_metrics(
                                                pnl_usd=pnl_percent_temp / 100 * (pos_info.get('position_notional', self.POSITION_NOTIONAL)),
                                                pnl_percent=pnl_percent_temp,
                                                symbol=symbol,
                                                duration_seconds=duration_seconds,
                                                timeframe=timeframe
                                            )
                                        except Exception as e:
                                            logger.error(f"⚠️ Ошибка обновления метрик для {symbol}: {e}")
                                    logger.info(f"✅ {symbol}: Позиция закрыта по времени удержания")
                                except Exception as e:
                                    logger.error(f"❌ {symbol}: Ошибка закрытия по времени: {e}")
                                # Удаляем позицию из словаря (с нормализацией)
                                symbol_norm = self.normalize_symbol(symbol)
                                for key in list(self.active_positions.keys()):
                                    if self.normalize_symbol(key) == symbol_norm:
                                        del self.active_positions[key]
                                        break
                                continue
                    
                    # Рассчитываем текущий PnL
                    if side.lower() == 'long' or signal.direction == 'buy':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # 🔴 ОТКЛЮЧЕНО: MTF валидация 30m+1h+4h+1D - может закрывать позиции сразу после открытия
                    # ВАЖНО: Эта проверка отключена, так как она закрывала позиции сразу после открытия
                    # Если нужно включить, добавьте проверку времени открытия (минимум 5 минут)
                    # try:
                    #     # Проверяем, что позиция открыта минимум 5 минут перед проверкой MTF
                    #     pos_info_check = self.active_positions.get(symbol)
                    #     if pos_info_check:
                    #         opened_at_check = pos_info_check.get('opened_at')
                    #         if opened_at_check:
                    #             if isinstance(opened_at_check, str):
                    #                 try:
                    #                     opened_at_check = datetime.fromisoformat(opened_at_check.replace('Z', '+00:00'))
                    #                 except:
                    #                     opened_at_check = None
                    #             if isinstance(opened_at_check, datetime):
                    #                 if opened_at_check.tzinfo is None:
                    #                     opened_at_check = WARSAW_TZ.localize(opened_at_check)
                    #                 now_check = datetime.now(WARSAW_TZ)
                    #                 time_since_open = (now_check - opened_at_check).total_seconds()
                    #                 if time_since_open < 300:  # Меньше 5 минут - пропускаем проверку
                    #                     logger.debug(f"⏭️ {symbol}: Пропускаем MTF проверку (позиция открыта {time_since_open:.0f} сек назад)")
                    #                     pass  # Пропускаем проверку для новых позиций
                    #                 else:
                    #                     # Выполняем проверку только для позиций старше 5 минут
                    #                     mtf_data_live = await self._fetch_multi_timeframe_data(symbol)
                    #                     c30 = mtf_data_live.get('30m', {}) or {}
                    #                     c1h = mtf_data_live.get('1h', {}) or {}
                    #                     c4h = mtf_data_live.get('4h', {}) or {}
                    #                     def _confirmed(dir_):
                    #                         if dir_ == 'buy':
                    #                             return (
                    #                                 c45.get('ema_9', 0) > c45.get('ema_21', 0)
                    #                                 and c1h.get('ema_9', 0) > c1h.get('ema_21', 0)
                    #                                 and c4h.get('ema_9', 0) > c4h.get('ema_21', 0)
                    #                             )
                    #                         if dir_ == 'sell':
                    #                             return (
                    #                                 c45.get('ema_9', 0) < c45.get('ema_21', 0)
                    #                                 and c1h.get('ema_9', 0) < c1h.get('ema_21', 0)
                    #                                 and c4h.get('ema_9', 0) < c4h.get('ema_21', 0)
                    #                             )
                    #                         return False
                    #                     intended_dir = 'buy' if (side.lower() == 'long' or signal.direction == 'buy') else 'sell'
                    #                     if not _confirmed(intended_dir):
                    #                         logger.warning(f"🚫 {symbol}: нет подтверждения 30m+1h+4h для {intended_dir.upper()} — закрываем позицию")
                    #                         try:
                    #                             await self.exchange.create_market_order(
                    #                                 symbol=symbol,
                    #                                 side='sell' if intended_dir == 'buy' else 'buy',
                    #                                 amount=size,
                    #                                 params={'category': 'linear', 'reduceOnly': True}
                    #                             )
                    #                             if self.telegram_bot:
                    #                                 await self.send_position_closed_v4(
                    #                                     symbol=symbol,
                    #                                     side=intended_dir,
                    #                                     entry_price=float(entry_price),
                    #                                     exit_price=float(current_price),
                    #                                     pnl_percent=0.0,
                    #                                     profit_usd=0.0,
                    #                                     reason='Авто-закрытие: нет подтверждения 30m+1h+4h'
                    #                                 )
                    #                         except Exception as e:
                    #                             logger.error(f"❌ {symbol}: Ошибка авто-закрытия без подтверждения: {e}")
                    #                         # Удаляем и переходим к следующей позиции
                    #                         if symbol in self.active_positions:
                    #                             del self.active_positions[symbol]
                    #                         continue
                    # except Exception as _:
                    #     pass

                    # Рассчитываем текущий PnL
                    if side.lower() == 'long' or signal.direction == 'buy':
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # 🔴 ИСПРАВЛЕНО: Break-even SL устанавливается ДО достижения TP1, а не при достижении
                    # ЛОГИКА: BE SL нужен только пока позиция открыта, при полном закрытии TP1 он не нужен
                    tp1_reached_flag = pos_info.get('tp1_reached', False)
                    break_even_sl_set = pos_info.get('break_even_sl_set', False)
                    
                    # Break-even SL устанавливается при +0.5-1% прибыли, но ДО достижения TP1 (+1.15%)
                    # При достижении TP1 позиция закрывается на 100%, поэтому BE SL не нужен
                    if 0.5 <= pnl_percent < 1.15 and not tp1_reached_flag:
                        # Проверяем, нужно ли установить или обновить break-even SL
                        now = datetime.now(WARSAW_TZ)
                        last_be_update = pos_info.get('last_break_even_update')
                        should_update = False
                        
                        if not break_even_sl_set:
                            # Первая установка break-even SL при +0.5-1% прибыли
                            should_update = True
                        elif last_be_update:
                            # Проверяем, прошло ли 10 секунд с последнего обновления
                            time_diff = (now - last_be_update).total_seconds()
                            if time_diff >= 10:
                                should_update = True
                        else:
                            # Если нет времени последнего обновления, обновляем
                            should_update = True
                        
                        if should_update:
                            try:
                                current_break_even_sl = pos_info.get('break_even_sl', entry_price)
                                
                                if not break_even_sl_set:
                                    # Первая установка: break-even SL = entry_price
                                    break_even_sl = entry_price
                                    pos_info['break_even_sl'] = break_even_sl
                                    pos_info['break_even_sl_set'] = True
                                    logger.info(f"🛡️ {symbol}: Break-even SL установлен на ${break_even_sl:.6f} при +{pnl_percent:.2f}% прибыли (до TP1)")
                                    
                                    if self.telegram_bot:
                                        try:
                                            await self.send_telegram_v4(
                                                f"🛡️ {symbol}: Break-even SL установлен\n"
                                                f"Прибыль: +{pnl_percent:.2f}%\n"
                                                f"SL: ${break_even_sl:.6f}\n"
                                                f"📈 Трейлинг каждые 10 сек до TP1 (+1.15%)"
                                            )
                                        except Exception:
                                            pass
                                else:
                                    # Трейлинг break-even SL каждые 10 секунд (до достижения TP1)
                                    trailing_distance_pct = 0.3  # 0.3% расстояние для трейлинга
                                    trailing_distance = current_price * (trailing_distance_pct / 100)
                                    
                                    if signal.direction == 'buy':
                                        # Для LONG: двигаем SL вверх вместе с ценой
                                        new_break_even_sl = current_price - trailing_distance
                                        # Не опускаем ниже entry_price
                                        new_break_even_sl = max(new_break_even_sl, entry_price)
                                        # Обновляем только если улучшили позицию
                                        if new_break_even_sl > current_break_even_sl:
                                            break_even_sl = new_break_even_sl
                                        else:
                                            break_even_sl = current_break_even_sl
                                    else:  # sell
                                        # Для SHORT: двигаем SL вниз вместе с ценой
                                        new_break_even_sl = current_price + trailing_distance
                                        # Не поднимаем выше entry_price
                                        new_break_even_sl = min(new_break_even_sl, entry_price)
                                        # Обновляем только если улучшили позицию
                                        if new_break_even_sl < current_break_even_sl or current_break_even_sl == entry_price:
                                            break_even_sl = new_break_even_sl
                                        else:
                                            break_even_sl = current_break_even_sl
                                
                                # Обновляем SL на бирже
                                sl_set_ok = await self._set_sl_tp_pybit(symbol, break_even_sl, None)
                                if sl_set_ok:
                                    pos_info['stop_loss'] = break_even_sl
                                    pos_info['break_even_sl'] = break_even_sl
                                    pos_info['last_break_even_update'] = now
                                    if break_even_sl != entry_price:
                                        logger.debug(f"📈 {symbol}: Break-even SL трейлится: ${break_even_sl:.6f} (прибыль: {pnl_percent:.2f}%)")
                            except Exception as e:
                                logger.debug(f"⚠️ {symbol}: Ошибка break-even SL трейлинга: {e}")

                    # ИЗМЕНЕНО: TP-ЛОГИКА - ТОЛЬКО TP1 РЕАЛИЗОВАН В ПРОДАКШЕНЕ
                    # TP1 (+1.15%): Закрывает 100% позиции (реализовано выше, строки 5068-5331)
                    # TP2-TP6: НЕ РЕАЛИЗОВАНЫ в текущей версии (планируются для будущих обновлений)
                    # 
                    # ПРИЧИНА: Фокус на быстрой фиксации прибыли (скальпинг) через TP1
                    # TP1 обеспечивает компенсацию комиссии и перевод в без убыток
                    # 
                    # КОД ДЛЯ ЧАСТИЧНОГО ЗАКРЫТИЯ (TP2-TP6) НИЖЕ - НЕ ИСПОЛЬЗУЕТСЯ:
                    # Оставлен для будущей реализации многоуровневых TP
                    
                    # ⚠️ ВНИМАНИЕ: Следующий блок кода НЕ АКТИВЕН в текущей версии
                    # Раскомментировать при реализации TP2-TP6
                    """
                    # Проверяем Take Profit уровни (TP2-TP6) - НЕ АКТИВНО
                    tp_levels = pos_info.get('tp_levels', signal.tp_levels)
                    closed_tps = pos_info.get('closed_tps', set())
                    
                    for tp in tp_levels:
                        if tp.level in closed_tps or tp.level == 1:  # TP1 уже обработан выше
                            continue
                        
                        # Проверяем, достигнут ли TP уровень
                        if signal.direction == 'buy':
                            tp_hit = current_price >= tp.price
                        else:
                            tp_hit = current_price <= tp.price
                        
                        if tp_hit:
                            # Закрываем часть позиции (для TP2-TP6)
                            close_percent = tp.close_percent / 100
                            close_size = size * close_percent
                            
                            # ⚠️ КОД ЗАКОММЕНТИРОВАН - TP2-TP6 не реализованы
                            # try:
                            #     close_order = await self.exchange.create_market_order(...)
                            #     ...
                            # except Exception as e:
                            #     logger.error(f"❌ {symbol}: Ошибка закрытия TP{tp.level}: {e}")
                    """
                    
                    # Обновляем информацию о позиции
                    pos_info['current_price'] = current_price
                    pos_info['pnl_percent'] = pnl_percent

                    # 🔴 ИЗМЕНЕНО: ТОЛЬКО TP1 = +1.15% (без трейлинга до других уровней)
                    # Пользователь подтвердил: достаточно одного TP
                    try:
                        import math
                        favorable = pnl_percent if ((signal.direction=='buy' and current_price>=entry_price) or (signal.direction=='sell' and current_price<=entry_price)) else 0.0
                        # Только TP1 = +1.15% (закрывает 100% позиции)
                        target_tp_pct = 1.15  # TP1: 1.15% (закрывает 100% позиции, компенсация комиссии)
                        last_applied = float(pos_info.get('tp_trail_percent', 0.0) or 0.0)
                        tp1_reached = (target_tp_pct >= 1.15 and last_applied < 1.15) or (favorable >= 1.15 and not pos_info.get('tp1_reached', False))
                        
                        if target_tp_pct > last_applied + 1e-9:
                            if signal.direction == 'buy':
                                new_tp_price = entry_price * (1 + target_tp_pct/100.0)
                            else:
                                new_tp_price = entry_price * (1 - target_tp_pct/100.0)
                            py_ok = await self._set_sl_tp_pybit(symbol, None, new_tp_price)
                            if py_ok:
                                pos_info['tp_trail_percent'] = target_tp_pct
                                logger.info(f"🎯 {symbol}: Обновлён TP трейлингом: {target_tp_pct:.1f}% → ${new_tp_price:.6f}")
                                
                                # 🔴 ИСПРАВЛЕНО: ПРИ ДОСТИЖЕНИИ TP1 (+1.15%): Закрываем 100% позиции
                                # Break-even SL НЕ устанавливается, т.к. позиция закрывается полностью
                                if tp1_reached or (target_tp_pct >= 1.15 and not pos_info.get('tp1_reached', False)):
                                    pos_info['tp1_reached'] = True
                                    logger.info(f"🎯 {symbol}: TP1 (+1.15%) достигнут! Закрываем 100% позиции (полный выход, BE SL не нужен)")
                                    
                                    # ИЗМЕНЕНО: Закрываем 100% позиции при достижении TP1
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
                                        
                                        # 🔴 ПРИОРИТЕТ 2.5: Улучшенная проверка статуса ордера после TP1
                                        await asyncio.sleep(2)  # Ждём исполнения ордера
                                        order_id = close_order.get('id') or close_order.get('orderId')
                                        
                                        # Проверяем статус ордера с повторными попытками
                                        order_closed = False
                                        partial_fill = False
                                        remaining_size = 0
                                        
                                        if order_id:
                                            # Проверяем статус ордера с retry логикой
                                            for check_attempt in range(3):
                                                try:
                                                    order_status = await self._retry_critical_operation(
                                                        lambda: self.exchange.fetch_order(order_id, symbol),
                                                        f"Проверка статуса ордера TP1 {symbol}",
                                                        max_retries=2,
                                                        delay=1.0
                                                    )
                                                    
                                                    if order_status:
                                                        filled = float(order_status.get('filled', 0) or 0)
                                                        remaining = float(order_status.get('remaining', 0) or 0)
                                                        order_status_str = order_status.get('status', 'unknown')
                                                        
                                                        if order_status_str in ['closed', 'filled'] and remaining == 0:
                                                            order_closed = True
                                                            logger.info(f"✅ {symbol}: Ордер TP1 полностью исполнен (filled: {filled}, remaining: {remaining})")
                                                            break
                                                        elif filled > 0 and remaining > 0:
                                                            # Частичное исполнение
                                                            partial_fill = True
                                                            remaining_size = remaining
                                                            logger.warning(f"⚠️ {symbol}: Ордер TP1 частично исполнен! Status: {order_status_str}, filled: {filled}, remaining: {remaining}")
                                                            
                                                            # Дозаполняем позицию
                                                            if check_attempt < 2:  # Пробуем еще раз
                                                                await asyncio.sleep(2)
                                                                continue
                                                            else:
                                                                # Последняя попытка - дозаполняем
                                                                logger.info(f"🔄 {symbol}: Дозаполняем позицию: {remaining:.6f}")
                                                                try:
                                                                    fill_order = await self._retry_critical_operation(
                                                                        lambda: self.exchange.create_market_order(
                                                                            symbol=symbol,
                                                                            side='sell' if signal.direction == 'buy' else 'buy',
                                                                            amount=remaining,
                                                                            params={'category': 'linear', 'reduceOnly': True}
                                                                        ),
                                                                        f"Дозаполнение TP1 {symbol}",
                                                                        max_retries=2,
                                                                        delay=2.0
                                                                    )
                                                                    if fill_order:
                                                                        logger.info(f"✅ {symbol}: Позиция дозаполнена до полного закрытия")
                                                                        order_closed = True
                                                                        break
                                                                except Exception as e:
                                                                    logger.warning(f"⚠️ {symbol}: Ошибка дозаполнения: {e}")
                                                        else:
                                                            # Ордер еще не исполнен, ждем
                                                            if check_attempt < 2:
                                                                await asyncio.sleep(2)
                                                                continue
                                                except Exception as e:
                                                    logger.warning(f"⚠️ {symbol}: Не удалось проверить статус ордера {order_id} (попытка {check_attempt + 1}/3): {e}")
                                                    if check_attempt < 2:
                                                        await asyncio.sleep(2)
                                        
                                        # Дополнительная проверка через позиции на бирже
                                        if not order_closed:
                                            try:
                                                positions = await self._retry_critical_operation(
                                                    lambda: self.exchange.fetch_positions(params={'category': 'linear', 'symbol': symbol}),
                                                    f"Проверка позиции TP1 {symbol}",
                                                    max_retries=2,
                                                    delay=1.0
                                                )
                                                
                                                position_still_open = False
                                                if positions:
                                                    for pos in positions:
                                                        pos_size = float(pos.get('contracts', 0) or pos.get('size', 0))
                                                        if pos_size > 0:
                                                            position_still_open = True
                                                            remaining_size = pos_size
                                                            logger.warning(f"⚠️ {symbol}: Позиция ВСЁ ЕЩЁ ОТКРЫТА на бирже! Размер: {pos_size}")
                                                            
                                                            # Пробуем закрыть оставшуюся часть
                                                            if pos_size < size * 0.1:  # Если осталось меньше 10%
                                                                logger.info(f"🔄 {symbol}: Закрываем оставшуюся часть позиции: {pos_size:.6f}")
                                                                try:
                                                                    final_close = await self._retry_critical_operation(
                                                                        lambda: self.exchange.create_market_order(
                                                                            symbol=symbol,
                                                                            side='sell' if signal.direction == 'buy' else 'buy',
                                                                            amount=pos_size,
                                                                            params={'category': 'linear', 'reduceOnly': True}
                                                                        ),
                                                                        f"Финальное закрытие TP1 {symbol}",
                                                                        max_retries=2,
                                                                        delay=2.0
                                                                    )
                                                                    if final_close:
                                                                        logger.info(f"✅ {symbol}: Оставшаяся часть позиции закрыта")
                                                                        order_closed = True
                                                                except Exception as e:
                                                                    logger.warning(f"⚠️ {symbol}: Ошибка закрытия оставшейся части: {e}")
                                                            break
                                                
                                                if not position_still_open:
                                                    logger.info(f"✅ {symbol}: Позиция подтверждена как закрытая на бирже")
                                                    order_closed = True
                                            except Exception as e:
                                                logger.warning(f"⚠️ {symbol}: Не удалось проверить позицию на бирже: {e}")
                                        
                                        if not order_closed:
                                            logger.error(f"🚨 {symbol}: КРИТИЧНО! Ордер TP1 может быть частично исполнен или не закрыт!")
                                            logger.error(f"   Оставшийся размер: {remaining_size:.6f}")
                                            logger.error(f"   Требуется ручная проверка!")
                                            
                                            # Отправляем критическое уведомление в Telegram
                                            if self.telegram_bot:
                                                try:
                                                    await self.send_telegram_v4(
                                                        f"🚨 КРИТИЧНО: {symbol}\n"
                                                        f"Ордер TP1 может быть частично исполнен\n"
                                                        f"Оставшийся размер: {remaining_size:.6f}\n"
                                                        f"Требуется ручная проверка!"
                                                    )
                                                except:
                                                    pass
                                        
                                        # Сохраняем сделку в БД
                                        # ⚠️ ОТКЛЮЧЕНО: DataStorageSystem (упрощение архитектуры)
                                        if False and self.data_storage:  # Отключено
                                            try:
                                                from data_storage_system import TradeDecision
                                                trade_decision = TradeDecision(
                                                    timestamp=datetime.now(WARSAW_TZ).isoformat(),
                                                    symbol=symbol,
                                                    decision=signal.direction,
                                                    confidence=signal.confidence,
                                                    strategy_score=signal.strategy_score,
                                                    reasons=signal.reasons + ['TP1 достигнут'],
                                                    market_data={
                                                        'entry_price': entry_price,
                                                        'exit_price': current_price,
                                                        'tp_level': 1,
                                                        'tp_percent': 1.0
                                                    },
                                                    result='win',
                                                    pnl_percent=pnl_percent,
                                                    entry_price=entry_price,
                                                    exit_price=current_price
                                                )
                                                self.data_storage.store_trade_decision(trade_decision)
                                                logger.info(f"💾 {symbol}: Сделка TP1 сохранена в БД")
                                            except Exception as e:
                                                logger.error(f"❌ {symbol}: Ошибка сохранения TP1 в БД: {e}")
                                        
                                        logger.info(f"✅ {symbol}: TP1 достигнут! Позиция закрыта на ${current_price:.4f} (PnL: {pnl_percent:.2f}%)")
                                        
                                        # Отправляем уведомление
                                        if self.telegram_bot:
                                            await self.send_position_closed_v4(
                                                symbol=symbol,
                                                side=signal.direction,
                                                entry_price=entry_price,
                                                exit_price=current_price,
                                                pnl_percent=pnl_percent,
                                                profit_usd=pnl_percent / 100 * pos_info.get('position_notional', self.POSITION_NOTIONAL),
                                                reason="TP1 достигнут (+1.15%)"
                                            )
                                        
                                        # 🔗 DISCO57: Запись результата для Shadow Learning и обучение RL-агента
                                        if self.disco57 and 'signal' in pos_info:
                                            try:
                                                signal = pos_info['signal']
                                                features = getattr(signal, 'disco57_features', None) if hasattr(signal, 'disco57_features') else None
                                                current_signal = {
                                                    'action': signal.direction,
                                                    'confidence': signal.confidence,
                                                    'entry_price': signal.entry_price
                                                }
                                                result = {
                                                    'pnl_usd': pnl_percent / 100 * pos_info.get('position_notional', self.POSITION_NOTIONAL),
                                                    'pnl_percent': pnl_percent,
                                                    'roe': pnl_percent * self.LEVERAGE,
                                                    'win': pnl_percent > 0,
                                                    'close_price': current_price,
                                                    'reason': 'take_profit_tp1'
                                                }
                                                self.disco57.record_decision(symbol, features, current_signal, None, result)
                                                
                                                # 🎓 ОБУЧЕНИЕ RL-АГЕНТА ПОСЛЕ КАЖДОЙ СДЕЛКИ (для быстрого улучшения)
                                                self.rl_training_counter += 1
                                                logger.info(f"🎓 Запись сделки TP1 для обучения RL-агента (сделка #{self.rl_training_counter})")
                                                
                                                # Пытаемся обучить RL-агента (может вернуть None если недостаточно данных)
                                                metrics = self.disco57.train_rl_agent(min_samples=1)  # ИЗМЕНЕНО: минимум 1 образец для обучения после каждой сделки
                                                if metrics:
                                                    logger.info(f"✅ RL-агент обучен после TP1 сделки #{self.rl_training_counter} | Loss: {metrics.get('loss', 0):.4f}")
                                                else:
                                                    logger.debug(f"⏸️ RL-агент: недостаточно данных для обучения (сделка #{self.rl_training_counter}), данные сохранены для будущего обучения")
                                            except Exception as e:
                                                logger.debug(f"⚠️ Ошибка записи решения Disco57 для TP1: {e}")
                                        
                                        # Удаляем позицию
                                        symbol_norm = self.normalize_symbol(symbol)
                                        for key in list(self.active_positions.keys()):
                                            if self.normalize_symbol(key) == symbol_norm:
                                                del self.active_positions[key]
                                                break
                                        
                                        # 🚀 МГНОВЕННЫЙ ПЕРЕХОД К ПОИСКУ НОВЫХ СДЕЛОК
                                        logger.info(f"🚀 {symbol}: Позиция закрыта, переходим к поиску новых возможностей")
                                        continue  # Переходим к следующей позиции
                                    except Exception as e:
                                        logger.error(f"❌ {symbol}: Ошибка закрытия по TP1: {e}")
                                    
                                    # Мониторим BB Position и RSI
                                    try:
                                        # Получаем текущие данные 30m для расчета BB Position и RSI
                                        mtf_data = await self._fetch_multi_timeframe_data(symbol)
                                        current_30m = mtf_data.get('30m', {})
                                        
                                        if current_30m:
                                            rsi = current_30m.get('rsi', 0)
                                            bb_position = current_30m.get('bb_position', 50)
                                            
                                            # Предупреждения при достижении критических уровней
                                            if bb_position > 80:
                                                logger.warning(f"⚠️ {symbol}: BB Position = {bb_position:.1f}% (критично высоко!) → Рассмотреть частичное закрытие")
                                                if self.telegram_bot:
                                                    await self.send_telegram_v4(
                                                        f"⚠️ {symbol}: BB Position = {bb_position:.1f}% (критично высоко!)\n"
                                                        f"Рекомендация: Рассмотреть частичное закрытие позиции"
                                                    )
                                            elif bb_position > 75:
                                                logger.info(f"📊 {symbol}: BB Position = {bb_position:.1f}% (высоко) → Мониторить")
                                            
                                            if rsi > 70:
                                                logger.warning(f"⚠️ {symbol}: RSI = {rsi:.1f} (перекупленность!) → Рассмотреть частичное закрытие")
                                                if self.telegram_bot:
                                                    await self.send_telegram_v4(
                                                        f"⚠️ {symbol}: RSI = {rsi:.1f} (перекупленность!)\n"
                                                        f"Рекомендация: Рассмотреть частичное закрытие позиции"
                                                    )
                                            elif rsi > 65:
                                                logger.info(f"📊 {symbol}: RSI = {rsi:.1f} (близко к перекупленности) → Мониторить")
                                            
                                            # Логируем текущие значения
                                            logger.info(f"📊 {symbol}: TP1 достигнут | BB Position: {bb_position:.1f}% | RSI: {rsi:.1f}")
                                    except Exception as e:
                                        logger.debug(f"⚠️ {symbol}: Ошибка мониторинга BB Position/RSI при TP1: {e}")
                                
                                # Периодический мониторинг BB Position и RSI после достижения TP1 (каждые 5 минут)
                                last_bb_rsi_check = pos_info.get('last_bb_rsi_check_time', None)
                                check_interval = timedelta(minutes=5)
                                
                                if not last_bb_rsi_check or (datetime.now(WARSAW_TZ) - last_bb_rsi_check) >= check_interval:
                                    try:
                                        mtf_data = await self._fetch_multi_timeframe_data(symbol)
                                        current_30m = mtf_data.get('30m', {})
                                        
                                        if current_30m:
                                            rsi = current_30m.get('rsi', 0)
                                            bb_position = current_30m.get('bb_position', 50)
                                            
                                            # Предупреждения при достижении критических уровней
                                            if bb_position > 80:
                                                logger.warning(f"⚠️ {symbol}: BB Position = {bb_position:.1f}% (критично высоко!) → Рассмотреть частичное закрытие")
                                                if self.telegram_bot:
                                                    await self.send_telegram_v4(
                                                        f"⚠️ {symbol}: BB Position = {bb_position:.1f}% (критично высоко!)\n"
                                                        f"Рекомендация: Рассмотреть частичное закрытие позиции"
                                                    )
                                            elif bb_position > 75:
                                                logger.info(f"📊 {symbol}: BB Position = {bb_position:.1f}% (высоко) → Мониторить")
                                            
                                            if rsi > 70:
                                                logger.warning(f"⚠️ {symbol}: RSI = {rsi:.1f} (перекупленность!) → Рассмотреть частичное закрытие")
                                                if self.telegram_bot:
                                                    await self.send_telegram_v4(
                                                        f"⚠️ {symbol}: RSI = {rsi:.1f} (перекупленность!)\n"
                                                        f"Рекомендация: Рассмотреть частичное закрытие позиции"
                                                    )
                                            elif rsi > 65:
                                                logger.info(f"📊 {symbol}: RSI = {rsi:.1f} (близко к перекупленности) → Мониторить")
                                            
                                            # Обновляем время последней проверки
                                            pos_info['last_bb_rsi_check_time'] = datetime.now(WARSAW_TZ)
                                    except Exception as e:
                                        logger.debug(f"⚠️ {symbol}: Ошибка периодического мониторинга BB Position/RSI: {e}")
                    except Exception:
                        pass
                    
                    # 🚨 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем SL ВСЕГДА, независимо от прибыли!
                    # SL - это основная защита от убытков, должна работать ВСЕГДА!
                    stop_loss = pos_info.get('stop_loss', signal.stop_loss)
                    initial_sl = pos_info.get('initial_sl', stop_loss)
                    tp1_reached_flag = pos_info.get('tp1_reached', False)
                    
                    # 🚨 ПЕРВЫЙ ПРИОРИТЕТ: Проверяем SL ВСЕГДА (даже если позиция в убытке)
                    sl_hit_primary = False
                    if stop_loss:
                        if signal.direction == 'buy':
                            sl_hit_primary = current_price <= stop_loss
                        else:
                            sl_hit_primary = current_price >= stop_loss
                        
                        # 🚨 КРИТИЧНО: Закрываем позицию сразу при достижении SL
                        if sl_hit_primary:
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
                                
                                logger.warning(f"🛑 {symbol}: Stop Loss сработал! Позиция закрыта на ${current_price:.4f} (PnL: {pnl_percent:.2f}%)")
                                
                                # Отправляем уведомление
                                if self.telegram_bot:
                                    await self.send_position_closed_v4(
                                        symbol=symbol,
                                        side=signal.direction,
                                        entry_price=entry_price,
                                        exit_price=current_price,
                                        pnl_percent=pnl_percent,
                                        profit_usd=current_pnl_usd if 'current_pnl_usd' in locals() else (pnl_percent / 100 * pos_info.get('position_notional', self.POSITION_NOTIONAL)),
                                        reason="Stop Loss сработал"
                                    )
                                
                                # Удаляем позицию из словаря (с нормализацией)
                                symbol_norm = self.normalize_symbol(symbol)
                                for key in list(self.active_positions.keys()):
                                    if self.normalize_symbol(key) == symbol_norm:
                                        del self.active_positions[key]
                                        break
                                
                                # Обновляем метрики
                                self._update_performance_metrics(
                                    pnl_usd=current_pnl_usd if 'current_pnl_usd' in locals() else (pnl_percent / 100 * pos_info.get('position_notional', self.POSITION_NOTIONAL)),
                                    pnl_percent=pnl_percent,
                                    symbol=symbol
                                )
                                
                                # 🔗 DISCO57: Запись результата для Shadow Learning
                                if self.disco57 and 'signal' in pos_info:
                                    try:
                                        signal = pos_info['signal']
                                        features = getattr(signal, 'disco57_features', None) if hasattr(signal, 'disco57_features') else None
                                        current_signal = {
                                            'action': signal.direction,
                                            'confidence': signal.confidence,
                                            'entry_price': signal.entry_price
                                        }
                                        result = {
                                            'pnl_usd': current_pnl_usd if 'current_pnl_usd' in locals() else (pnl_percent / 100 * pos_info.get('position_notional', self.POSITION_NOTIONAL)),
                                            'pnl_percent': pnl_percent,
                                            'roe': pnl_percent * self.LEVERAGE,
                                            'win': pnl_percent > 0,
                                            'close_price': current_price,
                                            'reason': 'stop_loss'
                                        }
                                        self.disco57.record_decision(symbol, features, current_signal, None, result)
                                        
                                        # 🎓 ОБУЧЕНИЕ RL-АГЕНТА ПОСЛЕ КАЖДОЙ СДЕЛКИ (для быстрого улучшения)
                                        self.rl_training_counter += 1
                                        logger.info(f"🎓 Запись сделки для обучения RL-агента (сделка #{self.rl_training_counter})")
                                        
                                        # Пытаемся обучить RL-агента (может вернуть None если недостаточно данных)
                                        metrics = self.disco57.train_rl_agent(min_samples=1)  # ИЗМЕНЕНО: минимум 1 образец для обучения после каждой сделки
                                        if metrics:
                                            logger.info(f"✅ RL-агент обучен после сделки #{self.rl_training_counter} | Loss: {metrics.get('loss', 0):.4f}")
                                        else:
                                            logger.debug(f"⏸️ RL-агент: недостаточно данных для обучения (сделка #{self.rl_training_counter}), данные сохранены для будущего обучения")
                                    except Exception as e:
                                        logger.debug(f"⚠️ Ошибка записи решения Disco57: {e}")
                                
                                continue  # Переходим к следующей позиции
                            except Exception as e:
                                logger.error(f"❌ {symbol}: Ошибка закрытия по SL: {e}")
                    
                    # 🎯 TRAILING STOP ORDER: Скользящий стоп-ордер как на фото
                    # Activation price (цена активации) - когда цена достигает этого уровня, trailing stop активируется
                    # Для LONG: отслеживаем highest_price, SL = highest_price - correction_level
                    # Для SHORT: отслеживаем lowest_price, SL = lowest_price + correction_level
                    
                    trailing_stop_active = pos_info.get('trailing_stop_active', False)
                    activation_price = pos_info.get('activation_price')
                    highest_price = pos_info.get('highest_price', entry_price if signal.direction == 'buy' else None)
                    lowest_price = pos_info.get('lowest_price', entry_price if signal.direction == 'sell' else None)
                    correction_level_pct = pos_info.get('correction_level_pct', 0.5)  # Уровень коррекции 0.5%
                    
                    # Определяем цену активации (например, +0.5% прибыли для активации trailing stop)
                    if activation_price is None:
                        activation_price_pct = 0.5  # Активируем при +0.5% прибыли
                        if signal.direction == 'buy':
                            activation_price = entry_price * (1 + activation_price_pct / 100.0)
                        else:
                            activation_price = entry_price * (1 - activation_price_pct / 100.0)
                        pos_info['activation_price'] = activation_price
                        logger.debug(f"🎯 {symbol}: Цена активации trailing stop: ${activation_price:.6f} ({activation_price_pct}% от входа)")
                    
                    # Проверяем, достигнута ли цена активации
                    if not trailing_stop_active:
                        if signal.direction == 'buy':
                            if current_price >= activation_price:
                                trailing_stop_active = True
                                pos_info['trailing_stop_active'] = True
                                highest_price = current_price
                                pos_info['highest_price'] = highest_price
                                logger.info(f"🎯 {symbol}: Trailing stop активирован! Цена: ${current_price:.6f} >= ${activation_price:.6f}")
                        else:  # sell
                            if current_price <= activation_price:
                                trailing_stop_active = True
                                pos_info['trailing_stop_active'] = True
                                lowest_price = current_price
                                pos_info['lowest_price'] = lowest_price
                                logger.info(f"🎯 {symbol}: Trailing stop активирован! Цена: ${current_price:.6f} <= ${activation_price:.6f}")
                    
                    # Если trailing stop активирован, обновляем highest/lowest price и пересчитываем SL
                    if trailing_stop_active:
                        if signal.direction == 'buy':
                            # Для LONG: отслеживаем максимальную цену
                            if current_price > highest_price:
                                highest_price = current_price
                                pos_info['highest_price'] = highest_price
                            
                            # SL = highest_price - correction_level
                            correction_amount = highest_price * (correction_level_pct / 100.0)
                            new_trailing_sl = highest_price - correction_amount
                            
                            # SL не должен быть ниже текущего stop_loss (только улучшаем позицию)
                            if new_trailing_sl > stop_loss:
                                stop_loss = new_trailing_sl
                                pos_info['stop_loss'] = stop_loss
                                # Обновляем SL на бирже
                                try:
                                    await self._set_sl_tp_pybit(symbol, stop_loss, None)
                                    logger.debug(f"📈 {symbol}: Trailing SL обновлен: ${stop_loss:.6f} (highest: ${highest_price:.6f}, коррекция: {correction_level_pct}%)")
                                except Exception as e:
                                    logger.debug(f"⚠️ {symbol}: Ошибка обновления trailing SL: {e}")
                            
                            sl_hit = current_price <= stop_loss
                        else:  # sell
                            # Для SHORT: отслеживаем минимальную цену
                            if current_price < lowest_price:
                                lowest_price = current_price
                                pos_info['lowest_price'] = lowest_price
                            
                            # SL = lowest_price + correction_level
                            correction_amount = lowest_price * (correction_level_pct / 100.0)
                            new_trailing_sl = lowest_price + correction_amount
                            
                            # SL не должен быть выше текущего stop_loss (только улучшаем позицию)
                            if new_trailing_sl < stop_loss or stop_loss == initial_sl:
                                stop_loss = new_trailing_sl
                                pos_info['stop_loss'] = stop_loss
                                # Обновляем SL на бирже
                                try:
                                    await self._set_sl_tp_pybit(symbol, stop_loss, None)
                                    logger.debug(f"📈 {symbol}: Trailing SL обновлен: ${stop_loss:.6f} (lowest: ${lowest_price:.6f}, коррекция: {correction_level_pct}%)")
                                except Exception as e:
                                    logger.debug(f"⚠️ {symbol}: Ошибка обновления trailing SL: {e}")
                            
                            sl_hit = current_price >= stop_loss
                    else:
                        # Trailing stop еще не активирован, используем обычную проверку SL
                        if signal.direction == 'buy':
                            sl_hit = current_price <= stop_loss
                        else:
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
                                
                                # Удаляем позицию из словаря (с нормализацией)
                                symbol_norm = self.normalize_symbol(symbol)
                                for key in list(self.active_positions.keys()):
                                    if self.normalize_symbol(key) == symbol_norm:
                                        del self.active_positions[key]
                                        break
                                
                                # Отправляем уведомление через правильный метод
                                if self.telegram_bot:
                                    await self.send_position_closed_v4(
                                        symbol=symbol,
                                        side=signal.direction,
                                        entry_price=entry_price,
                                        exit_price=current_price,
                                        pnl_percent=pnl_percent,
                                        profit_usd=current_pnl_usd if 'current_pnl_usd' in locals() else (pnl_percent / 100 * pos_info.get('position_notional', self.POSITION_NOTIONAL)),
                                        reason="Stop Loss сработал (trailing)"
                                    )
                                
                            except Exception as e:
                                logger.error(f"❌ {symbol}: Ошибка закрытия по SL: {e}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка мониторинга позиции {position.get('symbol', 'unknown')}: {e}")
            
            # Удаляем закрытые позиции из словаря
            # НОРМАЛИЗАЦИЯ СИМВОЛОВ: используем единую функцию для корректного сравнения
            active_symbols = {self.normalize_symbol(p.get('symbol', '')) for p in open_positions if (p.get('contracts', 0) or p.get('size', 0)) > 0}
            
            # ИСПРАВЛЕНИЕ: Проверяем не только позиции из словаря, но и все, что могли открыть ранее
            # Сохраняем предыдущее состояние открытых позиций
            if not hasattr(self, '_prev_open_positions'):
                self._prev_open_positions = set()
            
            # Нормализуем предыдущие позиции для сравнения
            prev_open_normalized = {self.normalize_symbol(s) for s in self._prev_open_positions}
            active_positions_normalized = {self.normalize_symbol(s) for s in self.active_positions.keys()}
            
            # Находим позиции, которые были открыты ранее, но теперь закрыты
            closed_detected = prev_open_normalized - active_symbols
            
            # Проверяем позиции из словаря (те, что открыл этот бот)
            closed_symbols = active_positions_normalized - active_symbols
            
            # Объединяем все закрытые позиции
            all_closed = closed_symbols.union(closed_detected)
            
            for symbol in all_closed:
                try:
                    # Ищем позицию в словаре по нормализованному символу
                    # Перебираем все ключи и ищем совпадение по нормализованному значению
                    pos_info = None
                    signal = None
                    for key in self.active_positions.keys():
                        if self.normalize_symbol(key) == symbol:
                            pos_info = self.active_positions[key]
                            signal = pos_info.get('signal')
                            break
                    
                    # Если позиции нет в словаре, получаем данные с биржи
                    if not pos_info:
                        # Позиция была открыта до запуска бота - получаем данные из closed PnL
                        try:
                            from pybit.unified_trading import HTTP
                            session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False, recv_window=5000, timeout=10)
                            # Bybit использует формат без USDT в конце для get_closed_pnl
                            bybit_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                            cp = session.get_closed_pnl(category='linear', symbol=bybit_symbol, limit=1)
                            items = cp.get('result',{}).get('list',[]) or []
                            if items:
                                it = items[0]
                                closed_pnl = float(it.get('closedPnl') or 0)
                                avg_entry_raw = it.get('avgEntryPrice')
                                avg_exit_raw = it.get('avgExitPrice')
                                
                                # Используем данные из API только если они валидны
                                ex_side = it.get('side', 'Buy')  # Получаем side ДО использования
                                qty = float(it.get('qty', 0))  # Получаем qty для расчета
                                
                                avg_entry = float(avg_entry_raw) if avg_entry_raw and float(avg_entry_raw) > 0 else entry_price
                                avg_exit = float(avg_exit_raw) if avg_exit_raw and float(avg_exit_raw) > 0 else None
                                
                                # Если цена выхода не получена из API, используем текущую цену или рассчитываем из closedPnl
                                if avg_exit is None or avg_exit == 0 or avg_exit == avg_entry:
                                    logger.warning(f"⚠️ {symbol}: avgExitPrice из API невалидна ({avg_exit_raw}), пытаемся получить текущую цену")
                                    try:
                                        ticker = await self.exchange.fetch_ticker(symbol)
                                        current_mark = float(ticker.get('last') or ticker.get('close') or 0)
                                        if current_mark > 0 and current_mark != avg_entry:
                                            avg_exit = current_mark
                                            logger.info(f"✅ {symbol}: Использована текущая цена как цена выхода: {avg_exit:.5f} (вместо невалидной из API)")
                                        else:
                                            logger.error(f"🚨 {symbol}: Текущая цена тоже невалидна ({current_mark}) или равна entry ({avg_entry:.5f})")
                                            # Рассчитываем exit_price из closedPnl
                                            if closed_pnl != 0 and avg_entry > 0 and qty > 0:
                                                if ex_side == 'Buy':
                                                    avg_exit = avg_entry * (1 + closed_pnl / (avg_entry * qty))
                                                else:
                                                    avg_exit = avg_entry * (1 - closed_pnl / (avg_entry * qty))
                                                logger.info(f"✅ {symbol}: Exit цена рассчитана из closedPnl: {avg_exit:.5f}")
                                            else:
                                                logger.error(f"🚨 {symbol}: Невозможно рассчитать exit_price! Entry: {avg_entry}, PnL: {closed_pnl}, Qty: {qty}")
                                                # Пропускаем эту позицию - не отправляем сообщение с невалидными данными
                                                continue
                                    except Exception as e:
                                        logger.error(f"❌ {symbol}: Не удалось получить текущую цену: {e}")
                                        # Рассчитываем exit_price из closedPnl
                                        if closed_pnl != 0 and avg_entry > 0 and qty > 0:
                                            if ex_side == 'Buy':
                                                avg_exit = avg_entry * (1 + closed_pnl / (avg_entry * qty))
                                            else:
                                                avg_exit = avg_entry * (1 - closed_pnl / (avg_entry * qty))
                                            logger.info(f"✅ {symbol}: Exit цена рассчитана из closedPnl (fallback): {avg_exit:.5f}")
                                        else:
                                            logger.error(f"🚨 {symbol}: Невозможно рассчитать exit_price! Entry: {avg_entry}, PnL: {closed_pnl}, Qty: {qty}")
                                            # Пропускаем эту позицию - не отправляем сообщение с невалидными данными
                                            continue
                                
                                side = 'buy' if ex_side == 'Buy' else 'sell'
                                profit_usd = closed_pnl
                                real_entry = avg_entry
                                real_exit = avg_exit
                                position_notional = avg_entry * float(it.get('qty', 0)) if avg_entry > 0 else self.POSITION_NOTIONAL
                                pnl_percent = (profit_usd / position_notional) * 100 if position_notional > 0 else 0
                                
                                logger.info(f"✅ {symbol}: Позиция закрыта (обнаружена из истории биржи) | PnL=${profit_usd:.2f}")
                                
                                # НОВОЕ: Отслеживаем убыточные монеты для cooldown
                                if profit_usd < -0.5:  # Если убыток больше $0.50
                                    symbol_norm_loss = self.normalize_symbol(symbol)
                                    self.losing_symbols[symbol_norm_loss] = (abs(profit_usd), datetime.now(WARSAW_TZ))
                                    logger.warning(f"⚠️ {symbol}: Добавлена в список убыточных монет (cooldown 12ч). Потеря: {profit_usd:.2f} USDT")
                                
                                # 🧠 АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ: Обучение сразу после каждой сделки
                                # ⚠️ ОТКЛЮЧЕНО: UniversalLearningSystem
                                # ⚠️ ОТКЛЮЧЕНО: UniversalLearningSystem (упрощение архитектуры)
                                # Обучение происходит только через Disco57 (PPO Agent)
                                if False and self.universal_learning:  # Отключено
                                    try:
                                        market_data = {
                                            'symbol': symbol,
                                            'side': side,
                                            'entry_price': real_entry,
                                            'exit_price': real_exit,
                                            'pnl': profit_usd,
                                            'pnl_percent': pnl_percent,
                                            'market_condition': getattr(self, '_current_market_condition', 'NEUTRAL'),
                                            'confidence': 0  # Не сохраняем confidence для исторических данных
                                        }
                                        decision = 'buy' if side.lower() == 'buy' or side.lower() == 'long' else 'sell'
                                        result = 'success' if profit_usd > 0 else 'failure'
                                        
                                        # self.universal_learning.learn_from_decision(market_data, decision, result)
                                        logger.debug(f"ℹ️ {symbol}: Обучение через Disco57 (PPO Agent) | Решение: {decision.upper()}, Результат: {result}, PnL: ${profit_usd:.2f}")
                                    except Exception as e:
                                        logger.error(f"⚠️ {symbol}: Ошибка обучения на исторической сделке: {e}")
                                
                                if self.telegram_bot:
                                    try:
                                        await self.send_position_closed_v4(
                                            symbol=symbol,
                                            side=side,
                                            entry_price=real_entry,
                                            exit_price=real_exit,
                                            pnl_percent=pnl_percent,
                                            profit_usd=profit_usd,
                                            reason="Закрыта на бирже/по TP/SL (обнаружено из истории)"
                                        )
                                    except Exception as e:
                                        logger.error(f"⚠️ Ошибка отправки Telegram при обнаружении закрытия {symbol}: {e}")
                                continue
                        except Exception as e:
                            logger.debug(f"⚠️ Не удалось получить данные о закрытии {symbol} из истории: {e}")
                            continue
                    
                    # Если позиция была в словаре - используем сохраненные данные
                    side = 'buy' if (signal and signal.direction == 'buy') else 'sell'
                    entry_price = float(pos_info.get('entry_price') or 0)
                    last_price = float(pos_info.get('current_price') or 0)
                    pnl_percent = float(pos_info.get('pnl_percent') or 0)
                    profit_usd = pnl_percent / 100 * float(pos_info.get('position_notional', self.POSITION_NOTIONAL) or self.POSITION_NOTIONAL)

                    # 🚨 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Получаем ФАКТИЧЕСКИЙ closed PnL и цены из Bybit API
                    # Проблема: сообщения с нулевыми данными из-за одинаковых цен
                    real_entry = entry_price if entry_price > 0 else 0
                    real_exit = 0
                    profit_usd = 0
                    pnl_percent = 0
                    
                    # ПРИОРИТЕТ #1: Получаем данные из closed PnL API (самый надежный источник)
                    try:
                        from pybit.unified_trading import HTTP
                        session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False, recv_window=5000, timeout=10)
                        # Bybit использует формат без USDT в конце для get_closed_pnl
                        bybit_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                        cp = session.get_closed_pnl(category='linear', symbol=bybit_symbol, limit=10)
                        items = cp.get('result',{}).get('list',[]) or []
                        
                        if items:
                            it = items[0]
                            closed_pnl = float(it.get('closedPnl') or 0)
                            avg_entry_raw = it.get('avgEntryPrice')
                            avg_exit_raw = it.get('avgExitPrice')
                            
                            # Используем данные из API только если они валидны
                            if avg_entry_raw and float(avg_entry_raw) > 0:
                                real_entry = float(avg_entry_raw)
                            if avg_exit_raw and float(avg_exit_raw) > 0:
                                real_exit = float(avg_exit_raw)
                            
                            # Если цены все еще одинаковые, пытаемся использовать markPrice или текущую цену
                            if real_entry == real_exit or real_exit == 0:
                                # Получаем текущую цену как fallback
                                try:
                                    ticker = self.exchange.fetch_ticker(symbol)
                                    current_mark = float(ticker.get('last') or ticker.get('close') or 0)
                                    if current_mark > 0 and current_mark != real_entry:
                                        real_exit = current_mark
                                        logger.debug(f"⚠️ {symbol}: Использована текущая цена как цена выхода: {real_exit:.5f}")
                                except Exception as e:
                                    logger.debug(f"⚠️ {symbol}: Не удалось получить текущую цену для выхода: {e}")
                            
                            # КРИТИЧНО: Используем closedPnl из API как основной источник данных
                            profit_usd = closed_pnl
                            
                            # Пересчёт процента от базового нотионала
                            base_notional = float(pos_info.get('position_notional', self.POSITION_NOTIONAL) or self.POSITION_NOTIONAL)
                            if base_notional > 0 and profit_usd != 0:
                                # Пересчитываем процент на основе реального PnL из API
                                pnl_percent = (profit_usd / base_notional) * 100
                            elif real_entry > 0 and real_exit > 0 and real_exit != real_entry:
                                # Если нотионал неизвестен, рассчитываем процент от цены
                                if side.lower() == 'buy' or side.lower() == 'long':
                                    pnl_percent = ((real_exit - real_entry) / real_entry) * 100
                                else:
                                    pnl_percent = ((real_entry - real_exit) / real_entry) * 100
                                # Пересчитываем profit_usd на основе процента
                                if base_notional > 0:
                                    profit_usd = (pnl_percent / 100) * base_notional
                            else:
                                # Если все данные недоступны, используем closedPnl напрямую
                                logger.warning(f"⚠️ {symbol}: Неполные данные из API, используем closedPnl напрямую: ${profit_usd:.2f}")
                                if base_notional > 0:
                                    pnl_percent = (profit_usd / base_notional) * 100
                            
                            logger.info(f"📊 {symbol}: Данные из API - Entry: {real_entry:.5f}, Exit: {real_exit:.5f}, Closed PnL: ${profit_usd:.2f} ({pnl_percent:.2f}%)")
                    except Exception as e:
                        logger.warning(f"⚠️ {symbol}: Не удалось получить данные о закрытии из API: {e}")
                        # Если не удалось получить данные из API, рассчитываем PnL вручную
                        if real_entry > 0 and real_exit > 0 and real_exit != real_entry:
                            base_notional = float(pos_info.get('position_notional', self.POSITION_NOTIONAL) or self.POSITION_NOTIONAL)
                            if side.lower() == 'buy' or side.lower() == 'long':
                                pnl_percent = ((real_exit - real_entry) / real_entry) * 100
                            else:
                                pnl_percent = ((real_entry - real_exit) / real_entry) * 100
                            profit_usd = (pnl_percent / 100) * base_notional if base_notional > 0 else 0
                        else:
                            # Если цены одинаковые, это критическая ошибка - логируем
                            logger.error(f"🚨 {symbol}: КРИТИЧЕСКАЯ ОШИБКА - цены входа и выхода одинаковые! Entry: {real_entry:.5f}, Exit: {real_exit:.5f}")
                            # Устанавливаем минимальный убыток, чтобы статистика отражала проблему
                            profit_usd = -0.01  # Минимальный убыток для отображения проблемы
                            pnl_percent = -0.05  # Минимальный процент

                    logger.info(f"✅ {symbol}: Позиция закрыта (обнаружено по сверке) | PnL=${profit_usd:.2f}")
                    
                    # НОВОЕ: Отслеживаем убыточные монеты для cooldown
                    if profit_usd < -0.5:  # Если убыток больше $0.50
                        symbol_norm_loss = self.normalize_symbol(symbol)
                        self.losing_symbols[symbol_norm_loss] = (abs(profit_usd), datetime.now(WARSAW_TZ))
                        logger.warning(f"⚠️ {symbol}: Добавлена в список убыточных монет (cooldown 12ч). Потеря: {profit_usd:.2f} USDT")
                    
                    # 🧠 АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ: Обучение сразу после каждой сделки
                    # ⚠️ ОТКЛЮЧЕНО: UniversalLearningSystem
                    if False and self.universal_learning:  # Отключено
                        try:
                            market_data = {
                                'symbol': symbol,
                                'side': side,
                                'entry_price': real_entry,
                                'exit_price': real_exit,
                                'pnl': profit_usd,
                                'pnl_percent': pnl_percent,
                                'market_condition': getattr(self, '_current_market_condition', 'NEUTRAL'),
                                'confidence': pos_info.get('confidence', 0) if pos_info else 0
                            }
                            decision = 'buy' if side.lower() == 'buy' or side.lower() == 'long' else 'sell'
                            result = 'success' if profit_usd > 0 else 'failure'
                            
                            # ⚠️ ОТКЛЮЧЕНО: UniversalLearningSystem (упрощение архитектуры)
                            # Обучение происходит только через Disco57 (PPO Agent)
                            # if self.universal_learning:
                            #     self.universal_learning.learn_from_decision(market_data, decision, result)
                            logger.debug(f"ℹ️ {symbol}: Обучение через Disco57 (PPO Agent) | Решение: {decision.upper()}, Результат: {result}, PnL: ${profit_usd:.2f}")
                        except Exception as e:
                            logger.error(f"⚠️ {symbol}: Ошибка обучения на сделке: {e}")
                    
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
                            # 📊 Обновляем метрики производительности
                            try:
                                pos_info_sync = self.active_positions.get(symbol, {})
                                timeframe = pos_info_sync.get('timeframe') if pos_info_sync else None
                                self._update_performance_metrics(
                                    pnl_usd=profit_usd,
                                    pnl_percent=pnl_percent,
                                    symbol=symbol,
                                    timeframe=timeframe
                                )
                            except Exception as e:
                                logger.error(f"⚠️ Ошибка обновления метрик для {symbol}: {e}")
                        except Exception as e:
                            logger.error(f"⚠️ Ошибка отправки Telegram при сверке закрытия {symbol}: {e}")
                finally:
                    # Удаляем из словаря по оригинальному ключу, если он найден
                    symbol_norm_final = self.normalize_symbol(symbol)
                    for key in list(self.active_positions.keys()):
                        if self.normalize_symbol(key) == symbol_norm_final:
                            del self.active_positions[key]
                            break
            
            # Обновляем состояние открытых позиций для следующей проверки
            # Используем оригинальные символы из биржи (без нормализации)
            self._prev_open_positions = {p.get('symbol', '') for p in open_positions if (p.get('contracts', 0) or p.get('size', 0)) > 0}

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
        # 🔴 ПРИОРИТЕТ 2.3: Улучшенная обработка ошибок с автоматическим перезапуском
        max_retries = 3
        retry_delay = 60  # 1 минута между попытками
        
        for attempt in range(max_retries):
            try:
                await self._trading_loop_v4_internal()
                break  # Успешное выполнение, выходим из цикла retry
            except KeyboardInterrupt:
                logger.info("🛑 Торговый цикл остановлен пользователем")
                raise  # Пробрасываем KeyboardInterrupt
            except Exception as e:
                error_msg = str(e)
                is_critical = any(keyword in error_msg.lower() for keyword in [
                    'connection', 'timeout', 'network', 'api', 'exchange', 'critical'
                ])
                
                if attempt < max_retries - 1:
                    logger.error(f"❌ Ошибка в торговом цикле (попытка {attempt + 1}/{max_retries}): {e}")
                    if is_critical:
                        logger.warning(f"⚠️ Критическая ошибка обнаружена, перезапуск через {retry_delay}с...")
                        # Отправляем уведомление в Telegram при критической ошибке
                        if self.telegram_bot and attempt == 0:  # Только при первой попытке
                            try:
                                await self.send_telegram_v4(
                                    f"🚨 КРИТИЧЕСКАЯ ОШИБКА В ТОРГОВОМ ЦИКЛЕ\n\n"
                                    f"Ошибка: {error_msg[:200]}\n"
                                    f"Попытка восстановления: {attempt + 1}/{max_retries}\n"
                                    f"Перезапуск через {retry_delay}с..."
                                )
                            except:
                                pass
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.warning(f"⚠️ Некритическая ошибка, перезапуск через {retry_delay}с...")
                        await asyncio.sleep(retry_delay)
                else:
                    # Последняя попытка провалилась
                    logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Торговый цикл не удалось восстановить после {max_retries} попыток")
                    logger.error(f"   Последняя ошибка: {e}")
                    # Отправляем критическое уведомление
                    if self.telegram_bot:
                        try:
                            await self.send_telegram_v4(
                                f"🚨 КРИТИЧЕСКАЯ ОШИБКА: ТОРГОВЫЙ ЦИКЛ ОСТАНОВЛЕН\n\n"
                                f"Ошибка: {error_msg[:200]}\n"
                                f"Попытки восстановления: {max_retries}\n"
                                f"Требуется ручное вмешательство!"
                            )
                        except:
                            pass
                    raise  # Пробрасываем ошибку дальше
    
    async def _trading_loop_v4_internal(self):
        """Внутренний метод торгового цикла (без retry логики)"""
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
            
            # Инициализируем список кандидатов для этого цикла
            self.candidates_list = []
            
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
            # ⚠️ ОТКЛЮЧЕНО: DataStorageSystem
            if False and self.data_storage:  # Отключено
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
                            
                            # ⚠️ ОТКЛЮЧЕНО: DataStorageSystem и AdvancedMLSystem (упрощение архитектуры)
                            # ПРИЧИНА: Перегруженность ML систем, высокое потребление памяти, не влияют на торговые решения
                            # Оставлено только Disco57 (PPO Agent) для RL обучения на реальных данных
                            # if self.data_storage:
                            #     market_data_obj = MarketData(...)
                            #     self.data_storage.store_market_data(market_data_obj)
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
                # ИСПРАВЛЕНИЕ: Используем нормализацию для предотвращения дубликатов
                symbol_norm = self.normalize_symbol(symbol)
                position_exists = False
                for key in self.active_positions.keys():
                    if self.normalize_symbol(key) == symbol_norm:
                        position_exists = True
                        break
                if position_exists:
                    continue
                
                # КРИТИЧНО: Проверяем позиции на бирже перед анализом (с нормализацией символов)
                position_exists_on_exchange = False
                try:
                    # Нормализуем символ для проверки используя единую функцию
                    symbol_norm = self.normalize_symbol(symbol)
                    
                    # Получаем ВСЕ позиции, а не только по одному символу
                    all_positions = await self.exchange.fetch_positions(params={'category': 'linear'})
                    for pos in all_positions:
                        pos_symbol = pos.get('symbol', '')
                        pos_symbol_norm = self.normalize_symbol(pos_symbol)
                        
                        # Сравниваем нормализованные символы
                        if pos_symbol_norm == symbol_norm:
                            pos_size = pos.get('contracts', 0) or pos.get('size', 0)
                            if pos_size > 0:
                                logger.warning(f"🚫 {symbol}: ПРОПУЩЕН - уже есть открытая позиция на бирже ({pos_symbol}, размер: {pos_size})")
                                # Добавляем в active_positions для синхронизации (используем оригинальный символ с биржи)
                                self.active_positions[pos_symbol] = {
                                    'side': pos.get('side', ''),
                                    'entry_price': pos.get('entryPrice', pos.get('markPrice', 0)),
                                    'size': pos_size,
                                    'pnl_percent': pos.get('percentage', 0),
                                    'opened_at': datetime.now(WARSAW_TZ)  # Используем текущее время, если нет createdTime
                                }
                                # Пытаемся получить createdTime из биржи
                                created_time = pos.get('createdTime') or pos.get('updatedTime')
                                if created_time:
                                    try:
                                        if isinstance(created_time, (int, float)):
                                            self.active_positions[pos_symbol]['opened_at'] = datetime.fromtimestamp(int(created_time) / 1000, tz=WARSAW_TZ)
                                    except Exception as e:
                                        logger.debug(f"⚠️ Не удалось распарсить createdTime для {pos_symbol}: {e}")
                                
                                position_exists_on_exchange = True
                                break
                except Exception as e:
                    logger.error(f"❌ Ошибка проверки позиции на бирже для {symbol}: {e}")
                
                if position_exists_on_exchange:
                    continue  # Переходим к следующему символу
                
                # 🔴 КРИТИЧНО: Проверяем blacklist проблемных символов
                symbol_norm_blacklist = self.normalize_symbol(symbol)
                if symbol_norm_blacklist in self.problem_symbols_blacklist:
                    logger.warning(f"🚫 {symbol}: ПРОПУЩЕН - в blacklist проблемных символов (0% Win Rate или критический убыток)")
                    continue
                
                # НОВОЕ: Проверяем, не была ли эта монета убыточной в последние 12 часов
                symbol_norm_cooldown = self.normalize_symbol(symbol)
                if symbol_norm_cooldown in self.losing_symbols:
                    loss_info = self.losing_symbols[symbol_norm_cooldown]
                    loss_time = loss_info[1]
                    hours_since_loss = (datetime.now(WARSAW_TZ) - loss_time).total_seconds() / 3600
                    if hours_since_loss < 12:
                        logger.warning(f"🚫 {symbol}: ПРОПУЩЕН - убыточная монета (потеря {loss_info[0]:.2f} USDT {hours_since_loss:.1f}ч назад). Cooldown 12ч")
                        continue
                    else:
                        # Удаляем старую запись (прошло более 12 часов)
                        del self.losing_symbols[symbol_norm_cooldown]
                
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
                        
                        # 🚨 ПРОВЕРКА MAX_DAILY_DRAWDOWN перед открытием позиции
                        if self._trading_paused_due_to_drawdown:
                            logger.warning(f"🚫 {signal.symbol}: Пропущен! ТОРГОВЛЯ ПРИОСТАНОВЛЕНА из-за превышения MAX_DAILY_DRAWDOWN")
                            rejected_signals += 1
                            continue
                        
                        # 🚨 ПСИХОЛОГИЧЕСКИЙ СТОП-КОНТУР: Проверка MAX_CONSECUTIVE_LOSSES
                        if self._trading_paused_due_to_losses:
                            logger.warning(f"🚫 {signal.symbol}: Пропущен! ТОРГОВЛЯ ПРИОСТАНОВЛЕНА из-за {self.consecutive_losses} убытков подряд (лимит: {self.max_consecutive_losses})")
                            rejected_signals += 1
                            continue
                        
                        # Проверяем текущую дневную просадку
                        today = datetime.now(WARSAW_TZ).date().isoformat()
                        if today in self.daily_pnl_tracker:
                            daily_drawdown = self.daily_pnl_tracker[today].get('drawdown', 0.0)
                            if daily_drawdown >= self.max_daily_drawdown_percent:
                                logger.warning(f"🚫 {signal.symbol}: Пропущен! Дневная просадка {daily_drawdown:.2f}% >= {self.max_daily_drawdown_percent}%")
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
                        
                        # Логируем найденный сигнал
                        logger.info(f"🎯 {symbol}: Сигнал найден | "
                                  f"{signal.direction.upper()} | "
                                  f"Уверенность: {signal.confidence:.1f}% | "
                                  f"Стратегия: {signal.strategy_score:.1f}/20 | "
                                  f"Цена: ${signal.entry_price:.5f} | "
                                  f"Рынок: {signal.market_condition}")
                        
                        # Сохраняем сигнал
                        self.last_signals[symbol] = {
                            'signal': signal,
                            'timestamp': datetime.now(WARSAW_TZ)
                        }
                    else:
                        # Логируем причину отклонения
                        if signal is None:
                            logger.debug(f"⚪ {symbol}: Нет сигнала")
                        elif signal.confidence < self.MIN_CONFIDENCE_BASE:
                            logger.debug(f"⚪ {symbol}: Низкая уверенность ({signal.confidence:.1f}% < {self.MIN_CONFIDENCE_BASE}%)")
                        else:
                            logger.debug(f"⚪ {symbol}: Сигнал не прошел дополнительные фильтры")
                    
                    # Небольшая пауза между анализами
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка анализа {symbol}: {e}")
                    continue
            
            # Детальная статистика цикла
            # 🔴 ПРИОРИТЕТ 3.3: Структурированное логирование метрик цикла
            cycle_metrics = {
                'timestamp': datetime.now(WARSAW_TZ).isoformat(),
                'total_symbols': total_symbols,
                'excluded': excluded_count,
                'analyzed': analyzed_count,
                'signals_found': signals_found,
                'rejected': rejected_signals,
                'market_condition': market_condition,
                'open_positions': len(self.active_positions)
            }
            logger.info(f"✅ V4.0: Цикл завершен | {cycle_metrics}")
            
            # Выводим топ-5 близких кандидатов (даже если не прошли порог)
            if hasattr(self, 'candidates_list') and self.candidates_list:
                # Сортируем по уверенности
                sorted_candidates = sorted(self.candidates_list, key=lambda x: x['confidence'], reverse=True)
                top_candidates = sorted_candidates[:5]
                
                logger.info(f"📊 Топ-{len(top_candidates)} близких кандидатов (уверенность >= 70%):")
                for i, cand in enumerate(top_candidates, 1):
                    diff = cand['adaptive_min_confidence'] - cand['confidence']
                    status = "✅ ПРОШЕЛ" if cand['confidence'] >= cand['adaptive_min_confidence'] else f"⚠️ -{diff:.0f}%"
                    logger.info(f"   {i}. {cand['symbol']}: {cand['signal'].upper() if cand['signal'] else 'N/A'} | "
                              f"Уверенность: {cand['confidence']:.1f}% (требуется: {cand['adaptive_min_confidence']:.0f}%) | "
                              f"{status} | "
                              f"RSI={cand['rsi']:.0f} BB={cand['bb_position']:.0f}% Vol={cand['volume_ratio']:.1f}x")
                
                # Очищаем список для следующего цикла
                self.candidates_list = []
            else:
                logger.info("📊 Близких кандидатов не найдено (уверенность < 70%)")
            
            # 🧹 АВТОМАТИЧЕСКАЯ ОЧИСТКА ПАМЯТИ после каждого цикла анализа
            try:
                # Очищаем старые DataFrame из кэша (если есть)
                if hasattr(self, 'api_optimizer') and self.api_optimizer:
                    # Очищаем старый кэш (>5 минут)
                    self.api_optimizer.cache.clear_old_cache(max_age_hours=0.083)  # 5 минут
                
                # Очищаем старые записи из losing_symbols и failed_open_attempts
                now = datetime.now(WARSAW_TZ)
                symbols_to_remove = []
                for symbol_key, loss_info in self.losing_symbols.items():
                    loss_time = loss_info[1]
                    hours_since_loss = (now - loss_time).total_seconds() / 3600
                    if hours_since_loss >= 12:
                        symbols_to_remove.append(symbol_key)
                for symbol_key in symbols_to_remove:
                    del self.losing_symbols[symbol_key]
                
                attempts_to_remove = []
                for symbol_key, attempt_time in self.failed_open_attempts.items():
                    time_since_attempt = (now - attempt_time).total_seconds() / 60
                    if time_since_attempt >= 30:
                        attempts_to_remove.append(symbol_key)
                for symbol_key in attempts_to_remove:
                    del self.failed_open_attempts[symbol_key]
                
                # Очищаем старые сигналы из last_signals (оставляем только последние 50)
                if hasattr(self, 'last_signals') and len(self.last_signals) > 50:
                    # Сортируем по времени и оставляем только последние 50
                    sorted_signals = sorted(
                        self.last_signals.items(),
                        key=lambda x: x[1].get('timestamp', datetime.now(WARSAW_TZ)) if isinstance(x[1], dict) else datetime.now(WARSAW_TZ),
                        reverse=True
                    )
                    self.last_signals = dict(sorted_signals[:50])
                
                # Очищаем старые записи из daily_pnl_tracker (оставляем только последние 7 дней)
                if hasattr(self, 'daily_pnl_tracker'):
                    cutoff_date = (now - timedelta(days=7)).date().isoformat()
                    dates_to_remove = [date for date in self.daily_pnl_tracker.keys() if date < cutoff_date]
                    for date in dates_to_remove:
                        del self.daily_pnl_tracker[date]
                
                # 🔴 ПРИОРИТЕТ 3.2: Мониторинг утечек памяти
                if not hasattr(self, '_cleanup_counter'):
                    self._cleanup_counter = 0
                if not hasattr(self, '_memory_monitor'):
                    self._memory_monitor = {'peak_memory_mb': 0, 'last_check': datetime.now(WARSAW_TZ)}
                
                self._cleanup_counter += 1
                
                # Проверка памяти каждые 5 циклов
                if self._cleanup_counter % 5 == 0:
                    try:
                        import psutil
                        import os
                        process = psutil.Process(os.getpid())
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        
                        # Обновляем пик памяти
                        if memory_mb > self._memory_monitor['peak_memory_mb']:
                            self._memory_monitor['peak_memory_mb'] = memory_mb
                        
                        # Алерт при превышении 1.5GB
                        if memory_mb > 1500:
                            logger.warning(f"⚠️ ВЫСОКОЕ ПОТРЕБЛЕНИЕ ПАМЯТИ: {memory_mb:.0f}MB (лимит: 1500MB)")
                            if self.telegram_bot:
                                try:
                                    await self.send_telegram_v4(
                                        f"⚠️ ВЫСОКОЕ ПОТРЕБЛЕНИЕ ПАМЯТИ\n"
                                        f"Текущее: {memory_mb:.0f}MB\n"
                                        f"Пик: {self._memory_monitor['peak_memory_mb']:.0f}MB\n"
                                        f"Лимит: 1500MB"
                                    )
                                except:
                                    pass
                        
                        # Автоматическая очистка при превышении 1.5GB
                        if memory_mb > 1500:
                            logger.info("🧹 Запуск агрессивной очистки памяти...")
                            gc.collect()
                            gc.collect()
                            gc.collect()
                            # Очищаем кэши
                            if hasattr(self, 'api_optimizer') and self.api_optimizer:
                                self.api_optimizer.cache.clear_old_cache(max_age_hours=0.01)  # 1 минута
                            
                            # Проверяем память после очистки
                            memory_after = process.memory_info().rss / 1024 / 1024
                            freed = memory_mb - memory_after
                            logger.info(f"✅ Очистка памяти: освобождено {freed:.0f}MB (было: {memory_mb:.0f}MB, стало: {memory_after:.0f}MB)")
                    except ImportError:
                        logger.debug("⚠️ psutil не установлен, мониторинг памяти недоступен")
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка мониторинга памяти: {e}")
                
                # Более агрессивная очистка каждые 10 циклов
                if self._cleanup_counter >= 10:
                    # Первая очистка
                    collected1 = gc.collect()
                    
                    # Вторая очистка (более агрессивная)
                    collected2 = gc.collect()
                    
                    # Третья очистка (максимально агрессивная)
                    collected3 = gc.collect()
                    
                    total_collected = collected1 + collected2 + collected3
                    if total_collected > 0:
                        logger.info(f"🧹 Очистка памяти: удалено {total_collected} объектов (цикл #{self._cleanup_counter})")
                    
                    # Дополнительная очистка: удаляем ссылки на большие объекты
                    try:
                        # Очищаем кэш индикаторов если есть
                        if hasattr(self, '_indicators_cache'):
                            # Оставляем только последние 100 записей
                            if len(self._indicators_cache) > 100:
                                keys_to_remove = list(self._indicators_cache.keys())[:-100]
                                for key in keys_to_remove:
                                    del self._indicators_cache[key]
                        
                        # Очищаем кэш MTF данных если есть
                        if hasattr(self, '_mtf_cache'):
                            # Оставляем только последние 50 записей
                            if len(self._mtf_cache) > 50:
                                keys_to_remove = list(self._mtf_cache.keys())[:-50]
                                for key in keys_to_remove:
                                    del self._mtf_cache[key]
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка очистки кэшей: {e}")
                    
                    self._cleanup_counter = 0
                else:
                    # Лёгкая очистка каждый цикл
                    gc.collect(0)  # Только поколение 0
            except Exception as e:
                logger.debug(f"⚠️ Ошибка очистки памяти: {e}")
            
        except Exception as e:
            # 🔴 ПРИОРИТЕТ 2.3: Улучшенная обработка ошибок
            logger.error(f"❌ Ошибка торгового цикла V4.0: {e}", exc_info=True)
            # Пробрасываем ошибку вверх для retry логики
            raise
    
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
            
            message = f"""🚀 *БОТ ЗАПУЩЕН!*

💡 *Работает как TradeGPT*
🧠 *Disco57 (DiscoRL) обучение*

🎯 *TP: +1.15% (100% позиции) - компенсация комиссии, сразу в без убыток*
🛑 *SL: -${self.MAX_STOP_LOSS_USD:.2f} максимум*

💰 *Баланс*
💵 Всего: ${usdt_total:.2f}
💸 Свободно: ${usdt_free:.2f}

📈 *Торговля*
⚡ Сделка: ${self.POSITION_SIZE:.1f} x{self.LEVERAGE} = ${self.POSITION_NOTIONAL:.0f}
📌 Позиции: {active_positions}/{self.MAX_POSITIONS}

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
                logger.warning("⚠️ Telegram бот не инициализирован, сообщение не отправлено")
                return
            
            if not self.telegram_chat_id:
                logger.warning("⚠️ Telegram chat_id не установлен, сообщение не отправлено")
                return
                
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.debug(f"✅ Telegram сообщение отправлено успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сообщения V4.0: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

    async def send_position_opened_v4(self, symbol: str, side: str, entry_price: float, 
                                    amount_usdt: float, confidence: float, strategy_score: float):
        """V4.0: Уведомление об открытии позиции"""
        try:
            side_emoji = "🟢" if side == 'buy' else "🔴"
            direction = "LONG" if side == 'buy' else "SHORT"
            
            message = f"""
{side_emoji} **ПОЗИЦИЯ ОТКРЫТА**

💎 **{symbol}**
📊 {direction} | ${entry_price:.5f}
💰 Размер: ${amount_usdt:.0f} (20x)

🎯 **TP: +1.15% (100% позиции) - компенсация комиссии, сразу в без убыток**
🛑 **SL: -${self.MAX_STOP_LOSS_USD:.2f} максимум**

⏰ {datetime.now(WARSAW_TZ).strftime('%H:%M:%S %d.%m.%Y')}
"""
            
            await self.send_telegram_v4(message)
            logger.info(f"✅ V4.0: Уведомление об открытии {symbol} отправлено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка уведомления об открытии V4.0: {e}")

    async def send_position_closed_v4(self, symbol: str, side: str, entry_price: float, 
                                    exit_price: float, pnl_percent: float, profit_usd: float, 
                                    reason: str):
        """V4.0: Уведомление о закрытии позиции с защитой от дублирования и данными с биржи"""
        try:
            # 🚨 ЗАЩИТА ОТ ДУБЛИРОВАНИЯ: Проверяем, не отправляли ли уже уведомление
            symbol_norm = self.normalize_symbol(symbol)
            now = datetime.now(WARSAW_TZ)
            
            # Проверяем, было ли уже отправлено уведомление за последние 10 минут
            if symbol_norm in self.sent_close_notifications:
                last_sent = self.sent_close_notifications[symbol_norm]
                time_diff = (now - last_sent).total_seconds()
                if time_diff < 600:  # 10 минут
                    logger.debug(f"⏭️ {symbol}: Уведомление о закрытии уже отправлено {time_diff:.0f} сек назад, пропускаем")
                    return
            
            # 🔍 КРИТИЧНО: ВСЕГДА получаем данные с биржи для точности
            # Даже если данные переданы, проверяем их через API
            real_entry = entry_price
            real_exit = exit_price
            real_pnl_usd = profit_usd
            real_pnl_percent = pnl_percent
            position_size = 0.0
            position_notional = self.POSITION_NOTIONAL
            commission = 0.0
            hold_duration = "N/A"
            
            try:
                from pybit.unified_trading import HTTP
                session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False, recv_window=5000, timeout=10)
                bybit_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                
                # Получаем последнюю закрытую позицию
                cp = session.get_closed_pnl(category='linear', symbol=bybit_symbol, limit=1)
                items = cp.get('result', {}).get('list', []) or []
                
                if items:
                    it = items[0]
                    
                    # Получаем данные из API
                    api_entry = float(it.get('avgEntryPrice', 0) or 0)
                    api_exit = float(it.get('avgExitPrice', 0) or 0)
                    api_pnl = float(it.get('closedPnl', 0) or 0)
                    api_qty = float(it.get('qty', 0) or 0)
                    api_side = it.get('side', 'Buy')
                    
                    # Используем данные из API если они валидны
                    if api_entry > 0:
                        real_entry = api_entry
                    if api_exit > 0 and api_exit != api_entry:
                        real_exit = api_exit
                    if api_pnl != 0:
                        real_pnl_usd = api_pnl
                    
                    # Рассчитываем размер позиции
                    if api_qty > 0:
                        position_size = api_qty
                        position_notional = api_entry * api_qty if api_entry > 0 else position_notional
                    
                    # Рассчитываем процент PnL от нотионала
                    if position_notional > 0:
                        real_pnl_percent = (real_pnl_usd / position_notional) * 100
                    
                    # Рассчитываем комиссию (примерно 0.06% от нотионала за вход и выход)
                    if position_notional > 0:
                        commission = position_notional * 0.0006 * 2  # 0.06% * 2 (вход + выход)
                    
                    # Получаем время удержания
                    created_time = int(it.get('createdTime', 0) or 0)
                    updated_time = int(it.get('updatedTime', 0) or 0)
                    if created_time > 0 and updated_time > 0:
                        duration_seconds = (updated_time - created_time) / 1000
                        if duration_seconds < 60:
                            hold_duration = f"{int(duration_seconds)} сек"
                        elif duration_seconds < 3600:
                            hold_duration = f"{int(duration_seconds / 60)} мин"
                        else:
                            hours = int(duration_seconds / 3600)
                            minutes = int((duration_seconds % 3600) / 60)
                            hold_duration = f"{hours}ч {minutes}мин"
                    
                    logger.info(f"✅ {symbol}: Данные получены с биржи | Entry: ${real_entry:.5f}, Exit: ${real_exit:.5f}, PnL: ${real_pnl_usd:.2f} ({real_pnl_percent:+.2f}%)")
                else:
                    logger.warning(f"⚠️ {symbol}: Закрытых позиций не найдено в API, используем переданные данные")
                    
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: Не удалось получить данные из API: {e}, используем переданные данные")
            
            # Валидация финальных данных
            if real_entry <= 0 or real_exit <= 0:
                logger.error(f"🚨 {symbol}: НЕВЕРНЫЕ ДАННЫЕ после получения с биржи! Entry: {real_entry}, Exit: {real_exit}")
                return  # Не отправляем сообщение с неверными данными
            
            # Рассчитываем ROE (Return on Equity)
            roe_percent = (real_pnl_usd / (position_notional / self.LEVERAGE)) * 100 if position_notional > 0 and self.LEVERAGE > 0 else 0
            
            result_emoji = "💰" if real_pnl_percent > 0 else "💸"
            direction = "LONG" if (side.lower() == 'buy' or side.lower() == 'long') else "SHORT"
            
            # Формируем полное сообщение с данными с биржи
            message = f"""
{result_emoji} **ПОЗИЦИЯ ЗАКРЫТА V4.0**

💎 **{symbol}** {direction}
📥 Вход: ${real_entry:.5f}
📤 Выход: ${real_exit:.5f}

💹 **Результат:**
{'+' if real_pnl_percent > 0 else ''}{real_pnl_percent:.2f}% | ${'+' if real_pnl_usd > 0 else ''}{real_pnl_usd:.2f}
📊 ROE: {roe_percent:+.1f}%

📦 **Детали:**
💰 Размер: {position_size:.4f} контрактов (${position_notional:.2f})
💸 Комиссия: ~${commission:.3f}
⏱️ Удержание: {hold_duration}

📋 **Причина:** {reason}
⏰ {datetime.now(WARSAW_TZ).strftime('%H:%M:%S %d.%m.%Y')}
"""
            
            # Убеждаемся, что telegram_bot инициализирован
            if not self.telegram_bot:
                logger.error(f"❌ {symbol}: Telegram бот не инициализирован! Сообщение не отправлено.")
                return
            
            await self.send_telegram_v4(message)
            
            # Сохраняем время отправки для защиты от дублирования
            self.sent_close_notifications[symbol_norm] = now
            
            # Очищаем старые записи (старше 1 часа)
            cutoff_time = now - timedelta(hours=1)
            self.sent_close_notifications = {
                k: v for k, v in self.sent_close_notifications.items() 
                if v > cutoff_time
            }
            
            logger.info(f"✅ V4.0: Уведомление о закрытии {symbol} отправлено | Entry: ${real_entry:.5f}, Exit: ${real_exit:.5f}, PnL: ${real_pnl_usd:.2f} ({real_pnl_percent:+.2f}%), ROE: {roe_percent:+.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Ошибка уведомления о закрытии V4.0 для {symbol}: {e}", exc_info=True)

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

    async def _get_trade_statistics_from_api(self, hours: int = None, days: int = None) -> dict:
        """
        Получает статистику сделок из Bybit API за указанный период
        
        Args:
            hours: Количество часов назад (для расчета за 24 часа)
            days: Количество дней назад (для расчета за 7 дней)
        
        Returns:
            dict с ключами: total_trades, winning_trades, losing_trades, win_rate, total_pnl
        """
        try:
            from pybit.unified_trading import HTTP
            session = HTTP(api_key=self.api_key, api_secret=self.api_secret, testnet=False)
            
            # Определяем временной диапазон
            end_time = int(datetime.now().timestamp() * 1000)
            if hours:
                start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            elif days:
                start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            else:
                # Все время - получаем последние 1000 сделок
                start_time = None
            
            # Получаем закрытые сделки (увеличиваем лимит для более полных данных)
            params = {'category': 'linear', 'limit': 200}  # Увеличено с 100 до 200 для более полных данных
            if start_time:
                params['startTime'] = start_time
                params['endTime'] = end_time
            
            cp = session.get_closed_pnl(**params)
            items = cp.get('result', {}).get('list', []) or []
            
            if not items:
                return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0}
            
            # Фильтруем по времени если указано (используем updatedTime - время закрытия позиции)
            if start_time:
                filtered_items = []
                for item in items:
                    updated_time = int(item.get('updatedTime', 0) or 0)
                    # Позиция должна быть закрыта в указанном диапазоне
                    if updated_time >= start_time and updated_time <= end_time:
                        filtered_items.append(item)
                items = filtered_items
            
            # Рассчитываем статистику
            total_trades = len(items)
            winning_trades = sum(1 for item in items if float(item.get('closedPnl', 0)) > 0)
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            total_pnl = sum(float(item.get('closedPnl', 0)) for item in items)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики из API: {e}")
            return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0}
    
    async def send_daily_report_v4(self):
        """V4.0: Ежедневный отчет в 9:00"""
        try:
            # Получаем баланс (Bybit Unified Account) с fallback на pybit
            usdt_total = 0.0
            usdt_free = 0.0
            usdt_used = 0.0
            
            try:
                balance = await self.exchange.fetch_balance({'accountType': 'UNIFIED'})
                usdt_info = balance.get('USDT', {})
                usdt_total = usdt_info.get('total') if isinstance(usdt_info, dict) else 0
                usdt_free = usdt_info.get('free') or usdt_total if isinstance(usdt_info, dict) else usdt_total
                usdt_used = usdt_info.get('used') or 0 if isinstance(usdt_info, dict) else 0
            except Exception as e:
                logger.debug(f"Ошибка получения баланса через ccxt: {e}")
                # Fallback на pybit
                try:
                    from pybit.unified_trading import HTTP
                    session = HTTP(testnet=False, api_key=self.api_key, api_secret=self.api_secret)
                    r = session.get_wallet_balance(accountType='UNIFIED', coin='USDT')
                    coin_info = r.get('result', {}).get('list', [{}])[0].get('coin', [{}])[0]
                    usdt_total = float(coin_info.get('walletBalance', 0.0))
                    usdt_free = float(coin_info.get('availableToWithdraw', 0.0))
                    usdt_used = usdt_total - usdt_free
                except Exception as e2:
                    logger.error(f"❌ Ошибка получения баланса через pybit fallback: {e2}")
            
            # Получаем статистику из API за разные периоды (ПРИОРИТЕТ: данные с биржи)
            stats_24h = await self._get_trade_statistics_from_api(hours=24)
            stats_7d = await self._get_trade_statistics_from_api(days=7)
            stats_all = await self._get_trade_statistics_from_api()  # Все время
            
            # Статистика за сегодня (из API, а не из performance_stats)
            # Получаем статистику за последние 24 часа как "сегодня"
            today_stats = await self._get_trade_statistics_from_api(hours=24)
            today_trades = today_stats.get('total_trades', 0)
            today_winning = today_stats.get('winning_trades', 0)
            today_pnl = today_stats.get('total_pnl', 0.0)
            today_win_rate = today_stats.get('win_rate', 0.0)
            
            # Если нет данных за 24 часа, используем performance_stats как fallback
            if today_trades == 0:
                today_trades = self.performance_stats.get('total_trades', 0)
                today_winning = self.performance_stats.get('winning_trades', 0)
                today_pnl = self.performance_stats.get('total_pnl', 0.0)
                today_win_rate = (today_winning / today_trades * 100) if today_trades > 0 else 0.0
            
            # Активные позиции (получаем с биржи для точности)
            try:
                active_positions = await self._get_current_open_positions_count()
            except Exception as e:
                logger.debug(f"⚠️ Ошибка получения активных позиций для отчёта: {e}")
                active_positions = len(self.active_positions)  # Fallback на локальный словарь
            
            # Определяем статус на основе win rate за 7 дней
            wr_7d = stats_7d['win_rate']
            if wr_7d < 55:
                status_emoji = "🔴"
                status_text = "Критично"
            elif wr_7d < 60:
                status_emoji = "🟡"
                status_text = "Предупреждение"
            elif wr_7d < 65:
                status_emoji = "🟢"
                status_text = "Хороший уровень"
            else:
                status_emoji = "✅"
                status_text = "Отлично"
            
            # Отчет WIN RATE
            win_rate_report = f"""
📊 **ЕЖЕДНЕВНЫЙ ОТЧЕТ WIN RATE**

🕐 **Последние 24 часа:**
• WR: {stats_24h['win_rate']:.1f}% ({stats_24h['winning_trades']}W/{stats_24h['losing_trades']}L)

📅 **Последние 7 дней:**
• WR: {stats_7d['win_rate']:.1f}% ({stats_7d['winning_trades']}W/{stats_7d['losing_trades']}L)

📈 **Всего:**
• WR: {stats_all['win_rate']:.1f}% ({stats_all['winning_trades']}W/{stats_all['losing_trades']}L)

🎯 **Целевые показатели:**
• Критический уровень: < 55%
• Предупреждение: < 60%
• Хороший уровень: >= 65%
• Безубыток: >= 57%

{status_emoji} **Текущий статус:** {status_text}
"""
            
            # Отчет V4.0 PRO
            report = f"""
📊 **ЕЖЕДНЕВНЫЙ ОТЧЕТ V4.0 PRO**

💰 **Баланс:**
💵 Всего: ${usdt_total:.2f}
💸 Свободно: ${usdt_free:.2f}
🔒 В торговле: ${usdt_used:.2f}

📈 **Статистика за сегодня (24ч):**
🎯 Сделок: {today_trades}
✅ Прибыльных: {today_winning}
❌ Убыточных: {today_trades - today_winning}
📊 Винрейт: {today_win_rate:.1f}%
💹 Общий PnL: ${today_pnl:.2f} (с биржи)

🔄 **Активные позиции:** {active_positions}/3

🤖 **Системы V4.0:**
✅ ProbabilityCalculator
✅ StrategyEvaluator  
✅ RealismValidator
✅ AI+ML Adaptive (Disco57/DiscoRL)
✅ 4 таймфрейма (5m-15m-30m-1h)
✅ TP: +1.15% (100% позиции) - компенсация комиссии, сразу в без убыток

📅 {datetime.now(WARSAW_TZ).strftime('%d.%m.%Y')} | ⏰ {datetime.now(WARSAW_TZ).strftime('%H:%M')}

**Super Bot V4.0 PRO работает стабильно!** 🚀
"""
            
            # Отправляем оба отчета
            await self.send_telegram_v4(win_rate_report)
            await asyncio.sleep(1)  # Небольшая задержка между сообщениями
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
            # 🚀 СКАЛЬПЕРСКИЙ РЕЖИМ: более частый анализ для быстрых входов
            self.scheduler.add_job(
                self.trading_loop_v4,
                'interval',
                minutes=5,  # СКАЛЬПИНГ: было 15 минут, стало 5 минут
                id='trading_loop_v4'
            )
            
            # 📊 Мониторинг позиций (каждые 10 секунд для быстрой реакции на SL)
            # 🚨 КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Уменьшено до 10 секунд для защиты от убытков
            self.scheduler.add_job(
                self.monitor_positions,
                'interval',
                seconds=10,
                id='monitor_positions'
            )
            
            # 📊 Ежедневный отчёт V4.0 в 09:00 Warsaw (Europe/Warsaw = UTC+1/+2)
            try:
                from pytz import timezone as tz
                warsaw_tz = tz('Europe/Warsaw')
                
                # Удаляем старую задачу если есть
                try:
                    self.scheduler.remove_job('daily_report_v4')
                except:
                    pass
                
                self.scheduler.add_job(
                    self.send_daily_report_v4,
                    'cron',
                    hour=9,
                    minute=0,
                    timezone=warsaw_tz,
                    id='daily_report_v4',
                    replace_existing=True
                )
                logger.info(f"✅ V4.0: Ежедневный отчет настроен на 09:00 (Warsaw time, UTC+1/+2)")
                
                # Логируем следующее выполнение
                try:
                    job = self.scheduler.get_job('daily_report_v4')
                    if job and hasattr(job, 'next_run_time') and job.next_run_time:
                        next_run = job.next_run_time
                        logger.info(f"📅 Следующий ежедневный отчет: {next_run}")
                except Exception:
                    pass  # Игнорируем ошибки при получении времени следующего запуска
            except ImportError:
                logger.warning("⚠️ pytz не установлен, ежедневный отчет отключен")
            except Exception as e:
                logger.error(f"❌ Ошибка настройки ежедневного отчета: {e}")
            
            # Запуск планировщика
            self.scheduler.start()
            logger.info("✅ V4.0: Планировщик запущен (анализ: 15мин, MTF: 5m/15m/30m/1h, отчет: 09:00)")
            
            # Запуск Telegram бота для обработки команд (если есть)
            if self.application:
                try:
                    # Запускаем polling в отдельной задаче для избежания блокировки
                    async def run_telegram_polling():
                        retry_count = 0
                        max_retries = 5
                        while retry_count < max_retries:
                            try:
                                # Проверяем, не запущен ли уже polling
                                if hasattr(self.application, 'updater') and self.application.updater.running:
                                    logger.warning("⚠️ Telegram polling уже запущен, пропускаем повторный запуск")
                                    retry_count = 0
                                    break
                                
                                await self.application.initialize()
                                await self.application.start()
                                await self.application.updater.start_polling(
                                    drop_pending_updates=True,
                                    allowed_updates=None
                                )
                                retry_count = 0  # Сброс счетчика при успехе
                                logger.info("✅ Telegram polling успешно запущен")
                                
                                # Бесконечный цикл для поддержания polling (выходим только при ошибке)
                                try:
                                    while True:
                                        await asyncio.sleep(60)  # Проверяем каждую минуту
                                        # Проверяем, что polling все еще активен
                                        if not hasattr(self.application, 'updater') or not self.application.updater.running:
                                            logger.warning("⚠️ Telegram polling остановлен, перезапускаем...")
                                            break
                                except asyncio.CancelledError:
                                    logger.info("🛑 Telegram polling отменен")
                                    raise
                                except Exception as e:
                                    error_msg = str(e)
                                    if "409" in error_msg or "Conflict" in error_msg:
                                        logger.warning(f"⚠️ Telegram 409 Conflict в цикле polling: {e}")
                                        logger.info("🔄 Ожидание 30 секунд перед перезапуском...")
                                        await asyncio.sleep(30)
                                    else:
                                        logger.error(f"❌ Ошибка в цикле Telegram polling: {e}")
                                    break  # Выходим из цикла для перезапуска
                            except Exception as e:
                                error_msg = str(e)
                                # Если Application уже запущен, это не критично
                                if "already running" in error_msg.lower() or "already started" in error_msg.lower():
                                    logger.warning(f"⚠️ Telegram Application уже запущен: {e}")
                                    retry_count = 0  # Не считаем это ошибкой
                                    break
                                # Обработка 409 Conflict (несколько экземпляров polling)
                                elif "409" in error_msg or "Conflict" in error_msg or "terminated by other getUpdates" in error_msg.lower():
                                    logger.warning(f"⚠️ Telegram 409 Conflict обнаружен: {e}")
                                    logger.info("🔄 Ожидание 30 секунд перед повторной попыткой...")
                                    await asyncio.sleep(30)  # Ждем, чтобы другой экземпляр завершил polling
                                    retry_count += 1
                                    if retry_count >= max_retries:
                                        logger.error("❌ Telegram polling остановлен: слишком много конфликтов")
                                        break
                                else:
                                    retry_count += 1
                                    logger.error(f"❌ Ошибка Telegram polling (попытка {retry_count}/{max_retries}): {e}")
                                    if retry_count < max_retries:
                                        await asyncio.sleep(60)  # Ждем перед повтором
                                    else:
                                        logger.error("❌ Telegram polling остановлен после максимального числа попыток")
                    
                    polling_task = asyncio.create_task(run_telegram_polling())
                    logger.info("✅ Telegram бот запущен и готов к командам")
                except Exception as e:
                    logger.error(f"❌ Ошибка запуска Telegram бота: {e}")
            
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
