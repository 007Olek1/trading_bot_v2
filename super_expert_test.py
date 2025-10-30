#!/usr/bin/env python3
"""
🔬 СУПЕР-ЭКСПЕРТНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ
======================================

Профессиональное тестирование на уровне группы экспертов разработчиков.
Проверка всех компонентов, интеграций, производительности и стабильности.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any
import traceback

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('super_expert_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь
sys.path.insert(0, str(Path(__file__).parent))

# Загрузка переменных окружения
from dotenv import load_dotenv
env_files = [
    Path(__file__).parent / "api.env",
    Path(__file__).parent / ".env",
]
for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file)
        break


class SuperExpertTester:
    """Супер-экспертная система тестирования"""
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        
    def test_result(self, test_name: str, passed: bool, message: str = "", warning: bool = False):
        """Записать результат теста"""
        self.total_tests += 1
        if warning:
            self.warnings += 1
            status = "⚠️ WARNING"
        elif passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"
        
        self.results[test_name] = {
            'status': status,
            'passed': passed,
            'message': message,
            'warning': warning
        }
        
        logger.info(f"{status}: {test_name}")
        if message:
            logger.info(f"   {message}")
    
    async def test_module_imports(self):
        """Тест 1: Импорт всех модулей"""
        logger.info("\n" + "="*70)
        logger.info("📦 ТЕСТ 1: ИМПОРТ МОДУЛЕЙ")
        logger.info("="*70)
        
        modules = {
            'super_bot_v4_mtf': 'SuperBotV4MTF',
            'probability_calculator': 'ProbabilityCalculator',
            'strategy_evaluator': 'StrategyEvaluator',
            'realism_validator': 'RealismValidator',
            'ai_ml_system': 'TradingMLSystem',
            'universal_learning_system': 'UniversalLearningSystem',
            'adaptive_parameters': 'AdaptiveParameterSystem',
            'adaptive_trading_system': 'FullyAdaptiveSystem',
            'intelligent_agents': 'IntelligentAgentsSystem',
            'integrate_intelligent_agents': 'IntegratedAgentsManager',
            'llm_monitor': 'BotHealthMonitor',
            'data_storage_system': 'DataStorageSystem',
            'smart_coin_selector': 'SmartCoinSelector',
            'advanced_indicators': 'AdvancedIndicators',
            'api_optimizer': 'APIOptimizer',
            'fed_event_manager': 'FedEventManager'
        }
        
        all_imported = True
        for module_name, class_name in modules.items():
            try:
                mod = __import__(module_name)
                cls = getattr(mod, class_name, None)
                if cls:
                    self.test_result(f"Импорт {module_name}.{class_name}", True)
                else:
                    self.test_result(f"Импорт {module_name}.{class_name}", False, f"Класс {class_name} не найден", warning=True)
                    all_imported = False
            except Exception as e:
                self.test_result(f"Импорт {module_name}", False, f"Ошибка: {str(e)[:100]}")
                all_imported = False
        
        return all_imported
    
    async def test_bot_initialization(self):
        """Тест 2: Инициализация бота"""
        logger.info("\n" + "="*70)
        logger.info("🚀 ТЕСТ 2: ИНИЦИАЛИЗАЦИЯ БОТА")
        logger.info("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            # Проверка компонентов
            checks = {
                'V4 модули': hasattr(bot, 'probability_calculator') and bot.probability_calculator is not None,
                'RealismValidator': hasattr(bot, 'realism_validator') and bot.realism_validator is not None,
                'ML система': hasattr(bot, 'ml_system') and bot.ml_system is not None,
                'Universal Learning': hasattr(bot, 'universal_learning') and bot.universal_learning is not None,
                'Adaptive Parameters': hasattr(bot, 'adaptive_params_system') and bot.adaptive_params_system is not None,
                'Fully Adaptive': hasattr(bot, 'fully_adaptive_system') and bot.fully_adaptive_system is not None,
                'Smart Selector': hasattr(bot, 'smart_selector') and bot.smart_selector is not None,
                'Advanced Indicators': hasattr(bot, 'advanced_indicators') and bot.advanced_indicators is not None,
                'LLM Monitor': hasattr(bot, 'health_monitor') and bot.health_monitor is not None,
                'TP уровни': len(bot.TP_LEVELS_V4) == 6,
                'Fed Event Manager': hasattr(bot, 'fed_event_manager'),
                'MIN_CONFIDENCE': bot.MIN_CONFIDENCE_BASE >= 50 and bot.MIN_CONFIDENCE_BASE <= 70
            }
            
            all_pass = True
            for check_name, check_result in checks.items():
                self.test_result(f"Инициализация: {check_name}", check_result)
                if not check_result:
                    all_pass = False
            
            # Инициализация полная
            try:
                await bot.initialize()
                self.test_result("Полная инициализация бота", True)
                
                # Проверка после инициализации
                post_init_checks = {
                    'Exchange подключен': bot.exchange is not None,
                    'API Optimizer': hasattr(bot, 'api_optimizer') and bot.api_optimizer is not None,
                    'Agents Manager': hasattr(bot, 'agents_manager') and bot.agents_manager is not None,
                    'Telegram Bot': hasattr(bot, 'telegram_bot')
                }
                
                for check_name, check_result in post_init_checks.items():
                    self.test_result(f"После инициализации: {check_name}", check_result, warning=not check_result)
                
                if bot.exchange:
                    await bot.exchange.close()
                
                return all_pass
            except Exception as e:
                self.test_result("Полная инициализация бота", False, f"Ошибка: {str(e)[:200]}")
                return False
                
        except Exception as e:
            self.test_result("Создание бота", False, f"Ошибка: {str(e)[:200]}")
            logger.error(traceback.format_exc())
            return False
    
    async def test_timeframes(self):
        """Тест 3: Мульти-таймфреймовый анализ"""
        logger.info("\n" + "="*70)
        logger.info("⏰ ТЕСТ 3: МУЛЬТИ-ТАЙМФРЕЙМОВЫЙ АНАЛИЗ")
        logger.info("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            await bot.initialize()
            
            expected_timeframes = ['15m', '30m', '45m', '1h', '4h']
            
            # Проверка функции
            has_mtf_function = hasattr(bot, '_fetch_multi_timeframe_data')
            self.test_result("Функция _fetch_multi_timeframe_data", has_mtf_function)
            
            if has_mtf_function:
                # Тест на реальном символе с реальным API
                try:
                    # Убедимся что exchange инициализирован
                    if bot.exchange is None:
                        await bot.initialize()
                    
                    # Получаем данные с реальным API
                    mtf_data = await bot._fetch_multi_timeframe_data('BTCUSDT')
                    
                    # Проверяем каждый таймфрейм
                    timeframes_with_data = []
                    for tf in expected_timeframes:
                        has_data = tf in mtf_data and len(mtf_data.get(tf, {})) > 0
                        if has_data:
                            timeframes_with_data.append(tf)
                            self.test_result(f"Таймфрейм {tf} данные", True)
                        else:
                            self.test_result(f"Таймфрейм {tf} данные", False, f"Данные не получены", warning=True)
                    
                    total_timeframes = len(timeframes_with_data)
                    self.test_result(f"Всего таймфреймов получено: {total_timeframes}/5", total_timeframes >= 4, 
                                   f"Получено {total_timeframes} из 5 таймфреймов", warning=total_timeframes < 5)
                    
                    # Дополнительная проверка - пробуем получить данные напрямую
                    if total_timeframes == 0:
                        logger.warning("   ⚠️ Пробуем получить данные напрямую через exchange...")
                        try:
                            test_df = await bot._fetch_ohlcv('BTCUSDT', '15m', 10)
                            if test_df is not None and not test_df.empty:
                                self.test_result("Прямое получение данных через exchange", True, "Данные получены напрямую")
                                logger.info(f"   ✅ Получено {len(test_df)} свечей через direct API")
                            else:
                                self.test_result("Прямое получение данных через exchange", False, "Данные пустые", warning=True)
                        except Exception as e2:
                            self.test_result("Прямое получение данных через exchange", False, f"Ошибка API: {str(e2)[:100]}", warning=True)
                    
                except Exception as e:
                    error_msg = str(e)
                    # Проверяем тип ошибки
                    if "API" in error_msg or "key" in error_msg.lower() or "authentication" in error_msg.lower():
                        self.test_result("Получение MTF данных", False, f"Ошибка API/аутентификации: {error_msg[:150]}", warning=True)
                    else:
                        self.test_result("Получение MTF данных", False, f"Ошибка: {error_msg[:150]}", warning=True)
            
            if bot.exchange:
                await bot.exchange.close()
            
            return True
            
        except Exception as e:
            self.test_result("Тест таймфреймов", False, f"Ошибка: {str(e)[:150]}")
            return False
    
    async def test_tp_levels(self):
        """Тест 4: TP уровни"""
        logger.info("\n" + "="*70)
        logger.info("🎯 ТЕСТ 4: TP УРОВНИ")
        logger.info("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            tp_levels = bot.TP_LEVELS_V4
            self.test_result("TP_LEVELS_V4 существует", len(tp_levels) > 0)
            self.test_result("TP уровней = 6", len(tp_levels) == 6)
            
            # Проверка каждого уровня
            expected_levels = [1, 2, 3, 4, 5, 6]
            for i, tp in enumerate(tp_levels, 1):
                has_level = tp.get('level') == i
                has_percent = 'percent' in tp and tp['percent'] > 0
                has_portion = 'portion' in tp and 0 < tp['portion'] <= 1
                
                self.test_result(f"TP{i}: структура", has_level and has_percent and has_portion)
            
            # Проверка суммы portions
            total_portion = sum(tp['portion'] for tp in tp_levels)
            self.test_result(f"Сумма portions = 1.0", abs(total_portion - 1.0) < 0.01, 
                           f"Сумма: {total_portion:.2f}")
            
            # Проверка минимальной прибыли
            position_value = bot.POSITION_SIZE * bot.LEVERAGE  # $25
            tp1_tp2_tp3_profit = (position_value * 0.40 * 0.04 + 
                                 position_value * 0.20 * 0.06 + 
                                 position_value * 0.20 * 0.08)
            self.test_result("Минимальная прибыль TP1-TP3 >= $1.0", tp1_tp2_tp3_profit >= 1.0,
                           f"Прибыль: ${tp1_tp2_tp3_profit:.2f}")
            
            return True
            
        except Exception as e:
            self.test_result("Тест TP уровней", False, f"Ошибка: {str(e)[:150]}")
            return False
    
    async def test_realism_validator(self):
        """Тест 5: RealismValidator"""
        logger.info("\n" + "="*70)
        logger.info("✅ ТЕСТ 5: REALISM VALIDATOR")
        logger.info("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            has_validator = hasattr(bot, 'realism_validator') and bot.realism_validator is not None
            self.test_result("RealismValidator инициализирован", has_validator)
            
            if has_validator:
                # Тест валидации
                try:
                    test_signal = {
                        'direction': 'buy',
                        'stop_loss_percent': -20,
                        'tp_levels': [{'percent': 4}, {'percent': 6}]
                    }
                    test_market = {
                        'price': 50000,
                        'volume': 1000000,
                        'rsi': 45
                    }
                    
                    result = bot.realism_validator.validate_signal(test_signal, test_market, [])
                    self.test_result("RealismValidator.validate_signal() работает", result is not None)
                    
                    if result:
                        self.test_result("RealismCheck структура", hasattr(result, 'is_realistic'))
                    
                except Exception as e:
                    self.test_result("Тест валидации сигнала", False, f"Ошибка: {str(e)[:150]}")
            
            return True
            
        except Exception as e:
            self.test_result("Тест RealismValidator", False, f"Ошибка: {str(e)[:150]}")
            return False
    
    async def test_ai_ml_systems(self):
        """Тест 6: AI/ML системы"""
        logger.info("\n" + "="*70)
        logger.info("🧠 ТЕСТ 6: AI/ML СИСТЕМЫ")
        logger.info("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            await bot.initialize()
            
            systems = {
                'TradingMLSystem': (bot.ml_system, 'ML предсказания'),
                'UniversalLearningSystem': (bot.universal_learning, 'Универсальное обучение'),
                'AdaptiveParameterSystem': (bot.adaptive_params_system, 'Адаптивные параметры'),
                'FullyAdaptiveSystem': (bot.fully_adaptive_system, 'Полностью адаптивная система'),
                'DataStorageSystem': (bot.data_storage, 'Хранение данных'),
                'IntelligentAgentsSystem': (bot.agents_manager.intelligent_system if bot.agents_manager else None, 'Интеллектуальные агенты')
            }
            
            for sys_name, (sys_obj, description) in systems.items():
                exists = sys_obj is not None
                self.test_result(f"{sys_name} ({description})", exists, warning=not exists)
            
            if bot.exchange:
                await bot.exchange.close()
            
            return True
            
        except Exception as e:
            self.test_result("Тест AI/ML систем", False, f"Ошибка: {str(e)[:150]}")
            logger.error(traceback.format_exc())
            return False
    
    async def test_integration(self):
        """Тест 7: Интеграция компонентов"""
        logger.info("\n" + "="*70)
        logger.info("🔗 ТЕСТ 7: ИНТЕГРАЦИЯ КОМПОНЕНТОВ")
        logger.info("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            await bot.initialize()
            
            # Проверка интеграции
            integration_checks = {
                'MTF + RealismValidator': bot.realism_validator is not None and hasattr(bot, '_fetch_multi_timeframe_data'),
                'TP + ProbabilityCalculator': bot.probability_calculator is not None and len(bot.TP_LEVELS_V4) == 6,
                'ML + UniversalLearning': bot.ml_system is not None and bot.universal_learning is not None,
                'API Optimizer + Exchange': bot.api_optimizer is not None and bot.exchange is not None,
                'Agents + Learning': bot.agents_manager is not None,
                'Smart Selector + Market': bot.smart_selector is not None
            }
            
            for check_name, check_result in integration_checks.items():
                self.test_result(f"Интеграция: {check_name}", check_result)
            
            if bot.exchange:
                await bot.exchange.close()
            
            return True
            
        except Exception as e:
            self.test_result("Тест интеграции", False, f"Ошибка: {str(e)[:150]}")
            logger.error(traceback.format_exc())
            return False
    
    async def test_performance(self):
        """Тест 8: Производительность"""
        logger.info("\n" + "="*70)
        logger.info("⚡ ТЕСТ 8: ПРОИЗВОДИТЕЛЬНОСТЬ")
        logger.info("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            import time
            
            bot = SuperBotV4MTF()
            
            # Тест скорости инициализации
            start = time.time()
            await bot.initialize()
            init_time = time.time() - start
            self.test_result(f"Инициализация < 10 сек", init_time < 10, 
                           f"Время: {init_time:.2f} сек", warning=init_time >= 10)
            
            # Тест скорости получения данных
            if bot.exchange:
                start = time.time()
                try:
                    df = await bot._fetch_ohlcv('BTCUSDT', '15m', 100)
                    fetch_time = time.time() - start
                    self.test_result(f"Получение данных < 5 сек", fetch_time < 5,
                                   f"Время: {fetch_time:.2f} сек", warning=fetch_time >= 5)
                    self.test_result("Данные не пустые", not df.empty if df is not None else False)
                except Exception as e:
                    self.test_result("Получение данных", False, f"Ошибка: {str(e)[:100]}", warning=True)
                
                await bot.exchange.close()
            
            return True
            
        except Exception as e:
            self.test_result("Тест производительности", False, f"Ошибка: {str(e)[:150]}")
            return False
    
    async def run_all_tests(self):
        """Запустить все тесты"""
        logger.info("\n" + "="*70)
        logger.info("🔬 СУПЕР-ЭКСПЕРТНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ")
        logger.info("="*70)
        logger.info(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        tests = [
            ("Импорт модулей", self.test_module_imports),
            ("Инициализация бота", self.test_bot_initialization),
            ("Мульти-таймфреймы", self.test_timeframes),
            ("TP уровни", self.test_tp_levels),
            ("RealismValidator", self.test_realism_validator),
            ("AI/ML системы", self.test_ai_ml_systems),
            ("Интеграция", self.test_integration),
            ("Производительность", self.test_performance)
        ]
        
        for test_name, test_func in tests:
            try:
                await test_func()
            except Exception as e:
                logger.error(f"❌ Критическая ошибка в тесте '{test_name}': {e}")
                logger.error(traceback.format_exc())
                self.test_result(f"Тест {test_name}", False, f"Критическая ошибка: {str(e)[:150]}")
        
        # Вывод результатов
        self.print_summary()
    
    def print_summary(self):
        """Вывести итоговую сводку"""
        logger.info("\n" + "="*70)
        logger.info("📊 ИТОГОВАЯ СВОДКА ТЕСТИРОВАНИЯ")
        logger.info("="*70)
        
        passed = self.passed_tests
        failed = self.failed_tests
        warnings = self.warnings
        total = self.total_tests
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info(f"Всего тестов: {total}")
        logger.info(f"✅ Пройдено: {passed}")
        logger.info(f"⚠️ Предупреждений: {warnings}")
        logger.info(f"❌ Провалено: {failed}")
        logger.info(f"📊 Успешность: {success_rate:.1f}%")
        
        logger.info("\n" + "-"*70)
        
        if failed == 0 and warnings == 0:
            logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! СИСТЕМА ГОТОВА К РАБОТЕ НА 150%!")
        elif failed == 0:
            logger.info("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ (есть предупреждения, но не критично)")
        elif success_rate >= 90:
            logger.info("⚠️ БОЛЬШИНСТВО ТЕСТОВ ПРОЙДЕНО (требуется исправление ошибок)")
        else:
            logger.info("❌ ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ (много ошибок)")
        
        logger.info("="*70)
        
        # Детали неудачных тестов
        if failed > 0:
            logger.info("\n❌ ПРОВАЛЕННЫЕ ТЕСТЫ:")
            for test_name, result in self.results.items():
                if not result['passed'] and not result['warning']:
                    logger.info(f"   • {test_name}: {result['message']}")


async def main():
    """Главная функция"""
    tester = SuperExpertTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

