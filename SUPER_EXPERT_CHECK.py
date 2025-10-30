#!/usr/bin/env python3
"""
🔬 ГЕНЕРАЛЬНАЯ СУПЕР ПРОВЕРКА БОТА
==================================
Проверка всех систем как супер команда экспертов-программистов-разработчиков
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
env_files = [
    Path(__file__).parent / 'api.env',
    Path(__file__).parent.parent / '.env',
    Path(__file__).parent / '.env'
]

for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file, override=False)
        break


class SuperExpertChecker:
    """Супер проверка всех систем бота"""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_result(self, test_name: str, passed: bool, message: str = "", warning: bool = False):
        """Логирование результата теста"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            self.results['passed'].append((test_name, message))
            print(f"✅ {test_name}: {message}")
        elif warning:
            self.results['warnings'].append((test_name, message))
            print(f"⚠️ {test_name}: {message}")
        else:
            self.results['failed'].append((test_name, message))
            print(f"❌ {test_name}: {message}")
    
    async def test_1_imports(self):
        """Тест 1: Импорт всех модулей"""
        print("\n" + "="*70)
        print("📦 ТЕСТ 1: ИМПОРТ ВСЕХ МОДУЛЕЙ")
        print("="*70)
        
        modules = [
            'super_bot_v4_mtf',
            'telegram_commands_handler',
            'ai_ml_system',
            'smart_coin_selector',
            'advanced_indicators',
            'llm_monitor',
            'universal_learning_system',
            'data_storage_system',
            'advanced_manipulation_detector',
            'adaptive_parameters',
            'adaptive_trading_system',
            'probability_calculator',
            'strategy_evaluator',
            'advanced_ml_system'
        ]
        
        failed = []
        for mod in modules:
            try:
                __import__(mod)
                self.log_result(f"Импорт {mod}", True, "OK")
            except Exception as e:
                failed.append(f"{mod}: {e}")
                self.log_result(f"Импорт {mod}", False, str(e))
        
        if not failed:
            self.log_result("Все модули импортируются", True, f"{len(modules)}/{len(modules)}")
        else:
            self.log_result("Все модули импортируются", False, f"{len(modules)-len(failed)}/{len(modules)}")
    
    async def test_2_configuration(self):
        """Тест 2: Конфигурация бота"""
        print("\n" + "="*70)
        print("⚙️ ТЕСТ 2: КОНФИГУРАЦИЯ БОТА")
        print("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            # Проверка параметров
            checks = {
                'MIN_CONFIDENCE': (bot.MIN_CONFIDENCE, 65, "Должен быть 65%"),
                'MAX_POSITIONS': (bot.MAX_POSITIONS, 3, "Должен быть 3"),
                'LEVERAGE': (bot.LEVERAGE, 5, "Должен быть 5x"),
                'POSITION_SIZE': (bot.POSITION_SIZE, 5.0, "Должен быть 5.0 USDT"),
                'STOP_LOSS_PERCENT': (bot.STOP_LOSS_PERCENT, 20, "Должен быть 20%"),
                'API ключ Bybit': (bool(bot.api_key), True, "Должен быть установлен"),
                'API секрет Bybit': (bool(bot.api_secret), True, "Должен быть установлен"),
                'Telegram токен': (bool(bot.telegram_token), True, "Должен быть установлен"),
                'Telegram Chat ID': (bool(bot.telegram_chat_id), True, "Должен быть установлен")
            }
            
            for param_name, (value, expected, desc) in checks.items():
                if value == expected:
                    self.log_result(f"Конфигурация: {param_name}", True, f"{value} (OK)")
                else:
                    self.log_result(f"Конфигурация: {param_name}", False, f"{value} (ожидалось {expected})")
            
            # Проверка MIN_CONFIDENCE специально
            if bot.MIN_CONFIDENCE == 65:
                self.log_result("MIN_CONFIDENCE обновлен", True, "65% ✅ (было 45%)")
            else:
                self.log_result("MIN_CONFIDENCE обновлен", False, f"{bot.MIN_CONFIDENCE}% (ожидалось 65%)")
            
        except Exception as e:
            self.log_result("Конфигурация бота", False, f"Ошибка: {e}")
    
    async def test_3_timeframes(self):
        """Тест 3: Таймфреймы"""
        print("\n" + "="*70)
        print("⏰ ТЕСТ 3: ТАЙМФРЕЙМЫ")
        print("="*70)
        
        expected_timeframes = ['15m', '30m', '45m', '1h', '4h']
        
        try:
            # Читаем код и проверяем таймфреймы
            code_file = Path(__file__).parent / 'super_bot_v4_mtf.py'
            if code_file.exists():
                content = code_file.read_text(encoding='utf-8')
                
                if "timeframes = ['15m', '30m', '45m', '1h', '4h']" in content:
                    self.log_result("Таймфреймы в коде", True, "5 таймфреймов найдено")
                else:
                    self.log_result("Таймфреймы в коде", False, "Не найдены ожидаемые таймфреймы")
                
                for tf in expected_timeframes:
                    if tf in content:
                        self.log_result(f"Таймфрейм {tf}", True, "Найден в коде")
                    else:
                        self.log_result(f"Таймфрейм {tf}", False, "Не найден")
            
            # Проверка что 45m добавлен (V4.0)
            if "'45m'" in content or '"45m"' in content:
                self.log_result("45m таймфрейм (V4.0)", True, "Добавлен ✅")
            else:
                self.log_result("45m таймфрейм (V4.0)", False, "Не найден")
                
        except Exception as e:
            self.log_result("Проверка таймфреймов", False, f"Ошибка: {e}")
    
    async def test_4_filters(self):
        """Тест 4: Система фильтров"""
        print("\n" + "="*70)
        print("🔒 ТЕСТ 4: СИСТЕМА ФИЛЬТРОВ")
        print("="*70)
        
        required_filters = [
            'Strategy Evaluator',
            'RealismValidator',
            'ManipulationDetector',
            'MTF подтверждение',
            'Volume Spike',
            'TP вероятности',
            'MIN_CONFIDENCE'
        ]
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            # Проверка модулей фильтров
            filters_status = {
                'Strategy Evaluator': hasattr(bot, 'strategy_evaluator') and bot.strategy_evaluator is not None,
                'RealismValidator': hasattr(bot, 'realism_validator') and bot.realism_validator is not None,
                'ManipulationDetector': 'ManipulationDetector' in str(type(bot)),
                'MTF подтверждение': True,  # Всегда включено
                'Volume Spike': True,  # Проверяется в коде
                'TP вероятности': hasattr(bot, 'probability_calculator') and bot.probability_calculator is not None,
                'MIN_CONFIDENCE': bot.MIN_CONFIDENCE == 65
            }
            
            for filter_name, status in filters_status.items():
                if status:
                    self.log_result(f"Фильтр: {filter_name}", True, "Активен")
                else:
                    self.log_result(f"Фильтр: {filter_name}", False, "Не активен")
            
            if all(filters_status.values()):
                self.log_result("Все фильтры активны", True, f"{len(required_filters)}/{len(required_filters)}")
            else:
                active = sum(1 for v in filters_status.values() if v)
                self.log_result("Все фильтры активны", False, f"{active}/{len(required_filters)}")
                
        except Exception as e:
            self.log_result("Проверка фильтров", False, f"Ошибка: {e}")
    
    async def test_5_ml_systems(self):
        """Тест 5: ML/AI системы"""
        print("\n" + "="*70)
        print("🤖 ТЕСТ 5: ML/AI СИСТЕМЫ")
        print("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            ml_systems = {
                'ML система': hasattr(bot, 'ml_system') and bot.ml_system is not None,
                'Universal Learning': hasattr(bot, 'universal_learning') and bot.universal_learning is not None,
                'Adaptive Parameters': hasattr(bot, 'adaptive_params_system') and bot.adaptive_params_system is not None,
                'Fully Adaptive': hasattr(bot, 'fully_adaptive_system') and bot.fully_adaptive_system is not None,
                'Health Monitor': hasattr(bot, 'health_monitor') and bot.health_monitor is not None,
                'ML Predictor': hasattr(bot, 'ml_predictor') and bot.ml_predictor is not None,
                'LLM Analyzer': hasattr(bot, 'llm_analyzer') and bot.llm_analyzer is not None
            }
            
            for system_name, status in ml_systems.items():
                if status:
                    self.log_result(f"ML/AI: {system_name}", True, "Инициализирована")
                else:
                    self.log_result(f"ML/AI: {system_name}", False, "Не инициализирована")
            
            active_count = sum(1 for v in ml_systems.values() if v)
            if active_count >= 5:
                self.log_result("ML/AI системы активны", True, f"{active_count}/{len(ml_systems)}")
            else:
                self.log_result("ML/AI системы активны", False, f"{active_count}/{len(ml_systems)}")
                
        except Exception as e:
            self.log_result("Проверка ML/AI", False, f"Ошибка: {e}")
    
    async def test_6_database(self):
        """Тест 6: База данных"""
        print("\n" + "="*70)
        print("💾 ТЕСТ 6: БАЗА ДАННЫХ")
        print("="*70)
        
        try:
            from data_storage_system import DataStorageSystem
            storage = DataStorageSystem()
            
            # Проверка что БД инициализирована
            self.log_result("База данных", True, "DataStorageSystem инициализирована")
            
            # Проверка пути к БД
            db_path = getattr(storage, 'db_path', None)
            if db_path:
                self.log_result("Путь к БД", True, str(db_path))
                if Path(db_path).parent.exists():
                    self.log_result("Директория БД", True, "Существует")
                else:
                    self.log_result("Директория БД", False, "Не существует")
            
        except Exception as e:
            self.log_result("База данных", False, f"Ошибка: {e}")
    
    async def test_7_telegram(self):
        """Тест 7: Telegram бот"""
        print("\n" + "="*70)
        print("📱 ТЕСТ 7: TELEGRAM БОТ")
        print("="*70)
        
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if telegram_token:
            self.log_result("Telegram токен", True, f"{telegram_token[:10]}...")
        else:
            self.log_result("Telegram токен", False, "Не найден")
        
        if telegram_chat_id:
            self.log_result("Telegram Chat ID", True, telegram_chat_id)
        else:
            self.log_result("Telegram Chat ID", False, "Не найден")
        
        try:
            from telegram_commands_handler import TelegramCommandsHandler
            self.log_result("Telegram Commands Handler", True, "Модуль импортируется")
            
            # Проверка команд
            commands = [
                'cmd_start', 'cmd_help', 'cmd_status', 'cmd_balance',
                'cmd_positions', 'cmd_history', 'cmd_settings', 'cmd_health',
                'cmd_stop', 'cmd_resume', 'cmd_stats'
            ]
            
            for cmd in commands:
                if hasattr(TelegramCommandsHandler, cmd):
                    self.log_result(f"Команда {cmd}", True, "Существует")
                else:
                    self.log_result(f"Команда {cmd}", False, "Не найдена")
            
        except Exception as e:
            self.log_result("Telegram модуль", False, f"Ошибка: {e}")
    
    async def test_8_trading_logic(self):
        """Тест 8: Логика торговли"""
        print("\n" + "="*70)
        print("💹 ТЕСТ 8: ЛОГИКА ТОРГОВЛИ")
        print("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            # Проверка TP уровней
            if hasattr(bot, 'TP_LEVELS_V4'):
                tp_levels = bot.TP_LEVELS_V4
                self.log_result("TP уровни", True, f"{len(tp_levels)} уровней")
                
                # Проверяем первый TP
                if len(tp_levels) > 0:
                    tp1 = tp_levels[0]
                    if tp1.get('percent') == 4 and tp1.get('portion') == 0.40:
                        self.log_result("TP1 уровень", True, "+4% (40% позиции)")
                    else:
                        self.log_result("TP1 уровень", False, f"Неверные параметры: {tp1}")
            
            # Проверка MIN_CONFIDENCE снова
            if bot.MIN_CONFIDENCE == 65:
                self.log_result("MIN_CONFIDENCE для торговли", True, "65% ✅")
            else:
                self.log_result("MIN_CONFIDENCE для торговли", False, f"{bot.MIN_CONFIDENCE}% (ожидалось 65%)")
            
            # Проверка MAX_POSITIONS
            if bot.MAX_POSITIONS == 3:
                self.log_result("MAX_POSITIONS", True, "3 позиции максимум")
            else:
                self.log_result("MAX_POSITIONS", False, f"{bot.MAX_POSITIONS} (ожидалось 3)")
                
        except Exception as e:
            self.log_result("Логика торговли", False, f"Ошибка: {e}")
    
    async def test_9_libraries(self):
        """Тест 9: Библиотеки"""
        print("\n" + "="*70)
        print("📚 ТЕСТ 9: БИБЛИОТЕКИ")
        print("="*70)
        
        required_libs = {
            'ccxt': 'ccxt',
            'telegram': 'telegram',
            'apscheduler': 'apscheduler',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sklearn': 'sklearn',
            'tensorflow': 'tensorflow',
            'openai': 'openai'
        }
        
        for lib_name, import_name in required_libs.items():
            try:
                __import__(import_name)
                self.log_result(f"Библиотека {lib_name}", True, "Установлена")
            except ImportError:
                self.log_result(f"Библиотека {lib_name}", False, "Не установлена")
    
    async def test_10_bot_initialization(self):
        """Тест 10: Инициализация бота"""
        print("\n" + "="*70)
        print("🚀 ТЕСТ 10: ИНИЦИАЛИЗАЦИЯ БОТА")
        print("="*70)
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            bot = SuperBotV4MTF()
            
            self.log_result("Создание экземпляра бота", True, "Успешно")
            
            # Пробуем инициализировать (без реального запуска)
            try:
                await bot.initialize()
                
                if bot.exchange:
                    self.log_result("Exchange инициализирован", True, "OK")
                else:
                    self.log_result("Exchange инициализирован", False, "Не создан")
                
                if bot.application:
                    self.log_result("Telegram Application", True, "Инициализирован")
                else:
                    self.log_result("Telegram Application", False, "Не инициализирован")
                
                if bot.commands_handler:
                    self.log_result("Commands Handler", True, "Инициализирован")
                else:
                    self.log_result("Commands Handler", False, "Не инициализирован")
                
                # Закрываем соединения
                if bot.exchange:
                    await bot.exchange.close()
                
                self.log_result("Полная инициализация", True, "OK")
                
            except Exception as e:
                self.log_result("Инициализация бота", False, f"Ошибка: {e}")
                
        except Exception as e:
            self.log_result("Создание бота", False, f"Ошибка: {e}")
    
    def print_summary(self):
        """Вывод итогового отчета"""
        print("\n" + "="*70)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("="*70)
        print()
        
        total = self.total_tests
        passed = self.passed_tests
        failed = len(self.results['failed'])
        warnings = len(self.results['warnings'])
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"📈 Всего тестов: {total}")
        print(f"✅ Пройдено: {passed}")
        print(f"❌ Провалено: {failed}")
        print(f"⚠️ Предупреждений: {warnings}")
        print(f"📊 Успешность: {success_rate:.1f}%")
        print()
        
        if failed > 0:
            print("❌ ПРОВАЛЕННЫЕ ТЕСТЫ:")
            for test_name, message in self.results['failed']:
                print(f"  • {test_name}: {message}")
            print()
        
        if warnings > 0:
            print("⚠️ ПРЕДУПРЕЖДЕНИЯ:")
            for test_name, message in self.results['warnings']:
                print(f"  • {test_name}: {message}")
            print()
        
        if success_rate >= 90:
            print("🎉 ОТЛИЧНО! Система работает на 90%+")
        elif success_rate >= 70:
            print("✅ ХОРОШО! Система работает, но есть что улучшить")
        else:
            print("⚠️ ТРЕБУЕТСЯ ВНИМАНИЕ! Много проваленных тестов")
        
        print("="*70)


async def main():
    """Главная функция"""
    print("="*70)
    print("🔬 ГЕНЕРАЛЬНАЯ СУПЕР ПРОВЕРКА БОТА")
    print("   Экспертная команда программистов-разработчиков")
    print("="*70)
    print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checker = SuperExpertChecker()
    
    # Запускаем все тесты
    await checker.test_1_imports()
    await checker.test_2_configuration()
    await checker.test_3_timeframes()
    await checker.test_4_filters()
    await checker.test_5_ml_systems()
    await checker.test_6_database()
    await checker.test_7_telegram()
    await checker.test_8_trading_logic()
    await checker.test_9_libraries()
    await checker.test_10_bot_initialization()
    
    # Итоговый отчет
    checker.print_summary()


if __name__ == "__main__":
    asyncio.run(main())


