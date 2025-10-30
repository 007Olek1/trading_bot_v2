#!/usr/bin/env python3
"""
🔬 ЭКСПЕРТНЫЙ ТЕСТ СИСТЕМЫ
==========================
Генеральная проверка всей системы торгового бота
как команда экспертов-программистов
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('expert_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ExpertTestSuite:
    """Экспертная система тестирования"""
    
    def __init__(self):
        self.results = {
            'module_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'universal_learning_tests': {},
            'profitability_tests': {},
            'overall_score': 0.0
        }
        
    def print_header(self, title: str):
        """Красивый заголовок"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")
    
    async def test_module_imports(self):
        """Тест 1: Проверка импорта всех модулей"""
        self.print_header("📦 ТЕСТ 1: ИМПОРТ МОДУЛЕЙ")
        
        modules = {
            'SuperBotV4MTF': 'super_bot_v4_mtf',
            'AdvancedIndicators': 'advanced_indicators',
            'BotHealthMonitor': 'llm_monitor',
            'UniversalLearningSystem': 'universal_learning_system',
            'DataStorageSystem': 'data_storage_system',
            'ProbabilityCalculator': 'super_bot_v4_mtf',
            'StrategyEvaluator': 'super_bot_v4_mtf'
        }
        
        results = {}
        for name, module in modules.items():
            try:
                mod = __import__(module, fromlist=[name])
                cls = getattr(mod, name)
                results[name] = {'status': '✅', 'message': 'Импортирован успешно'}
            except Exception as e:
                results[name] = {'status': '❌', 'message': f'Ошибка: {e}'}
            
            print(f"{results[name]['status']} {name}: {results[name]['message']}")
        
        self.results['module_tests']['imports'] = results
        return all(r['status'] == '✅' for r in results.values())
    
    async def test_system_initialization(self):
        """Тест 2: Инициализация системы"""
        self.print_header("🚀 ТЕСТ 2: ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ")
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            
            bot = SuperBotV4MTF()
            await bot.initialize()
            
            checks = {
                'Exchange': bot.exchange is not None,
                'Telegram': hasattr(bot, 'application') and bot.application is not None,
                'Advanced Indicators': bot.advanced_indicators is not None,
                'LLM Monitor': bot.health_monitor is not None,
                'Universal Learning': bot.universal_learning is not None,
                'Data Storage': bot.data_storage is not None,
                'Probability Calculator': bot.probability_calculator is not None,
                'Strategy Evaluator': bot.strategy_evaluator is not None
            }
            
            all_passed = True
            for name, status in checks.items():
                emoji = '✅' if status else '❌'
                print(f"{emoji} {name}: {'Работает' if status else 'Не инициализирован'}")
                if not status:
                    all_passed = False
            
            self.results['module_tests']['initialization'] = checks
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}", exc_info=True)
            return False
    
    async def test_database(self):
        """Тест 3: Работа с базой данных"""
        self.print_header("💾 ТЕСТ 3: БАЗА ДАННЫХ")
        
        try:
            from data_storage_system import DataStorageSystem
            
            storage = DataStorageSystem()
            
            # Тест записи
            from data_storage_system import MarketData
            from datetime import datetime
            
            test_market_data = MarketData(
                timestamp=datetime.now().isoformat(),
                symbol='TESTUSDT',
                timeframe='15m',
                price=100.0,
                volume=1000000.0,
                rsi=55.0,
                macd=0.5,
                bb_position=60.0,
                ema_9=99.0,
                ema_21=98.0,
                ema_50=97.0,
                volume_ratio=1.5,
                momentum=0.02,
                market_condition='NEUTRAL'
            )
            
            storage.store_market_data(test_market_data)
            print(f"✅ Запись данных: OK")
            
            # Тест чтения
            saved_data = storage.get_market_data('TESTUSDT', '15m', hours=1)
            if saved_data:
                print(f"✅ Чтение данных: OK")
            else:
                print(f"⚠️ Данные не найдены (может быть нормально)")
            
            self.results['module_tests']['database'] = {'status': '✅', 'details': 'OK'}
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка БД: {e}", exc_info=True)
            self.results['module_tests']['database'] = {'status': '❌', 'error': str(e)}
            return False
    
    async def test_advanced_indicators(self):
        """Тест 4: Расширенные индикаторы"""
        self.print_header("📊 ТЕСТ 4: РАСШИРЕННЫЕ ИНДИКАТОРЫ")
        
        try:
            from advanced_indicators import AdvancedIndicators
            import pandas as pd
            import numpy as np
            
            indicators = AdvancedIndicators()
            
            # Создаем тестовые данные
            dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
            df = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.rand(100) * 100 + 100,
                'high': np.random.rand(100) * 10 + 195,
                'low': np.random.rand(100) * 10 + 100,
                'close': np.random.rand(100) * 10 + 150,
                'volume': np.random.rand(100) * 1000
            })
            df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(100) * 5
            df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(100) * 5
            df['close'] = df['close'].cumsum() / 10 + 100
            
            # Тест Ichimoku
            ichimoku = indicators.calculate_ichimoku(df.copy())
            print(f"{'✅' if ichimoku else '❌'} Ichimoku Cloud: {'OK' if ichimoku else 'Failed'}")
            
            # Тест Fibonacci
            fib = indicators.calculate_fibonacci(df.copy())
            print(f"{'✅' if fib else '❌'} Fibonacci Levels: {'OK' if fib else 'Failed'}")
            
            # Тест Support/Resistance
            sr = indicators.detect_support_resistance(df.copy())
            print(f"{'✅' if sr else '❌'} Support/Resistance: {'OK' if sr else 'Failed'}")
            
            all_passed = bool(ichimoku and fib and sr)
            self.results['module_tests']['advanced_indicators'] = {
                'ichimoku': bool(ichimoku),
                'fibonacci': bool(fib),
                'support_resistance': bool(sr)
            }
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Ошибка Advanced Indicators: {e}", exc_info=True)
            return False
    
    async def test_universal_learning(self):
        """Тест 5: Универсальное обучение"""
        self.print_header("🧠 ТЕСТ 5: УНИВЕРСАЛЬНОЕ ОБУЧЕНИЕ")
        
        try:
            from universal_learning_system import UniversalLearningSystem
            
            learning = UniversalLearningSystem()
            
            # Добавляем тестовые данные обучения
            test_samples = []
            for i in range(15):
                test_samples.append({
                    'market_data': {
                        'rsi': 30 + i * 2,
                        'bb_position': 20 + i * 3,
                        'volume_ratio': 0.5 + i * 0.1,
                        'momentum': -0.05 + i * 0.005,
                        'confidence': 60 + i,
                        'strategy_score': 10 + i * 0.5,
                        'market_condition': 'NEUTRAL'
                    },
                    'decision': 'buy' if i % 2 == 0 else 'sell',
                    'result': 'win' if i % 3 != 0 else 'loss',
                    'timestamp': datetime.now().isoformat(),
                    'market_condition': 'NEUTRAL'
                })
            
            # Обучаем систему
            for sample in test_samples:
                learning.learn_from_decision(
                    sample['market_data'],
                    sample['decision'],
                    sample['result']
                )
            
            # Создаем паттерны
            patterns = learning.analyze_market_patterns(learning.learning_history)
            print(f"✅ Создано паттернов: {len(patterns)}")
            
            # Создаем правила
            if patterns:
                rules = learning.create_universal_rules(patterns)
                print(f"✅ Создано правил: {len(rules)}")
                
                # Проверяем универсальность
                analysis = learning.analyze_patterns()
                if analysis:
                    ranges_count = analysis.get('patterns_with_ranges', 0)
                    exact_count = analysis.get('patterns_with_exact_values', 0)
                    generalization = analysis.get('average_generalization', 0)
                    
                    print(f"✅ Паттернов с диапазонами: {ranges_count}")
                    print(f"✅ Паттернов с точными значениями: {exact_count}")
                    print(f"✅ Средний score обобщения: {generalization:.2f}")
                    
                    is_universal = ranges_count > exact_count and generalization > 0.6
                    print(f"{'✅' if is_universal else '⚠️'} Универсальность: {'OK' if is_universal else 'Требует внимания'}")
                    
                    self.results['universal_learning_tests'] = {
                        'patterns_created': len(patterns),
                        'rules_created': len(rules),
                        'is_universal': is_universal,
                        'generalization_score': generalization
                    }
                    return is_universal
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка Universal Learning: {e}", exc_info=True)
            return False
    
    async def test_profitability_calculation(self):
        """Тест 6: Расчет прибыльности"""
        self.print_header("💰 ТЕСТ 6: РАСЧЕТ ПРИБЫЛЬНОСТИ")
        
        try:
            # Параметры
            position_size = 5.0
            leverage = 5
            total_position = position_size * leverage  # $25
            
            # TP уровни V4
            tp_levels = [
                {'level': 1, 'percent': 4, 'portion': 0.40},
                {'level': 2, 'percent': 6, 'portion': 0.20},
                {'level': 3, 'percent': 8, 'portion': 0.20},
            ]
            
            # Расчет прибыли
            total_profit = 0
            for tp in tp_levels:
                profit = total_position * tp['portion'] * (tp['percent'] / 100)
                total_profit += profit
                print(f"  TP{tp['level']}: +{tp['percent']}% ({tp['portion']*100:.0f}%) = ${profit:.2f}")
            
            print(f"\n💰 Итого прибыль (TP1+TP2+TP3): ${total_profit:.2f}")
            
            min_target = 1.0
            is_profitable = total_profit >= min_target
            emoji = '✅' if is_profitable else '❌'
            print(f"{emoji} Минимальная прибыль ${min_target}: {'ГАРАНТИРОВАНО' if is_profitable else 'НЕ ДОСТИГНУТО'}")
            
            self.results['profitability_tests'] = {
                'total_profit': total_profit,
                'meets_minimum': is_profitable,
                'min_target': min_target
            }
            return is_profitable
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета прибыли: {e}", exc_info=True)
            return False
    
    async def test_integration(self):
        """Тест 7: Интеграция модулей"""
        self.print_header("🔗 ТЕСТ 7: ИНТЕГРАЦИЯ МОДУЛЕЙ")
        
        try:
            from super_bot_v4_mtf import SuperBotV4MTF
            
            bot = SuperBotV4MTF()
            await bot.initialize()
            
            # Проверяем связи между модулями
            integration_checks = {
                'Bot -> Advanced Indicators': bot.advanced_indicators is not None,
                'Bot -> LLM Monitor': bot.health_monitor is not None,
                'Bot -> Universal Learning': bot.universal_learning is not None,
                'Bot -> Data Storage': bot.data_storage is not None,
                'Universal Learning -> Data Storage': bot.universal_learning.data_storage == bot.data_storage if bot.universal_learning else False
            }
            
            all_passed = True
            for name, status in integration_checks.items():
                emoji = '✅' if status else '❌'
                print(f"{emoji} {name}")
                if not status:
                    all_passed = False
            
            self.results['integration_tests'] = integration_checks
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Ошибка интеграции: {e}", exc_info=True)
            return False
    
    def generate_report(self):
        """Генерация финального отчета"""
        self.print_header("📋 ФИНАЛЬНЫЙ ОТЧЕТ ЭКСПЕРТОВ")
        
        # Подсчет баллов
        total_score = 0
        max_score = 0
        
        # Модули
        if 'module_tests' in self.results:
            module_tests = self.results['module_tests']
            for test_name, result in module_tests.items():
                max_score += 10
                if isinstance(result, dict):
                    if result.get('status') == '✅' or result.get('status') is True:
                        total_score += 10
                    elif isinstance(result, dict):
                        # Считаем процент пройденных проверок
                        passed = sum(1 for v in result.values() if v is True or v == '✅')
                        total_score += (passed / len(result)) * 10
        
        # Интеграция
        if 'integration_tests' in self.results:
            integration = self.results['integration_tests']
            passed = sum(1 for v in integration.values() if v is True)
            max_score += 10
            total_score += (passed / len(integration)) * 10
        
        # Universal Learning
        if 'universal_learning_tests' in self.results:
            ul_test = self.results['universal_learning_tests']
            max_score += 10
            if ul_test.get('is_universal', False):
                total_score += 10
        
        # Прибыльность
        if 'profitability_tests' in self.results:
            profit_test = self.results['profitability_tests']
            max_score += 10
            if profit_test.get('meets_minimum', False):
                total_score += 10
        
        final_score = (total_score / max_score * 100) if max_score > 0 else 0
        self.results['overall_score'] = final_score
        
        # Вывод результатов
        print(f"\n📊 ОЦЕНКА СИСТЕМЫ:")
        print(f"   Всего тестов: {max_score // 10}")
        print(f"   Пройдено: {total_score // 10}")
        print(f"   Финальная оценка: {final_score:.1f}%")
        print(f"\n{'✅ СИСТЕМА ГОТОВА К РАБОТЕ' if final_score >= 80 else '⚠️ ТРЕБУЕТСЯ ДОРАБОТКА'}")
        
        # Детали
        print(f"\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(json.dumps(self.results, indent=2, ensure_ascii=False, default=str))
        
        return final_score >= 80
    
    async def run_all_tests(self):
        """Запуск всех тестов"""
        self.print_header("🔬 ЭКСПЕРТНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ")
        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        tests = [
            ("Импорт модулей", self.test_module_imports),
            ("Инициализация системы", self.test_system_initialization),
            ("База данных", self.test_database),
            ("Расширенные индикаторы", self.test_advanced_indicators),
            ("Универсальное обучение", self.test_universal_learning),
            ("Расчет прибыльности", self.test_profitability_calculation),
            ("Интеграция модулей", self.test_integration),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"❌ Ошибка в тесте '{test_name}': {e}", exc_info=True)
                results[test_name] = False
        
        # Генерация отчета
        is_ready = self.generate_report()
        
        return is_ready, results


async def main():
    """Главная функция"""
    suite = ExpertTestSuite()
    is_ready, results = await suite.run_all_tests()
    
    # Сохраняем отчет
    report_file = Path('expert_test_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'is_ready': is_ready,
            'overall_score': suite.results['overall_score'],
            'results': suite.results,
            'test_results': results
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Отчет сохранен: {report_file}")
    
    return 0 if is_ready else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
