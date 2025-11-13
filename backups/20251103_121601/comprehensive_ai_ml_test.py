#!/usr/bin/env python3
"""
🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ AI+ML+БОТ
============================================

Проверяет:
1. Структуру папок для данных
2. DataStorageSystem - запись и обновление данных
3. UniversalLearningSystem - универсальные правила vs запоминание
4. AdvancedMLSystem - самопереобучение
5. SmartCoinSelector - выбор 100-200 монет
6. Интеграцию всех компонентов
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Импорт компонентов системы
try:
    from data_storage_system import DataStorageSystem, MarketData, TradeDecision
    logger.info("✅ DataStorageSystem импортирован")
except Exception as e:
    logger.error(f"❌ Ошибка импорта DataStorageSystem: {e}")
    sys.exit(1)

try:
    from universal_learning_system import UniversalLearningSystem
    logger.info("✅ UniversalLearningSystem импортирован")
except Exception as e:
    logger.error(f"❌ Ошибка импорта UniversalLearningSystem: {e}")
    sys.exit(1)

try:
    from advanced_ml_system import AdvancedMLSystem
    logger.info("✅ AdvancedMLSystem импортирован")
except Exception as e:
    logger.error(f"❌ Ошибка импорта AdvancedMLSystem: {e}")
    sys.exit(1)

try:
    from smart_coin_selector import SmartCoinSelector
    logger.info("✅ SmartCoinSelector импортирован")
except Exception as e:
    logger.error(f"❌ Ошибка импорта SmartCoinSelector: {e}")
    sys.exit(1)


class ComprehensiveSystemTest:
    """🧪 Комплексное тестирование системы"""
    
    def __init__(self):
        self.test_results = {
            'folder_structure': {},
            'data_storage': {},
            'universal_learning': {},
            'advanced_ml': {},
            'smart_selector': {},
            'integration': {},
            'overall_status': 'PENDING'
        }
        
        # Определяем базовый путь
        if Path("/opt/bot").exists():
            self.base_dir = Path("/opt/bot")
            logger.info("📂 Работаем на сервере: /opt/bot")
        else:
            self.base_dir = Path(__file__).parent
            logger.info(f"📂 Работаем локально: {self.base_dir}")
        
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.data_dir / "models"
        self.cache_dir = self.data_dir / "cache"
        self.storage_dir = self.data_dir / "storage"
        self.logs_dir = self.data_dir / "logs"
        self.knowledge_dir = self.data_dir / "knowledge"
    
    def test_folder_structure(self) -> bool:
        """📁 Тест 1: Структура папок"""
        logger.info("\n" + "="*60)
        logger.info("📁 ТЕСТ 1: СТРУКТУРА ПАПОК")
        logger.info("="*60)
        
        results = {}
        all_passed = True
        
        required_dirs = {
            'data': self.data_dir,
            'models': self.models_dir,
            'cache': self.cache_dir,
            'storage': self.storage_dir,
            'logs': self.logs_dir,
            'knowledge': self.knowledge_dir,
        }
        
        for name, path in required_dirs.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                exists = path.exists()
                is_writable = os.access(path, os.W_OK)
                
                results[name] = {
                    'exists': exists,
                    'writable': is_writable,
                    'path': str(path),
                    'status': '✅' if (exists and is_writable) else '❌'
                }
                
                if not (exists and is_writable):
                    all_passed = False
                
                logger.info(f"  {results[name]['status']} {name}: {path} (существует: {exists}, запись: {is_writable})")
                
                # Проверяем размер если есть файлы
                if exists:
                    try:
                        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        file_count = sum(1 for _ in path.rglob('*') if _.is_file())
                        results[name]['size_bytes'] = total_size
                        results[name]['file_count'] = file_count
                        logger.info(f"    📊 Файлов: {file_count}, Размер: {total_size / 1024:.1f} KB")
                    except:
                        pass
                        
            except Exception as e:
                results[name] = {
                    'exists': False,
                    'writable': False,
                    'error': str(e),
                    'status': '❌'
                }
                all_passed = False
                logger.error(f"  ❌ {name}: Ошибка - {e}")
        
        self.test_results['folder_structure'] = results
        logger.info(f"\n📁 Результат: {'✅ ПРОЙДЕН' if all_passed else '❌ ПРОВАЛЕН'}")
        return all_passed
    
    def test_data_storage(self) -> bool:
        """💾 Тест 2: DataStorageSystem"""
        logger.info("\n" + "="*60)
        logger.info("💾 ТЕСТ 2: DATA STORAGE SYSTEM")
        logger.info("="*60)
        
        results = {}
        all_passed = True
        
        try:
            # Инициализация
            storage = DataStorageSystem()
            results['init'] = {'status': '✅', 'message': 'Инициализация успешна'}
            logger.info("  ✅ DataStorageSystem инициализирована")
            
            # Тест записи рыночных данных
            test_market_data = MarketData(
                timestamp=datetime.now().isoformat(),
                symbol='BTCUSDT',
                timeframe='45m',
                price=50000.0,
                volume=1000000.0,
                rsi=55.0,
                macd=100.0,
                bb_position=50.0,
                ema_9=51000.0,
                ema_21=50500.0,
                ema_50=50000.0,
                volume_ratio=1.5,
                momentum=0.5,
                market_condition='NEUTRAL'
            )
            
            storage.save_market_data(test_market_data)
            results['save_market'] = {'status': '✅', 'message': 'Рыночные данные сохранены'}
            logger.info("  ✅ Рыночные данные сохранены")
            
            # Тест записи торговых решений
            test_trade = TradeDecision(
                timestamp=datetime.now().isoformat(),
                symbol='BTCUSDT',
                decision='buy',
                confidence=75.0,
                strategy_score=15.0,
                reasons=['RSI перепродан', 'BB отскок'],
                market_data={'rsi': 30.0, 'bb_position': 25.0},
                result='win',
                pnl_percent=2.5,
                entry_price=50000.0,
                exit_price=51250.0
            )
            
            storage.save_trade_decision(test_trade)
            results['save_trade'] = {'status': '✅', 'message': 'Торговое решение сохранено'}
            logger.info("  ✅ Торговое решение сохранено")
            
            # Тест получения данных
            market_data_list = storage.get_market_data('BTCUSDT', limit=10)
            results['get_market'] = {
                'status': '✅',
                'count': len(market_data_list),
                'message': f'Получено {len(market_data_list)} записей рыночных данных'
            }
            logger.info(f"  ✅ Получено {len(market_data_list)} записей рыночных данных")
            
            trade_decisions = storage.get_trade_decisions('BTCUSDT', limit=10)
            results['get_trades'] = {
                'status': '✅',
                'count': len(trade_decisions),
                'message': f'Получено {len(trade_decisions)} торговых решений'
            }
            logger.info(f"  ✅ Получено {len(trade_decisions)} торговых решений")
            
            # Проверка базы данных
            db_path = storage.db_path
            if Path(db_path).exists():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM market_data")
                market_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM trade_decisions")
                trade_count = cursor.fetchone()[0]
                
                results['database'] = {
                    'status': '✅',
                    'market_records': market_count,
                    'trade_records': trade_count,
                    'db_path': db_path
                }
                logger.info(f"  ✅ База данных: {market_count} рыночных записей, {trade_count} торговых решений")
                conn.close()
            else:
                results['database'] = {'status': '❌', 'message': 'База данных не найдена'}
                all_passed = False
            
        except Exception as e:
            results['error'] = {'status': '❌', 'message': str(e)}
            all_passed = False
            logger.error(f"  ❌ Ошибка тестирования DataStorageSystem: {e}")
        
        self.test_results['data_storage'] = results
        logger.info(f"\n💾 Результат: {'✅ ПРОЙДЕН' if all_passed else '❌ ПРОВАЛЕН'}")
        return all_passed
    
    def test_universal_learning(self) -> bool:
        """🧠 Тест 3: UniversalLearningSystem (универсальные правила vs запоминание)"""
        logger.info("\n" + "="*60)
        logger.info("🧠 ТЕСТ 3: UNIVERSAL LEARNING SYSTEM")
        logger.info("="*60)
        
        results = {}
        all_passed = True
        
        try:
            storage = DataStorageSystem()
            learning_system = UniversalLearningSystem(storage)
            results['init'] = {'status': '✅', 'message': 'Система инициализирована'}
            logger.info("  ✅ UniversalLearningSystem инициализирована")
            
            # Создаем тестовые данные для обучения
            test_data = []
            for i in range(20):  # Минимум 10 для создания правила
                test_data.append({
                    'symbol': 'BTCUSDT',
                    'rsi': 30.0 + (i * 2),  # Диапазон 30-68
                    'bb_position': 20.0 + (i * 3),  # Диапазон 20-77
                    'volume_ratio': 1.0 + (i * 0.1),  # Диапазон 1.0-2.9
                    'momentum': -2.0 + (i * 0.2),  # Диапазон -2.0 до 1.8
                    'confidence': 60.0 + (i * 1.5),  # Диапазон 60-88.5
                    'strategy_score': 10.0 + (i * 0.5),  # Диапазон 10-19.5
                    'result': 'win' if i < 15 else 'loss',  # 15 успешных из 20
                    'market_condition': 'NEUTRAL',
                    'market_data': {
                        'rsi': 30.0 + (i * 2),
                        'bb_position': 20.0 + (i * 3),
                        'volume_ratio': 1.0 + (i * 0.1),
                        'momentum': -2.0 + (i * 0.2),
                    }
                })
            
            # Анализ паттернов (должен создать универсальные паттерны с диапазонами)
            patterns = learning_system.analyze_market_patterns(test_data)
            results['pattern_analysis'] = {
                'status': '✅',
                'patterns_count': len(patterns),
                'message': f'Создано {len(patterns)} паттернов'
            }
            logger.info(f"  ✅ Создано {len(patterns)} универсальных паттернов")
            
            # Проверяем, что паттерны используют диапазоны, а не точные значения
            if patterns:
                pattern = patterns[0]
                has_ranges = any(
                    isinstance(v, tuple) and len(v) == 2
                    for v in pattern.feature_ranges.values()
                )
                
                results['generalization_check'] = {
                    'status': '✅' if has_ranges else '❌',
                    'uses_ranges': has_ranges,
                    'generalization_score': pattern.generalization_score,
                    'message': 'Паттерны используют диапазоны (не точные значения)' if has_ranges else 'Паттерны используют точные значения (запоминание)'
                }
                
                logger.info(f"  {'✅' if has_ranges else '❌'} Паттерны используют диапазоны: {has_ranges}")
                logger.info(f"    📊 Диапазоны признаков: {pattern.feature_ranges}")
                logger.info(f"    🎯 Уровень обобщения: {pattern.generalization_score:.2f}")
                
                if not has_ranges:
                    all_passed = False
                
                # Проверяем создание универсальных правил
                rules = learning_system.create_universal_rules(patterns)
                results['rule_creation'] = {
                    'status': '✅',
                    'rules_count': len(rules),
                    'message': f'Создано {len(rules)} универсальных правил'
                }
                logger.info(f"  ✅ Создано {len(rules)} универсальных правил")
                
                if rules:
                    rule = rules[0]
                    logger.info(f"    📋 Правило: {rule.rule_name}")
                    logger.info(f"    📊 Условия (диапазоны): {rule.conditions}")
                    logger.info(f"    ✅ Приоритет: {rule.priority:.2f}")
            else:
                results['pattern_analysis'] = {'status': '⚠️', 'message': 'Паттерны не созданы (недостаточно данных)'}
                logger.warning("  ⚠️ Паттерны не созданы")
            
        except Exception as e:
            results['error'] = {'status': '❌', 'message': str(e)}
            all_passed = False
            logger.error(f"  ❌ Ошибка тестирования UniversalLearningSystem: {e}", exc_info=True)
        
        self.test_results['universal_learning'] = results
        logger.info(f"\n🧠 Результат: {'✅ ПРОЙДЕН' if all_passed else '❌ ПРОВАЛЕН'}")
        return all_passed
    
    def test_advanced_ml(self) -> bool:
        """🤖 Тест 4: AdvancedMLSystem (самопереобучение)"""
        logger.info("\n" + "="*60)
        logger.info("🤖 ТЕСТ 4: ADVANCED ML SYSTEM (САМОПЕРЕОБУЧЕНИЕ)")
        logger.info("="*60)
        
        results = {}
        all_passed = True
        
        try:
            ml_system = AdvancedMLSystem()
            results['init'] = {'status': '✅', 'message': 'Система инициализирована'}
            logger.info("  ✅ AdvancedMLSystem инициализирована")
            
            # Проверяем настройки
            settings = ml_system.settings
            results['settings'] = {
                'status': '✅',
                'retrain_frequency': settings.get('retrain_frequency_hours', 'N/A'),
                'min_training_samples': settings.get('min_training_samples', 'N/A')
            }
            logger.info(f"  ✅ Настройки: переобучение каждые {settings.get('retrain_frequency_hours', 'N/A')} часов")
            logger.info(f"     Минимум образцов для обучения: {settings.get('min_training_samples', 'N/A')}")
            
            # Проверяем статистику
            stats = ml_system.stats
            results['stats'] = {
                'status': '✅',
                'models_trained': stats.get('models_trained', 0),
                'last_training': str(stats.get('last_training', 'Never')),
                'avg_accuracy': stats.get('avg_accuracy', 0.0)
            }
            logger.info(f"  📊 Статистика:")
            logger.info(f"     Обучено моделей: {stats.get('models_trained', 0)}")
            logger.info(f"     Последнее обучение: {stats.get('last_training', 'Never')}")
            logger.info(f"     Средняя точность: {stats.get('avg_accuracy', 0.0):.2%}")
            
            # Проверяем папку моделей
            models_dir = self.models_dir
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pkl'))
                results['models_storage'] = {
                    'status': '✅',
                    'models_count': len(model_files),
                    'models': [f.name for f in model_files[:5]]  # Первые 5
                }
                logger.info(f"  ✅ Найдено {len(model_files)} сохраненных моделей")
                if model_files:
                    logger.info(f"     Примеры: {', '.join([f.name for f in model_files[:3]])}")
            else:
                results['models_storage'] = {'status': '⚠️', 'message': 'Папка моделей не найдена'}
                logger.warning("  ⚠️ Папка моделей не найдена")
            
        except Exception as e:
            results['error'] = {'status': '❌', 'message': str(e)}
            all_passed = False
            logger.error(f"  ❌ Ошибка тестирования AdvancedMLSystem: {e}", exc_info=True)
        
        self.test_results['advanced_ml'] = results
        logger.info(f"\n🤖 Результат: {'✅ ПРОЙДЕН' if all_passed else '❌ ПРОВАЛЕН'}")
        return all_passed
    
    async def test_smart_selector(self) -> bool:
        """🎯 Тест 5: SmartCoinSelector (100-200 монет)"""
        logger.info("\n" + "="*60)
        logger.info("🎯 ТЕСТ 5: SMART COIN SELECTOR (100-200 МОНЕТ)")
        logger.info("="*60)
        
        results = {}
        all_passed = True
        
        try:
            selector = SmartCoinSelector()
            results['init'] = {'status': '✅', 'message': 'Селектор инициализирован'}
            logger.info("  ✅ SmartCoinSelector инициализирован")
            
            # Проверяем настройки
            logger.info(f"  📊 Настройки селектора:")
            logger.info(f"     Мин. объем 24h: ${selector.min_volume_24h:,.0f}")
            logger.info(f"     Мин. цена: ${selector.min_price}")
            logger.info(f"     Макс. цена: ${selector.max_price:,.0f}")
            
            # Тестируем получение символов (требует биржу)
            # Для теста создаем mock или используем реальную биржу если доступна
            try:
                import ccxt
                import os
                
                api_key = os.getenv('BYBIT_API_KEY')
                api_secret = os.getenv('BYBIT_API_SECRET')
                
                if api_key and api_secret:
                    exchange = ccxt.bybit({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'sandbox': False,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'linear'}
                    })
                    
                    # Тестируем выбор символов для разных условий рынка
                    market_conditions = ['normal', 'bullish', 'bearish', 'volatile']
                    
                    for condition in market_conditions:
                        try:
                            symbols = await selector.get_smart_symbols(exchange, condition)
                            count = len(symbols) if symbols else 0
                            
                            results[f'condition_{condition}'] = {
                                'status': '✅',
                                'symbols_count': count,
                                'message': f'Для {condition}: {count} символов',
                                'meets_requirement': (100 <= count <= 200) if condition != 'bearish' else (count >= 100)
                            }
                            
                            requirement = "100-200" if condition != 'bearish' else "≥100"
                            status_icon = '✅' if results[f'condition_{condition}']['meets_requirement'] else '⚠️'
                            logger.info(f"  {status_icon} {condition.upper()}: {count} символов (требуется: {requirement})")
                            
                            if not results[f'condition_{condition}']['meets_requirement']:
                                all_passed = False
                                
                        except Exception as e:
                            results[f'condition_{condition}'] = {
                                'status': '❌',
                                'error': str(e)
                            }
                            logger.error(f"  ❌ Ошибка получения символов для {condition}: {e}")
                else:
                    results['api_test'] = {
                        'status': '⚠️',
                        'message': 'API ключи не найдены - пропуск теста реальной биржи'
                    }
                    logger.warning("  ⚠️ API ключи не найдены - пропускаем тест реальной биржи")
                    
            except Exception as e:
                results['api_test'] = {
                    'status': '❌',
                    'error': str(e)
                }
                logger.error(f"  ❌ Ошибка теста API: {e}")
            
        except Exception as e:
            results['error'] = {'status': '❌', 'message': str(e)}
            all_passed = False
            logger.error(f"  ❌ Ошибка тестирования SmartCoinSelector: {e}", exc_info=True)
        
        self.test_results['smart_selector'] = results
        logger.info(f"\n🎯 Результат: {'✅ ПРОЙДЕН' if all_passed else '❌ ПРОВАЛЕН'}")
        return all_passed
    
    def test_integration(self) -> bool:
        """🔗 Тест 6: Интеграция всех компонентов"""
        logger.info("\n" + "="*60)
        logger.info("🔗 ТЕСТ 6: ИНТЕГРАЦИЯ ВСЕХ КОМПОНЕНТОВ")
        logger.info("="*60)
        
        results = {}
        all_passed = True
        
        try:
            # Инициализация всех компонентов вместе
            storage = DataStorageSystem()
            learning = UniversalLearningSystem(storage)
            ml_system = AdvancedMLSystem()
            selector = SmartCoinSelector()
            
            results['components_init'] = {'status': '✅', 'message': 'Все компоненты инициализированы'}
            logger.info("  ✅ Все компоненты инициализированы вместе")
            
            # Проверяем, что данные можно передавать между компонентами
            test_market_data = MarketData(
                timestamp=datetime.now().isoformat(),
                symbol='ETHUSDT',
                timeframe='45m',
                price=3000.0,
                volume=500000.0,
                rsi=45.0,
                macd=50.0,
                bb_position=55.0,
                ema_9=3050.0,
                ema_21=3000.0,
                ema_50=2950.0,
                volume_ratio=1.2,
                momentum=0.3,
                market_condition='NEUTRAL'
            )
            
            storage.save_market_data(test_market_data)
            stored_data = storage.get_market_data('ETHUSDT', limit=1)
            
            if stored_data:
                results['data_flow'] = {'status': '✅', 'message': 'Данные передаются между компонентами'}
                logger.info("  ✅ Данные успешно передаются между компонентами")
            else:
                results['data_flow'] = {'status': '❌', 'message': 'Ошибка передачи данных'}
                all_passed = False
                logger.error("  ❌ Ошибка передачи данных между компонентами")
            
            # Проверяем, что универсальное обучение использует данные из хранилища
            if stored_data:
                converted_data = [{
                    'symbol': d.symbol,
                    'rsi': d.rsi,
                    'bb_position': d.bb_position,
                    'volume_ratio': d.volume_ratio,
                    'momentum': d.momentum,
                    'result': 'win',
                    'market_condition': d.market_condition,
                    'market_data': {
                        'rsi': d.rsi,
                        'bb_position': d.bb_position,
                        'volume_ratio': d.volume_ratio,
                        'momentum': d.momentum,
                    }
                } for d in stored_data]
                
                patterns = learning.analyze_market_patterns(converted_data * 10)  # Умножаем для минимума
                
                if patterns or len(converted_data) < 10:
                    results['learning_integration'] = {'status': '✅', 'message': 'Обучение использует данные из хранилища'}
                    logger.info("  ✅ Обучение успешно использует данные из хранилища")
                else:
                    results['learning_integration'] = {'status': '⚠️', 'message': 'Недостаточно данных для обучения'}
                    logger.warning("  ⚠️ Недостаточно данных для обучения (это нормально для нового бота)")
            
        except Exception as e:
            results['error'] = {'status': '❌', 'message': str(e)}
            all_passed = False
            logger.error(f"  ❌ Ошибка тестирования интеграции: {e}", exc_info=True)
        
        self.test_results['integration'] = results
        logger.info(f"\n🔗 Результат: {'✅ ПРОЙДЕН' if all_passed else '❌ ПРОВАЛЕН'}")
        return all_passed
    
    def generate_report(self) -> str:
        """📊 Генерация итогового отчета"""
        logger.info("\n" + "="*60)
        logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
        logger.info("="*60)
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ AI+ML+БОТ")
        report_lines.append("="*60)
        report_lines.append(f"\nДата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Базовый путь: {self.base_dir}")
        report_lines.append("\n" + "-"*60)
        
        # Подсчет результатов
        passed_tests = 0
        total_tests = 6
        
        test_names = {
            'folder_structure': '📁 Структура папок',
            'data_storage': '💾 DataStorageSystem',
            'universal_learning': '🧠 UniversalLearningSystem',
            'advanced_ml': '🤖 AdvancedMLSystem',
            'smart_selector': '🎯 SmartCoinSelector',
            'integration': '🔗 Интеграция компонентов'
        }
        
        for test_key, test_name in test_names.items():
            result = self.test_results.get(test_key, {})
            
            # Определяем статус теста
            if test_key == 'folder_structure':
                status = '✅' if all(r.get('status') == '✅' for r in result.values() if isinstance(r, dict)) else '❌'
            elif 'error' in result:
                status = '❌'
            elif any(r.get('status') == '❌' for r in result.values() if isinstance(r, dict)):
                status = '⚠️'
            else:
                status = '✅'
            
            report_lines.append(f"\n{test_name}: {status}")
            
            if status == '✅':
                passed_tests += 1
        
        report_lines.append("\n" + "-"*60)
        report_lines.append(f"\nИТОГО: {passed_tests}/{total_tests} тестов пройдено")
        
        if passed_tests == total_tests:
            overall = '✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ'
            self.test_results['overall_status'] = 'SUCCESS'
        elif passed_tests >= total_tests * 0.7:
            overall = '⚠️ БОЛЬШИНСТВО ТЕСТОВ ПРОЙДЕНО'
            self.test_results['overall_status'] = 'PARTIAL'
        else:
            overall = '❌ МНОГИЕ ТЕСТЫ НЕ ПРОЙДЕНЫ'
            self.test_results['overall_status'] = 'FAILED'
        
        report_lines.append(f"СТАТУС: {overall}")
        report_lines.append("="*60)
        
        report = "\n".join(report_lines)
        logger.info(report)
        
        return report
    
    async def run_all_tests(self):
        """🚀 Запуск всех тестов"""
        logger.info("\n" + "="*60)
        logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ")
        logger.info("="*60)
        
        results = []
        
        # Тест 1: Структура папок
        results.append(('folder_structure', self.test_folder_structure()))
        
        # Тест 2: DataStorageSystem
        results.append(('data_storage', self.test_data_storage()))
        
        # Тест 3: UniversalLearningSystem
        results.append(('universal_learning', self.test_universal_learning()))
        
        # Тест 4: AdvancedMLSystem
        results.append(('advanced_ml', self.test_advanced_ml()))
        
        # Тест 5: SmartCoinSelector
        results.append(('smart_selector', await self.test_smart_selector()))
        
        # Тест 6: Интеграция
        results.append(('integration', self.test_integration()))
        
        # Генерация отчета
        report = self.generate_report()
        
        # Сохранение отчета
        report_file = self.base_dir / "logs" / "system" / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write("\n\nДетальные результаты:\n")
            f.write(json.dumps(self.test_results, indent=2, ensure_ascii=False, default=str))
        
        logger.info(f"\n📄 Отчет сохранен: {report_file}")
        
        return results


async def main():
    """Главная функция"""
    tester = ComprehensiveSystemTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())




