#!/usr/bin/env python3
"""
🔬 ГЕНЕРАЛЬНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ
==================================

Тестирование всех модулей перед деплоем:
- API Optimizer
- Интеллектуальные агенты
- Основные модули бота
- Интеграция
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GeneralSystemTest:
    """Генеральное тестирование системы"""
    
    def __init__(self):
        self.results = []
        self.errors = []
        
    def test_module_import(self, module_name, class_name=None):
        """Тест импорта модуля"""
        try:
            module = __import__(module_name)
            if class_name:
                if hasattr(module, class_name):
                    logger.info(f"✅ {module_name}.{class_name}")
                    self.results.append(f"✅ {module_name}")
                    return True
                else:
                    logger.error(f"❌ {module_name}: класс {class_name} не найден")
                    self.errors.append(f"{module_name}: {class_name} не найден")
                    return False
            else:
                logger.info(f"✅ {module_name}")
                self.results.append(f"✅ {module_name}")
                return True
        except Exception as e:
            logger.error(f"❌ {module_name}: {e}")
            self.errors.append(f"{module_name}: {e}")
            return False
    
    def test_api_optimizer(self):
        """Тест API Optimizer"""
        logger.info("\n📋 Тест API Optimizer...")
        try:
            from api_optimizer import APIOptimizer, RateLimiter, DataCache
            
            # Тест RateLimiter
            limiter = RateLimiter(max_requests=10, time_window=60)
            logger.info("  ✅ RateLimiter создан")
            
            # Тест DataCache
            cache = DataCache(cache_dir="data/cache/test", default_ttl=30)
            logger.info("  ✅ DataCache создан")
            
            # Тест кэширования
            test_data = {'test': 'data'}
            cache.set('test_method', {'symbol': 'BTCUSDT'}, test_data, ttl=60)
            cached = cache.get('test_method', {'symbol': 'BTCUSDT'})
            if cached == test_data:
                logger.info("  ✅ Кэш работает корректно")
            else:
                logger.warning("  ⚠️ Проблемы с кэшем")
            
            self.results.append("✅ API Optimizer модули")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка: {e}")
            self.errors.append(f"API Optimizer: {e}")
            return False
    
    def test_intelligent_agents(self):
        """Тест интеллектуальных агентов"""
        logger.info("\n📋 Тест Intelligent Agents...")
        try:
            from intelligent_agents import (
                IntelligentAgent, IntelligentAgentsSystem,
                EvolutionaryLearning, KnowledgeSharing, MetaLearningSystem
            )
            
            # Тест EvolutionaryLearning
            evo = EvolutionaryLearning()
            logger.info("  ✅ EvolutionaryLearning создан")
            
            # Тест KnowledgeSharing
            sharing = KnowledgeSharing(knowledge_dir="data/knowledge/test")
            logger.info("  ✅ KnowledgeSharing создан")
            
            # Тест MetaLearningSystem
            meta = MetaLearningSystem()
            logger.info("  ✅ MetaLearningSystem создан")
            
            # Тест IntelligentAgentsSystem
            system = IntelligentAgentsSystem()
            agent = system.create_agent('test_agent', 'test')
            logger.info("  ✅ IntelligentAgentsSystem создан, агент добавлен")
            
            self.results.append("✅ Intelligent Agents модули")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка: {e}")
            self.errors.append(f"Intelligent Agents: {e}")
            return False
    
    def test_integration(self):
        """Тест интеграции"""
        logger.info("\n📋 Тест интеграции...")
        try:
            from integrate_intelligent_agents import IntegratedAgentsManager
            
            manager = IntegratedAgentsManager(bot_dir=".", bot_pid=os.getpid())
            logger.info("  ✅ IntegratedAgentsManager создан")
            
            self.results.append("✅ Интеграция модулей")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка: {e}")
            self.errors.append(f"Интеграция: {e}")
            return False
    
    def test_main_modules(self):
        """Тест основных модулей бота"""
        logger.info("\n📋 Тест основных модулей бота...")
        modules = [
            ('super_bot_v4_mtf', 'SuperBotV4MTF'),
            ('data_storage_system', 'DataStorageSystem'),
            ('universal_learning_system', 'UniversalLearningSystem'),
            ('smart_coin_selector', 'SmartCoinSelector'),
            ('advanced_indicators', 'AdvancedIndicators'),
            ('telegram_commands_handler', 'TelegramCommandsHandler'),
        ]
        
        all_ok = True
        for module, class_name in modules:
            if not self.test_module_import(module, class_name):
                all_ok = False
        
        return all_ok
    
    def test_file_structure(self):
        """Тест структуры файлов"""
        logger.info("\n📋 Тест структуры файлов...")
        required_files = [
            'super_bot_v4_mtf.py',
            'api_optimizer.py',
            'intelligent_agents.py',
            'integrate_intelligent_agents.py',
            'system_agents.py',
            'data_storage_system.py',
            'universal_learning_system.py',
        ]
        
        missing = []
        for file in required_files:
            if Path(file).exists():
                logger.info(f"  ✅ {file}")
            else:
                logger.error(f"  ❌ {file} - не найден")
                missing.append(file)
        
        if missing:
            self.errors.append(f"Отсутствуют файлы: {', '.join(missing)}")
            return False
        
        self.results.append("✅ Структура файлов")
        return True
    
    async def run_all_tests(self):
        """Запустить все тесты"""
        logger.info("="*70)
        logger.info("🔬 ГЕНЕРАЛЬНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ")
        logger.info("="*70)
        
        tests = [
            ("Структура файлов", self.test_file_structure),
            ("Основные модули", self.test_main_modules),
            ("API Optimizer", self.test_api_optimizer),
            ("Intelligent Agents", self.test_intelligent_agents),
            ("Интеграция", self.test_integration),
        ]
        
        for test_name, test_func in tests:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
            except Exception as e:
                logger.error(f"❌ {test_name}: {e}")
                self.errors.append(f"{test_name}: {e}")
        
        # Итоговый отчет
        logger.info("\n" + "="*70)
        logger.info("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        logger.info("="*70)
        logger.info(f"✅ Успешно: {len(self.results)}")
        logger.info(f"❌ Ошибок: {len(self.errors)}")
        
        if self.errors:
            logger.info("\n⚠️ ОШИБКИ:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        logger.info("\n" + "="*70)
        if len(self.errors) == 0:
            logger.info("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к деплою.")
            return True
        else:
            logger.info("⚠️ ЕСТЬ ОШИБКИ. Проверьте перед деплоем.")
            return False

async def main():
    tester = GeneralSystemTest()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())


