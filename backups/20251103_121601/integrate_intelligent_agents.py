#!/usr/bin/env python3
"""
🔗 ИНТЕГРАЦИЯ ИНТЕЛЛЕКТУАЛЬНЫХ АГЕНТОВ С ОСНОВНОЙ СИСТЕМОЙ
"""

import asyncio
try:
    from intelligent_agents import IntelligentAgentsSystem
except ImportError:
    IntelligentAgentsSystem = None

try:
    from system_agents import AgentCleaner, AgentSecurity, AgentStability, AgentRecovery
except ImportError:
    AgentCleaner = None
    AgentSecurity = None
    AgentStability = None
    AgentRecovery = None

try:
    from api_optimizer import APIOptimizer
except ImportError:
    APIOptimizer = None

class IntegratedAgentsManager:
    """Менеджер интеллектуальных агентов"""
    
    def __init__(self, bot_dir: str = "/opt/bot", bot_pid: int = None):
        self.bot_dir = bot_dir
        self.bot_pid = bot_pid
        
        # Проверяем доступность модулей
        if IntelligentAgentsSystem is None:
            raise ImportError("IntelligentAgentsSystem недоступен. Установите intelligent_agents.py")
        
        if AgentCleaner is None or AgentSecurity is None:
            raise ImportError("System agents недоступны. Установите system_agents.py")
        
        # Инициализируем систему интеллектуальных агентов
        self.intelligent_system = IntelligentAgentsSystem()
        
        # Создаем интеллектуальных агентов
        self.cleaner = self.intelligent_system.create_agent('cleaner_agent', 'cleaner')
        self.security = self.intelligent_system.create_agent('security_agent', 'security')
        self.stability = self.intelligent_system.create_agent('stability_agent', 'stability')
        self.recovery = self.intelligent_system.create_agent('recovery_agent', 'recovery')
        
        # Инициализируем обычные агенты (они теперь наследники IntelligentAgent)
        self.cleaner_agent = AgentCleaner(bot_dir)
        self.security_agent = AgentSecurity(bot_dir)
        self.stability_agent = AgentStability(bot_pid)
        self.recovery_agent = AgentRecovery(bot_dir)
        
    async def run_all_with_learning(self):
        """Запустить всех агентов с обучением"""
        # Выполняем задачи агентов
        agent_results = await asyncio.gather(
            self.cleaner_agent.run(),
            self.security_agent.run(),
            self.stability_agent.run(),
            self.recovery_agent.run(),
            return_exceptions=True
        )
        
        # Цикл обучения агентов
        await self.intelligent_system.run_learning_cycle()
        
        return {
            'cleaner': agent_results[0],
            'security': agent_results[1],
            'stability': agent_results[2],
            'recovery': agent_results[3],
            'learning_completed': True
        }
    
    async def run_periodic_with_learning(self, 
                                        cleaner_interval: int = 3600,
                                        security_interval: int = 1800,
                                        stability_interval: int = 300,
                                        recovery_interval: int = 7200,
                                        learning_interval: int = 900):  # 15 минут
        """Периодический запуск с обучением"""
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        last_learning = time.time()
        last_stability = 0
        last_security = 0
        last_cleaner = 0
        last_recovery = 0
        
        while True:
            current_time = time.time()
            
            # Задачи агентов с проверкой интервалов
            if current_time - last_stability >= stability_interval:
                await self.stability_agent.run()
                last_stability = current_time
            
            if current_time - last_security >= security_interval:
                await self.security_agent.run()
                last_security = current_time
            
            if current_time - last_cleaner >= cleaner_interval:
                await self.cleaner_agent.run()
                last_cleaner = current_time
            
            if current_time - last_recovery >= recovery_interval:
                await self.recovery_agent.run()
                last_recovery = current_time
            
            # Обучение каждые learning_interval
            if current_time - last_learning >= learning_interval:
                await self.intelligent_system.run_learning_cycle()
                last_learning = current_time
                
                # Выводим статистику
                stats = self.intelligent_system.get_agent_statistics()
                logger.info(f"🧠 Статистика обучения агентов: {stats}")
            
            await asyncio.sleep(60)  # Проверка каждую минуту
    
    def get_all_statistics(self):
        """Получить всю статистику"""
        return {
            'intelligent_agents': self.intelligent_system.get_agent_statistics(),
            'regular_agents': {
                'cleaner': {'status': 'active'},
                'security': {'status': 'active'},
                'stability': {'status': 'active'},
                'recovery': {'status': 'active'}
            }
        }

