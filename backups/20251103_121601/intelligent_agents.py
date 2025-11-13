#!/usr/bin/env python3
"""
🤖 ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА САМООБУЧАЮЩИХСЯ АГЕНТОВ
==================================================

Система агентов с:
- Самообучением через универсальные правила (не запоминание)
- Эволюционным обучением (мутации, кроссовер)
- Обменом знаниями между агентами
- Мета-обучением (обучение учиться)
- Коллективным интеллектом
"""

import asyncio
import json
import random
import statistics
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class AgentKnowledge:
    """Знания агента (универсальные правила, не точные решения)"""
    agent_id: str
    agent_type: str  # 'cleaner', 'security', 'stability', 'recovery'
    learned_rules: Dict[str, Dict] = field(default_factory=dict)  # Универсальные правила
    adaptation_strategies: Dict[str, float] = field(default_factory=dict)  # Стратегии адаптации
    performance_history: List[Dict] = field(default_factory=list)
    success_rate: float = 0.0
    generalization_score: float = 0.0
    last_updated: str = ""
    evolution_generation: int = 0

@dataclass
class LearningPattern:
    """Паттерн обучения (диапазоны условий)"""
    pattern_id: str
    condition_ranges: Dict[str, Tuple[float, float]]  # Диапазоны, не точные значения
    action: str
    success_count: int
    failure_count: int
    market_conditions: List[str]
    generalization_level: float
    created_from_examples: int

class EvolutionaryLearning:
    """🧬 Эволюционное обучение для агентов"""
    
    def __init__(self):
        self.mutation_rate = 0.15
        self.crossover_rate = 0.25
        self.selection_pressure = 0.3
        self.generation = 0
        
    def create_rule_from_patterns(self, patterns: List[Dict], min_samples: int = 5) -> Dict:
        """Создать универсальное правило из паттернов (диапазоны)"""
        if len(patterns) < min_samples:
            return None
        
        # Извлекаем значения для каждого признака
        feature_ranges = {}
        for feature in patterns[0].get('features', {}).keys():
            values = []
            for pattern in patterns:
                val = pattern.get('features', {}).get(feature)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        continue
            
            if len(values) >= 3:
                # Создаем диапазон с запасом (не точное значение!)
                min_val = min(values)
                max_val = max(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                # Расширяем диапазон для обобщения
                range_min = min_val - (std_val * 0.5)
                range_max = max_val + (std_val * 0.5)
                
                feature_ranges[feature] = (range_min, range_max)
        
        # Вычисляем успешность
        successful = sum(1 for p in patterns if p.get('result') == 'success')
        success_rate = successful / len(patterns)
        
        # Определяем обобщающую способность
        generalization = self._calculate_generalization(patterns, feature_ranges)
        
        return {
            'feature_ranges': feature_ranges,
            'success_rate': success_rate,
            'generalization_score': generalization,
            'sample_size': len(patterns),
            'market_conditions': list(set([p.get('market_condition', 'NEUTRAL') for p in patterns]))
        }
    
    def _calculate_generalization(self, patterns: List[Dict], ranges: Dict) -> float:
        """Вычислить уровень обобщения (0-1)"""
        if not ranges:
            return 0.0
        
        # Ширина диапазонов относительно их значений
        range_widths = []
        for feature, (min_val, max_val) in ranges.items():
            if abs(min_val) > 0.0001:
                width = (max_val - min_val) / abs(min_val)
                range_widths.append(width)
        
        if not range_widths:
            return 0.0
        
        # Чем шире диапазон (но не слишком), тем лучше обобщение
        avg_width = statistics.mean(range_widths)
        # Нормализуем (оптимальная ширина ~0.3-0.5)
        generalization = min(avg_width / 0.5, 1.0) if avg_width > 0 else 0.0
        
        return generalization
    
    def mutate_rule(self, rule: Dict, mutation_strength: float = 0.1) -> Dict:
        """Мутация правила для эволюции"""
        mutated_rule = rule.copy()
        
        if random.random() < self.mutation_rate:
            # Мутируем диапазоны
            if 'feature_ranges' in mutated_rule:
                for feature, (min_val, max_val) in mutated_rule['feature_ranges'].items():
                    if random.random() < 0.3:  # Вероятность мутации признака
                        range_width = max_val - min_val
                        mutation = range_width * mutation_strength * random.uniform(-1, 1)
                        
                        mutated_rule['feature_ranges'][feature] = (
                            min_val + mutation,
                            max_val + mutation
                        )
            
            # Мутируем параметры стратегии
            if 'strategy_params' in mutated_rule:
                for param, value in mutated_rule['strategy_params'].items():
                    if isinstance(value, (int, float)):
                        mutation = value * mutation_strength * random.uniform(-0.2, 0.2)
                        mutated_rule['strategy_params'][param] = value + mutation
        
        return mutated_rule
    
    def crossover_rules(self, rule1: Dict, rule2: Dict) -> Dict:
        """Скрещивание двух правил"""
        if random.random() > self.crossover_rate:
            return rule1  # Нет скрещивания
        
        offspring = rule1.copy()
        
        # Скрещиваем диапазоны признаков
        if 'feature_ranges' in rule1 and 'feature_ranges' in rule2:
            for feature in set(list(rule1['feature_ranges'].keys()) + list(rule2['feature_ranges'].keys())):
                if feature in rule1['feature_ranges'] and feature in rule2['feature_ranges']:
                    # Берем среднее значение диапазонов
                    min1, max1 = rule1['feature_ranges'][feature]
                    min2, max2 = rule2['feature_ranges'][feature]
                    
                    offspring['feature_ranges'][feature] = (
                        (min1 + min2) / 2,
                        (max1 + max2) / 2
                    )
        
        return offspring
    
    def select_best_rules(self, rules: List[Dict], top_n: int = 10) -> List[Dict]:
        """Отбор лучших правил (эволюционный отбор)"""
        # Сортируем по комбинации успешности и обобщения
        scored_rules = []
        for rule in rules:
            score = (
                rule.get('success_rate', 0) * 0.6 +
                rule.get('generalization_score', 0) * 0.4
            )
            scored_rules.append((score, rule))
        
        scored_rules.sort(reverse=True)
        
        # Отбираем топ-N
        return [rule for _, rule in scored_rules[:top_n]]

class KnowledgeSharing:
    """🤝 Система обмена знаниями между агентами"""
    
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.shared_knowledge = {}
        self.knowledge_updates = deque(maxlen=100)
        
    def save_agent_knowledge(self, agent_id: str, knowledge: AgentKnowledge):
        """Сохранить знания агента"""
        try:
            knowledge_file = self.knowledge_dir / f"{agent_id}_knowledge.json"
            
            knowledge_dict = {
                'agent_id': knowledge.agent_id,
                'agent_type': knowledge.agent_type,
                'learned_rules': knowledge.learned_rules,
                'adaptation_strategies': knowledge.adaptation_strategies,
                'success_rate': knowledge.success_rate,
                'generalization_score': knowledge.generalization_score,
                'last_updated': datetime.now().isoformat(),
                'evolution_generation': knowledge.evolution_generation
            }
            
            with open(knowledge_file, 'w') as f:
                json.dump(knowledge_dict, f, indent=2)
            
            # Обновляем общее хранилище
            self.shared_knowledge[agent_id] = knowledge_dict
            self.knowledge_updates.append({
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat(),
                'update_type': 'knowledge_saved'
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения знаний агента {agent_id}: {e}")
    
    def load_agent_knowledge(self, agent_id: str) -> Optional[AgentKnowledge]:
        """Загрузить знания агента"""
        try:
            knowledge_file = self.knowledge_dir / f"{agent_id}_knowledge.json"
            
            if not knowledge_file.exists():
                return None
            
            with open(knowledge_file, 'r') as f:
                data = json.load(f)
            
            return AgentKnowledge(
                agent_id=data['agent_id'],
                agent_type=data['agent_type'],
                learned_rules=data.get('learned_rules', {}),
                adaptation_strategies=data.get('adaptation_strategies', {}),
                success_rate=data.get('success_rate', 0.0),
                generalization_score=data.get('generalization_score', 0.0),
                last_updated=data.get('last_updated', ''),
                evolution_generation=data.get('evolution_generation', 0)
            )
        
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки знаний агента {agent_id}: {e}")
            return None
    
    def share_knowledge(self, from_agent: str, to_agent: str, rule_name: str) -> bool:
        """Поделиться правилом между агентами"""
        try:
            from_knowledge = self.load_agent_knowledge(from_agent)
            if not from_knowledge or rule_name not in from_knowledge.learned_rules:
                return False
            
            to_knowledge = self.load_agent_knowledge(to_agent)
            if not to_knowledge:
                # Создаем новое знание для получателя
                to_knowledge = AgentKnowledge(
                    agent_id=to_agent,
                    agent_type=from_knowledge.agent_type
                )
            
            # Адаптируем правило под тип агента получателя
            rule = from_knowledge.learned_rules[rule_name].copy()
            rule['shared_from'] = from_agent
            rule['adaptation_date'] = datetime.now().isoformat()
            
            to_knowledge.learned_rules[rule_name] = rule
            self.save_agent_knowledge(to_agent, to_knowledge)
            
            logger.info(f"🤝 Знание '{rule_name}' передано от {from_agent} к {to_agent}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Ошибка обмена знаниями: {e}")
            return False
    
    def find_similar_agents(self, agent_type: str) -> List[str]:
        """Найти агентов того же типа для обмена опытом"""
        similar = []
        for agent_id, knowledge_data in self.shared_knowledge.items():
            if knowledge_data.get('agent_type') == agent_type:
                similar.append(agent_id)
        return similar
    
    def get_collective_wisdom(self, agent_type: str) -> Dict:
        """Получить коллективную мудрость для типа агентов"""
        all_rules = defaultdict(list)
        
        for agent_id, knowledge_data in self.shared_knowledge.items():
            if knowledge_data.get('agent_type') == agent_type:
                for rule_name, rule_data in knowledge_data.get('learned_rules', {}).items():
                    all_rules[rule_name].append({
                        'source': agent_id,
                        'success_rate': rule_data.get('success_rate', 0),
                        'rule': rule_data
                    })
        
        # Находим лучшие правила на основе коллективного опыта
        collective_rules = {}
        for rule_name, rule_instances in all_rules.items():
            # Сортируем по успешности
            rule_instances.sort(key=lambda x: x['success_rate'], reverse=True)
            # Берем лучшее правило
            if rule_instances:
                collective_rules[rule_name] = rule_instances[0]['rule']
        
        return collective_rules

class MetaLearningSystem:
    """🧠 Мета-обучение: обучение учиться"""
    
    def __init__(self):
        self.learning_methods = {}
        self.method_performance = defaultdict(list)
        self.optimal_methods = {}
        
    def learn_optimal_learning_method(self, task_type: str, 
                                      method_params: Dict,
                                      performance: float):
        """Изучить какой метод обучения лучше для задачи"""
        if task_type not in self.learning_methods:
            self.learning_methods[task_type] = []
        
        self.learning_methods[task_type].append({
            'params': method_params,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
        
        self.method_performance[task_type].append(performance)
        
        # Обновляем оптимальный метод
        if len(self.method_performance[task_type]) > 10:
            avg_performance = statistics.mean(self.method_performance[task_type][-10:])
            if task_type not in self.optimal_methods or \
               avg_performance > self.optimal_methods[task_type].get('performance', 0):
                # Находим лучшие параметры за последние опыты
                recent_methods = self.learning_methods[task_type][-10:]
                best_method = max(recent_methods, key=lambda x: x['performance'])
                
                self.optimal_methods[task_type] = {
                    'params': best_method['params'],
                    'performance': avg_performance,
                    'last_updated': datetime.now().isoformat()
                }
        
        logger.debug(f"🧠 Мета-обучение: обновлен метод для {task_type}")
    
    def get_optimal_method(self, task_type: str) -> Optional[Dict]:
        """Получить оптимальный метод обучения для задачи"""
        if task_type not in self.optimal_methods:
            # Возвращаем дефолтные параметры
            return {
                'mutation_rate': 0.15,
                'crossover_rate': 0.25,
                'selection_pressure': 0.3,
                'min_samples': 5
            }
        
        return self.optimal_methods[task_type].get('params')

class IntelligentAgent:
    """🤖 Базовый интеллектуальный агент с самообучением"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.evolutionary_learning = EvolutionaryLearning()
        self.knowledge_sharing = KnowledgeSharing()
        self.meta_learning = MetaLearningSystem()
        
        self.knowledge = self.knowledge_sharing.load_agent_knowledge(agent_id)
        if not self.knowledge:
            self.knowledge = AgentKnowledge(
                agent_id=agent_id,
                agent_type=agent_type,
                last_updated=datetime.now().isoformat()
            )
        
        self.experience_history = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=100)
        
    def learn_from_experience(self, experience: Dict):
        """Обучение на опыте (создание универсальных правил)"""
        self.experience_history.append({
            **experience,
            'timestamp': datetime.now().isoformat()
        })
        
        # Группируем опыт по результатам
        successful_experiences = [
            e for e in self.experience_history
            if e.get('result') == 'success'
        ]
        
        failed_experiences = [
            e for e in self.experience_history
            if e.get('result') == 'failure'
        ]
        
        # Создаем правила из успешного опыта (минимум 5 примеров)
        if len(successful_experiences) >= 5:
            rule = self.evolutionary_learning.create_rule_from_patterns(
                successful_experiences,
                min_samples=5
            )
            
            if rule and rule.get('generalization_score', 0) > 0.5:
                rule_name = f"{self.agent_type}_rule_{len(self.knowledge.learned_rules)}"
                rule['created_at'] = datetime.now().isoformat()
                rule['agent_id'] = self.agent_id
                
                self.knowledge.learned_rules[rule_name] = rule
                self.knowledge.success_rate = rule['success_rate']
                self.knowledge.generalization_score = rule['generalization_score']
                self.knowledge.last_updated = datetime.now().isoformat()
                
                # Сохраняем знания
                self.knowledge_sharing.save_agent_knowledge(self.agent_id, self.knowledge)
                
                logger.info(f"🧠 {self.agent_id} создал универсальное правило: {rule_name} "
                          f"(успешность: {rule['success_rate']:.1%}, обобщение: {rule['generalization_score']:.2f})")
        
        # Эволюция правил (каждые 20 опытов)
        if len(self.experience_history) % 20 == 0 and len(self.knowledge.learned_rules) > 1:
            self._evolve_rules()
    
    def _evolve_rules(self):
        """Эволюция правил (мутации и скрещивание)"""
        rules = list(self.knowledge.learned_rules.values())
        
        if len(rules) < 2:
            return
        
        # Отбираем лучшие правила
        best_rules = self.evolutionary_learning.select_best_rules(rules, top_n=5)
        
        # Создаем новые правила через мутации и скрещивание
        new_rules = []
        for i, rule in enumerate(best_rules):
            # Мутация
            mutated = self.evolutionary_learning.mutate_rule(rule.copy())
            new_rules.append(mutated)
            
            # Скрещивание с другим правилом
            if i < len(best_rules) - 1:
                crossover = self.evolutionary_learning.crossover_rules(
                    rule, best_rules[i + 1]
                )
                new_rules.append(crossover)
        
        # Тестируем новые правила и заменяем худшие
        for new_rule in new_rules[:3]:  # Берем топ-3 новых
            # Проверяем на исторических данных
            applicable_experiences = [
                e for e in self.experience_history
                if self._rule_matches_experience(new_rule, e)
            ]
            
            if applicable_experiences:
                success_count = sum(1 for e in applicable_experiences if e.get('result') == 'success')
                new_rule['success_rate'] = success_count / len(applicable_experiences)
                
                # Если новое правило лучше старого, заменяем
                worst_rule_name = min(
                    self.knowledge.learned_rules.keys(),
                    key=lambda k: self.knowledge.learned_rules[k].get('success_rate', 0)
                )
                
                if new_rule['success_rate'] > self.knowledge.learned_rules[worst_rule_name].get('success_rate', 0):
                    new_rule_name = f"{self.agent_type}_evolved_{self.knowledge.evolution_generation}"
                    new_rule['evolution_generation'] = self.knowledge.evolution_generation + 1
                    self.knowledge.learned_rules[new_rule_name] = new_rule
                    del self.knowledge.learned_rules[worst_rule_name]
        
        self.knowledge.evolution_generation += 1
        self.knowledge_sharing.save_agent_knowledge(self.agent_id, self.knowledge)
    
    def _rule_matches_experience(self, rule: Dict, experience: Dict) -> bool:
        """Проверить, подходит ли правило к опыту"""
        if 'feature_ranges' not in rule:
            return False
        
        features = experience.get('features', {})
        for feature, (min_val, max_val) in rule['feature_ranges'].items():
            if feature in features:
                value = float(features[feature])
                if not (min_val <= value <= max_val):
                    return False
        
        return True
    
    def apply_learned_rules(self, context: Dict) -> Optional[Dict]:
        """Применить изученные правила к контексту"""
        best_rule = None
        best_score = 0.0
        
        for rule_name, rule in self.knowledge.learned_rules.items():
            if self._rule_matches_experience(rule, context):
                score = (
                    rule.get('success_rate', 0) * 0.7 +
                    rule.get('generalization_score', 0) * 0.3
                )
                if score > best_score:
                    best_score = score
                    best_rule = rule
        
        return best_rule
    
    def share_knowledge_with_peers(self):
        """Поделиться знаниями с другими агентами"""
        similar_agents = self.knowledge_sharing.find_similar_agents(self.agent_type)
        
        for peer_id in similar_agents:
            if peer_id != self.agent_id:
                # Находим лучшее правило для обмена
                best_rule_name = max(
                    self.knowledge.learned_rules.keys(),
                    key=lambda k: self.knowledge.learned_rules[k].get('success_rate', 0),
                    default=None
                )
                
                if best_rule_name:
                    self.knowledge_sharing.share_knowledge(
                        self.agent_id, peer_id, best_rule_name
                    )
    
    def learn_from_collective_wisdom(self):
        """Изучить коллективную мудрость других агентов"""
        collective_rules = self.knowledge_sharing.get_collective_wisdom(self.agent_type)
        
        for rule_name, rule in collective_rules.items():
            # Адаптируем коллективное правило под себя
            if rule_name not in self.knowledge.learned_rules:
                rule_copy = rule.copy()
                rule_copy['adapted_from_collective'] = True
                rule_copy['adaptation_date'] = datetime.now().isoformat()
                self.knowledge.learned_rules[rule_name] = rule_copy
        
        self.knowledge_sharing.save_agent_knowledge(self.agent_id, self.knowledge)

class IntelligentAgentsSystem:
    """🌐 Система интеллектуальных агентов"""
    
    def __init__(self):
        self.agents = {}
        self.knowledge_sharing = KnowledgeSharing()
        self.meta_learning = MetaLearningSystem()
        
    def create_agent(self, agent_id: str, agent_type: str) -> IntelligentAgent:
        """Создать интеллектуального агента"""
        agent = IntelligentAgent(agent_id, agent_type)
        self.agents[agent_id] = agent
        return agent
    
    async def run_learning_cycle(self):
        """Запустить цикл обучения всех агентов"""
        logger.info("🧠 Запуск цикла обучения агентов")
        
        tasks = []
        for agent_id, agent in self.agents.items():
            # Агенты учатся друг у друга
            tasks.append(asyncio.create_task(self._agent_learning_process(agent)))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _agent_learning_process(self, agent: IntelligentAgent):
        """Процесс обучения агента"""
        try:
            # 1. Обучение на опыте
            if len(agent.experience_history) >= 5:
                # Агент уже создает правила из опыта в learn_from_experience
                pass
            
            # 2. Эволюция правил
            if len(agent.knowledge.learned_rules) > 1:
                agent._evolve_rules()
            
            # 3. Обмен знаниями
            agent.share_knowledge_with_peers()
            
            # 4. Изучение коллективной мудрости
            agent.learn_from_collective_wisdom()
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения агента {agent.agent_id}: {e}")
    
    def get_agent_statistics(self) -> Dict:
        """Получить статистику агентов"""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                'type': agent.agent_type,
                'rules_count': len(agent.knowledge.learned_rules),
                'success_rate': agent.knowledge.success_rate,
                'generalization_score': agent.knowledge.generalization_score,
                'experience_count': len(agent.experience_history),
                'evolution_generation': agent.knowledge.evolution_generation
            }
        return stats

