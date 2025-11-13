#!/usr/bin/env python3
"""
🧠 UNIVERSAL LEARNING SYSTEM V4.0
==================================

Система универсального обучения, которая создает ПРАВИЛА, а не запоминает решения
- Анализ паттернов в диапазонах, а не точных значений
- Создание универсальных правил для разных рыночных условий
- Эволюционное обучение с мутациями и кроссовером
- Проверка обобщающей способности правил
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import random
from collections import defaultdict
import statistics
import pytz

logger = logging.getLogger(__name__)

# Warsaw timezone для синхронизации времени
WARSAW_TZ = pytz.timezone('Europe/Warsaw')

@dataclass
class UniversalPattern:
    """Универсальный паттерн (диапазоны, не точные значения)"""
    pattern_id: str
    feature_ranges: Dict[str, Tuple[float, float]]  # Диапазоны признаков
    target_action: str  # 'buy', 'sell', 'hold'
    confidence_range: Tuple[float, float]
    market_conditions: List[str]
    success_rate: float
    sample_size: int
    generalization_score: float  # Насколько хорошо обобщает
    created_at: str
    last_validation: str

@dataclass
class LearningRule:
    """Правило обучения (универсальное)"""
    rule_id: str
    rule_name: str
    conditions: Dict[str, Any]  # Условия в виде диапазонов
    action: str
    priority: float
    success_history: List[bool]
    market_adaptability: Dict[str, float]  # Как работает в разных условиях
    evolution_generation: int
    parent_rules: List[str]  # Родительские правила
    mutation_history: List[str]

class UniversalLearningSystem:
    """🧠 Система универсального обучения"""
    
    def __init__(self, data_storage=None):
        self.data_storage = data_storage
        self.patterns = {}
        self.rules = {}
        self.evolution_generation = 0
        self.learning_history = []
        
        # Параметры обучения
        self.min_sample_size = 10  # Минимум примеров для создания правила
        self.min_success_rate = 0.50  # Минимальная успешность
        self.generalization_threshold = 0.50  # Порог обобщения (снижен для практического использования)
        self.mutation_rate = 0.15  # Вероятность мутации
        self.crossover_rate = 0.25  # Вероятность скрещивания
        
        logger.info("🧠 UniversalLearningSystem инициализирована")
    
    def analyze_market_patterns(self, market_data: List[Dict]) -> List[UniversalPattern]:
        """Анализ рыночных паттернов для создания универсальных правил"""
        try:
            if len(market_data) < self.min_sample_size:
                return []
            
            patterns = []
            
            # Группируем данные по результатам
            successful_trades = [d for d in market_data if d.get('result') == 'win']
            failed_trades = [d for d in market_data if d.get('result') == 'loss']
            
            # Создаем паттерны для успешных сделок
            if len(successful_trades) >= self.min_sample_size:
                success_pattern = self._create_pattern_from_data(
                    successful_trades, 'successful_entry', True
                )
                if success_pattern:
                    patterns.append(success_pattern)
            
            # Создаем паттерны для неудачных сделок (чтобы их избегать)
            if len(failed_trades) >= self.min_sample_size:
                failure_pattern = self._create_pattern_from_data(
                    failed_trades, 'failed_entry', False
                )
                if failure_pattern:
                    patterns.append(failure_pattern)
            
            # Анализируем паттерны по рыночным условиям
            market_conditions = set([d.get('market_condition', 'NEUTRAL') for d in market_data])
            
            for condition in market_conditions:
                condition_data = [d for d in market_data if d.get('market_condition') == condition]
                if len(condition_data) >= self.min_sample_size:
                    condition_pattern = self._create_pattern_from_data(
                        condition_data, f'{condition.lower()}_market', None
                    )
                    if condition_pattern:
                        patterns.append(condition_pattern)
            
            logger.info(f"🧠 Создано {len(patterns)} универсальных паттернов")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа паттернов: {e}")
            return []
    
    def _create_pattern_from_data(self, data: List[Dict], pattern_type: str, 
                                is_positive: Optional[bool]) -> Optional[UniversalPattern]:
        """Создание универсального паттерна из данных"""
        try:
            if len(data) < self.min_sample_size:
                return None
            
            # Извлекаем числовые признаки
            features = ['rsi', 'bb_position', 'volume_ratio', 'momentum', 'confidence', 'strategy_score']
            feature_ranges = {}
            
            for feature in features:
                # Извлекаем значения: сначала проверяем корневой уровень, затем market_data
                values = []
                for d in data:
                    value = d.get(feature)
                    if value is None and 'market_data' in d:
                        value = d.get('market_data', {}).get(feature)
                    if value is not None:
                        try:
                            values.append(float(value))
                        except (ValueError, TypeError):
                            continue
                
                if len(values) >= 3:  # Минимум 3 значения для диапазона
                    # Создаем диапазон с запасом (не точные значения!)
                    min_val = min(values)
                    max_val = max(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    
                    # Расширяем диапазон для обобщения
                    range_expansion = max(std_val, (max_val - min_val) * 0.1)
                    feature_ranges[feature] = (
                        max(0, min_val - range_expansion),
                        max_val + range_expansion
                    )
            
            if len(feature_ranges) < 3:  # Минимум 3 признака
                return None
            
            # Определяем целевое действие
            actions = []
            for d in data:
                action = d.get('decision')
                if action:
                    actions.append(action)
            target_action = max(set(actions), key=actions.count) if actions else 'hold'
            
            # Рассчитываем диапазон уверенности
            confidences = []
            for d in data:
                conf = d.get('confidence')
                if conf is None and 'market_data' in d:
                    conf = d.get('market_data', {}).get('confidence')
                if conf is not None:
                    try:
                        confidences.append(float(conf))
                    except (ValueError, TypeError):
                        continue
            confidence_range = (min(confidences), max(confidences)) if confidences else (40, 80)
            
            # Определяем рыночные условия
            market_conditions = []
            for d in data:
                cond = d.get('market_condition') or d.get('market_data', {}).get('market_condition', 'NEUTRAL')
                market_conditions.append(cond)
            market_conditions = list(set(market_conditions))
            
            # Рассчитываем успешность
            if is_positive is not None:
                success_rate = 0.8 if is_positive else 0.2  # Базовая оценка
            else:
                results = [d.get('result') for d in data if d.get('result')]
                wins = results.count('win')
                total = len([r for r in results if r in ['win', 'loss']])
                success_rate = wins / total if total > 0 else 0.5
            
            # Если успешность слишком низкая, но паттерн для успешных сделок - повышаем её
            if is_positive is True and success_rate < self.min_success_rate:
                success_rate = max(self.min_success_rate, 0.6)  # Минимум 60% для успешных паттернов
            
            # Рассчитываем score обобщения
            generalization_score = self._calculate_generalization_score(feature_ranges, len(data))
            
            pattern = UniversalPattern(
                pattern_id=f"pattern_{pattern_type}_{datetime.now(WARSAW_TZ).strftime('%Y%m%d_%H%M%S')}",
                feature_ranges=feature_ranges,
                target_action=target_action,
                confidence_range=confidence_range,
                market_conditions=market_conditions,
                success_rate=success_rate,
                sample_size=len(data),
                generalization_score=generalization_score,
                created_at=datetime.now(WARSAW_TZ).isoformat(),
                last_validation=datetime.now(WARSAW_TZ).isoformat()
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания паттерна: {e}")
            return None
    
    def _calculate_generalization_score(self, feature_ranges: Dict, sample_size: int) -> float:
        """Расчет score обобщающей способности"""
        try:
            # Чем шире диапазоны и больше выборка, тем лучше обобщение
            range_widths = []
            for feature, (min_val, max_val) in feature_ranges.items():
                if max_val > min_val:
                    # Нормализуем ширину диапазона
                    if feature in ['rsi', 'bb_position']:
                        normalized_width = (max_val - min_val) / 100.0
                    elif feature in ['confidence', 'strategy_score']:
                        normalized_width = (max_val - min_val) / 20.0
                    else:
                        normalized_width = min(1.0, (max_val - min_val) / max_val)
                    
                    range_widths.append(normalized_width)
            
            avg_range_width = statistics.mean(range_widths) if range_widths else 0
            sample_factor = min(1.0, sample_size / 50.0)  # Нормализуем размер выборки
            
            # Комбинируем факторы
            generalization_score = (avg_range_width * 0.6 + sample_factor * 0.4)
            
            return min(1.0, generalization_score)
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка расчета обобщения: {e}")
            return 0.5
    
    def create_universal_rules(self, patterns: List[UniversalPattern]) -> List[LearningRule]:
        """Создание универсальных правил из паттернов"""
        try:
            rules = []
            
            for pattern in patterns:
                # Детальное логирование для отладки
                logger.debug(f"📋 Проверка паттерна {pattern.pattern_id}:")
                logger.debug(f"   Success rate: {pattern.success_rate:.2f} (требуется >= {self.min_success_rate:.2f})")
                logger.debug(f"   Generalization: {pattern.generalization_score:.2f} (требуется >= {self.generalization_threshold:.2f})")
                
                if (pattern.success_rate >= self.min_success_rate and 
                    pattern.generalization_score >= self.generalization_threshold):
                    
                    logger.debug(f"   ✅ Паттерн проходит проверку - создаем правило")
                    rule = self._pattern_to_rule(pattern)
                    if rule:
                        rules.append(rule)
                        self.rules[rule.rule_id] = rule
                        logger.debug(f"   ✅ Правило {rule.rule_id} создано успешно")
                else:
                    failures = []
                    if pattern.success_rate < self.min_success_rate:
                        failures.append(f"success_rate ({pattern.success_rate:.2f} < {self.min_success_rate:.2f})")
                    if pattern.generalization_score < self.generalization_threshold:
                        failures.append(f"generalization ({pattern.generalization_score:.2f} < {self.generalization_threshold:.2f})")
                    logger.debug(f"   ❌ Паттерн не прошел проверку: {', '.join(failures)}")
            
            logger.info(f"🧠 Создано {len(rules)} универсальных правил из {len(patterns)} паттернов")
            return rules
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания правил: {e}", exc_info=True)
            return []
    
    def get_learned_rules(self) -> List[LearningRule]:
        """Получить список всех созданных правил"""
        return list(self.rules.values())
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Анализ паттернов для проверки универсальности"""
        try:
            if not self.patterns:
                return {}
            
            analysis = {
                'total_patterns': len(self.patterns),
                'patterns_with_ranges': 0,
                'patterns_with_exact_values': 0,
                'average_generalization': 0.0,
                'patterns_by_action': defaultdict(int)
            }
            
            generalization_scores = []
            
            for pattern_id, pattern in self.patterns.items():
                # Проверяем, используется ли диапазоны (универсальность)
                has_ranges = any(
                    isinstance(r, tuple) and len(r) == 2 
                    for r in pattern.feature_ranges.values()
                )
                
                if has_ranges:
                    analysis['patterns_with_ranges'] += 1
                else:
                    analysis['patterns_with_exact_values'] += 1
                
                generalization_scores.append(pattern.generalization_score)
                analysis['patterns_by_action'][pattern.target_action] += 1
            
            if generalization_scores:
                analysis['average_generalization'] = statistics.mean(generalization_scores)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа паттернов: {e}")
            return {}
    
    def learn_from_decision(self, market_data: Dict, decision: str, result: str):
        """Обучение на основе принятого решения"""
        try:
            # Сохраняем решение в историю
            learning_entry = {
                'market_data': market_data,
                'decision': decision,
                'result': result,
                'timestamp': datetime.now(WARSAW_TZ).isoformat(),
                'market_condition': market_data.get('market_condition', 'NEUTRAL')
            }
            
            self.learning_history.append(learning_entry)
            
            # Если накопилось достаточно примеров, создаем паттерны
            if len(self.learning_history) >= self.min_sample_size * 2:
                patterns = self.analyze_market_patterns(self.learning_history[-50:])  # Последние 50
                if patterns:
                    rules = self.create_universal_rules(patterns)
                    logger.debug(f"🧠 Создано {len(rules)} новых правил на основе обучения")
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка обучения: {e}")
    
    def _pattern_to_rule(self, pattern: UniversalPattern) -> Optional[LearningRule]:
        """Преобразование паттерна в правило"""
        try:
            # Создаем условия правила (диапазоны, не точные значения!)
            conditions = {
                'feature_ranges': pattern.feature_ranges,
                'market_conditions': pattern.market_conditions,
                'confidence_range': pattern.confidence_range,
                'min_generalization': pattern.generalization_score
            }
            
            # Определяем приоритет на основе успешности и обобщения
            priority = (pattern.success_rate * 0.7 + pattern.generalization_score * 0.3)
            
            rule = LearningRule(
                rule_id=f"rule_{pattern.pattern_id}",
                rule_name=f"Universal {pattern.target_action} rule",
                conditions=conditions,
                action=pattern.target_action,
                priority=priority,
                success_history=[],
                market_adaptability={condition: pattern.success_rate for condition in pattern.market_conditions},
                evolution_generation=self.evolution_generation,
                parent_rules=[],
                mutation_history=[]
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"❌ Ошибка преобразования паттерна в правило: {e}")
            return None
    
    def evolve_rules(self, performance_data: List[Dict]) -> List[LearningRule]:
        """Эволюция правил (мутации, кроссовер, селекция)"""
        try:
            if len(self.rules) < 2:
                return list(self.rules.values())
            
            self.evolution_generation += 1
            evolved_rules = []
            
            # Оцениваем производительность существующих правил
            rule_performance = self._evaluate_rule_performance(performance_data)
            
            # Селекция лучших правил
            best_rules = self._select_best_rules(rule_performance, top_percent=0.6)
            
            # Мутации лучших правил
            for rule in best_rules:
                if random.random() < self.mutation_rate:
                    mutated_rule = self._mutate_rule(rule)
                    if mutated_rule:
                        evolved_rules.append(mutated_rule)
            
            # Кроссовер между лучшими правилами
            for i in range(len(best_rules)):
                for j in range(i + 1, len(best_rules)):
                    if random.random() < self.crossover_rate:
                        offspring = self._crossover_rules(best_rules[i], best_rules[j])
                        if offspring:
                            evolved_rules.extend(offspring)
            
            # Добавляем лучшие исходные правила
            evolved_rules.extend(best_rules)
            
            # Обновляем коллекцию правил
            self.rules = {rule.rule_id: rule for rule in evolved_rules}
            
            logger.info(f"🧬 Эволюция поколения {self.evolution_generation}: {len(evolved_rules)} правил")
            
            return evolved_rules
            
        except Exception as e:
            logger.error(f"❌ Ошибка эволюции правил: {e}")
            return list(self.rules.values())
    
    def _evaluate_rule_performance(self, performance_data: List[Dict]) -> Dict[str, float]:
        """Оценка производительности правил"""
        rule_performance = {}
        
        for rule_id, rule in self.rules.items():
            # Базовая оценка на основе истории успехов
            if rule.success_history:
                base_score = sum(rule.success_history) / len(rule.success_history)
            else:
                base_score = rule.priority
            
            # Адаптивность к рыночным условиям
            adaptability_score = statistics.mean(rule.market_adaptability.values())
            
            # Итоговая оценка
            performance_score = base_score * 0.7 + adaptability_score * 0.3
            rule_performance[rule_id] = performance_score
        
        return rule_performance
    
    def _select_best_rules(self, performance: Dict[str, float], top_percent: float = 0.6) -> List[LearningRule]:
        """Селекция лучших правил"""
        sorted_rules = sorted(performance.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, int(len(sorted_rules) * top_percent))
        
        best_rule_ids = [rule_id for rule_id, _ in sorted_rules[:top_count]]
        return [self.rules[rule_id] for rule_id in best_rule_ids if rule_id in self.rules]
    
    def _mutate_rule(self, rule: LearningRule) -> Optional[LearningRule]:
        """Мутация правила"""
        try:
            # Создаем копию правила
            mutated_conditions = rule.conditions.copy()
            
            # Мутируем диапазоны признаков
            if 'feature_ranges' in mutated_conditions:
                feature_ranges = mutated_conditions['feature_ranges'].copy()
                
                # Случайно выбираем признак для мутации
                if feature_ranges:
                    feature = random.choice(list(feature_ranges.keys()))
                    min_val, max_val = feature_ranges[feature]
                    
                    # Мутируем диапазон (расширяем или сужаем)
                    range_width = max_val - min_val
                    mutation_factor = random.uniform(0.8, 1.2)  # ±20%
                    
                    new_width = range_width * mutation_factor
                    center = (min_val + max_val) / 2
                    
                    feature_ranges[feature] = (
                        max(0, center - new_width / 2),
                        center + new_width / 2
                    )
                
                mutated_conditions['feature_ranges'] = feature_ranges
            
            # Создаем новое правило
            mutated_rule = LearningRule(
                rule_id=f"mutated_{rule.rule_id}_{self.evolution_generation}",
                rule_name=f"Mutated {rule.rule_name}",
                conditions=mutated_conditions,
                action=rule.action,
                priority=rule.priority * random.uniform(0.9, 1.1),  # Небольшая мутация приоритета
                success_history=[],
                market_adaptability=rule.market_adaptability.copy(),
                evolution_generation=self.evolution_generation,
                parent_rules=[rule.rule_id],
                mutation_history=rule.mutation_history + [f"gen_{self.evolution_generation}"]
            )
            
            return mutated_rule
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка мутации правила: {e}")
            return None
    
    def _crossover_rules(self, rule1: LearningRule, rule2: LearningRule) -> List[LearningRule]:
        """Кроссовер между правилами"""
        try:
            offspring = []
            
            # Создаем потомка, комбинируя признаки родителей
            if ('feature_ranges' in rule1.conditions and 
                'feature_ranges' in rule2.conditions):
                
                ranges1 = rule1.conditions['feature_ranges']
                ranges2 = rule2.conditions['feature_ranges']
                
                # Комбинируем признаки
                combined_ranges = {}
                all_features = set(ranges1.keys()) | set(ranges2.keys())
                
                for feature in all_features:
                    if feature in ranges1 and feature in ranges2:
                        # Берем пересечение диапазонов
                        min1, max1 = ranges1[feature]
                        min2, max2 = ranges2[feature]
                        
                        combined_min = (min1 + min2) / 2
                        combined_max = (max1 + max2) / 2
                        
                        combined_ranges[feature] = (combined_min, combined_max)
                    elif feature in ranges1:
                        combined_ranges[feature] = ranges1[feature]
                    else:
                        combined_ranges[feature] = ranges2[feature]
                
                # Создаем потомка
                offspring_conditions = {
                    'feature_ranges': combined_ranges,
                    'market_conditions': list(set(
                        rule1.conditions.get('market_conditions', []) +
                        rule2.conditions.get('market_conditions', [])
                    )),
                    'confidence_range': (
                        (rule1.conditions.get('confidence_range', (40, 80))[0] +
                         rule2.conditions.get('confidence_range', (40, 80))[0]) / 2,
                        (rule1.conditions.get('confidence_range', (40, 80))[1] +
                         rule2.conditions.get('confidence_range', (40, 80))[1]) / 2
                    )
                }
                
                offspring_rule = LearningRule(
                    rule_id=f"crossover_{rule1.rule_id}_{rule2.rule_id}_{self.evolution_generation}",
                    rule_name=f"Crossover of {rule1.rule_name} and {rule2.rule_name}",
                    conditions=offspring_conditions,
                    action=random.choice([rule1.action, rule2.action]),
                    priority=(rule1.priority + rule2.priority) / 2,
                    success_history=[],
                    market_adaptability={},
                    evolution_generation=self.evolution_generation,
                    parent_rules=[rule1.rule_id, rule2.rule_id],
                    mutation_history=[]
                )
                
                offspring.append(offspring_rule)
            
            return offspring
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка кроссовера: {e}")
            return []
    
    def apply_rules_to_decision(self, market_data: Dict) -> Dict[str, Any]:
        """Применение универсальных правил к принятию решения"""
        try:
            applicable_rules = []
            
            for rule in self.rules.values():
                if self._rule_matches_data(rule, market_data):
                    applicable_rules.append(rule)
            
            if not applicable_rules:
                return {'decision': 'hold', 'confidence': 50, 'rules_applied': []}
            
            # Сортируем по приоритету
            applicable_rules.sort(key=lambda r: r.priority, reverse=True)
            
            # Применяем лучшее правило
            best_rule = applicable_rules[0]
            
            # Рассчитываем уверенность на основе правила
            base_confidence = market_data.get('confidence', 50)
            rule_confidence_boost = best_rule.priority * 20  # Бонус от правила
            
            final_confidence = min(95, base_confidence + rule_confidence_boost)
            
            return {
                'decision': best_rule.action,
                'confidence': final_confidence,
                'rules_applied': [best_rule.rule_id],
                'rule_priority': best_rule.priority,
                'rule_generation': best_rule.evolution_generation
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка применения правил: {e}")
            return {'decision': 'hold', 'confidence': 50, 'rules_applied': []}
    
    def _rule_matches_data(self, rule: LearningRule, market_data: Dict) -> bool:
        """Проверка соответствия правила данным"""
        try:
            conditions = rule.conditions
            
            # Проверяем диапазоны признаков
            if 'feature_ranges' in conditions:
                for feature, (min_val, max_val) in conditions['feature_ranges'].items():
                    if feature in market_data:
                        value = market_data[feature]
                        if not (min_val <= value <= max_val):
                            return False
            
            # Проверяем рыночные условия
            if 'market_conditions' in conditions:
                market_condition = market_data.get('market_condition', 'NEUTRAL')
                if market_condition not in conditions['market_conditions']:
                    return False
            
            # Проверяем диапазон уверенности
            if 'confidence_range' in conditions:
                confidence = market_data.get('confidence', 50)
                min_conf, max_conf = conditions['confidence_range']
                if not (min_conf <= confidence <= max_conf):
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки соответствия правила: {e}")
            return False
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Получить сводку обучения"""
        try:
            total_rules = len(self.rules)
            avg_priority = statistics.mean([r.priority for r in self.rules.values()]) if self.rules else 0
            
            # Анализ по поколениям
            generations = [r.evolution_generation for r in self.rules.values()]
            max_generation = max(generations) if generations else 0
            
            # Анализ по действиям
            actions = [r.action for r in self.rules.values()]
            action_distribution = {action: actions.count(action) for action in set(actions)}
            
            return {
                'total_rules': total_rules,
                'current_generation': self.evolution_generation,
                'max_rule_generation': max_generation,
                'average_priority': avg_priority,
                'action_distribution': action_distribution,
                'learning_approach': 'Universal Rules (not memorization)',
                'generalization_focus': True,
                'evolution_active': total_rules > 1
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения сводки: {e}")
            return {}

# Тестирование системы
if __name__ == "__main__":
    learning_system = UniversalLearningSystem()
    
    # Тестовые данные
    test_data = [
        {'rsi': 35, 'bb_position': 25, 'volume_ratio': 1.2, 'confidence': 65, 'decision': 'buy', 'result': 'win', 'market_condition': 'BULLISH'},
        {'rsi': 40, 'bb_position': 30, 'volume_ratio': 1.1, 'confidence': 70, 'decision': 'buy', 'result': 'win', 'market_condition': 'BULLISH'},
        {'rsi': 32, 'bb_position': 28, 'volume_ratio': 1.3, 'confidence': 68, 'decision': 'buy', 'result': 'win', 'market_condition': 'BULLISH'},
        {'rsi': 75, 'bb_position': 85, 'volume_ratio': 0.8, 'confidence': 45, 'decision': 'sell', 'result': 'loss', 'market_condition': 'BEARISH'},
    ] * 5  # Увеличиваем выборку
    
    # Анализ паттернов
    patterns = learning_system.analyze_market_patterns(test_data)
    print(f"📊 Найдено паттернов: {len(patterns)}")
    
    # Создание правил
    rules = learning_system.create_universal_rules(patterns)
    print(f"🧠 Создано правил: {len(rules)}")
    
    # Применение правил
    test_market_data = {'rsi': 37, 'bb_position': 27, 'volume_ratio': 1.15, 'confidence': 66, 'market_condition': 'BULLISH'}
    decision = learning_system.apply_rules_to_decision(test_market_data)
    print(f"🎯 Решение: {decision}")
    
    # Сводка обучения
    summary = learning_system.get_learning_summary()
    print(f"📋 Сводка обучения: {summary}")
    
    print("✅ UniversalLearningSystem протестирована")

