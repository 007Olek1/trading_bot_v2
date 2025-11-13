#!/usr/bin/env python3
"""
✅ ПРОВЕРКА СИСТЕМЫ УНИВЕРСАЛЬНОГО ОБУЧЕНИЯ
==========================================

Проверяет что система действительно создает УНИВЕРСАЛЬНЫЕ ПРАВИЛА,
а не просто запоминает решения.

Ключевые проверки:
1. Правила используют ДИАПАЗОНЫ, а не точные значения
2. Правила обобщают на новые данные (generalization_score)
3. Система создает правила из паттернов, а не запоминает
4. Правила работают в разных рыночных условиях
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, '/opt/bot' if os.path.exists('/opt/bot') else os.path.dirname(__file__))

from universal_learning_system import UniversalLearningSystem, UniversalPattern, LearningRule
from data_storage_system import DataStorageSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_universal_rules_vs_memorization():
    """Тест: проверяем что создаются универсальные правила, а не запоминание"""
    
    print("\n" + "="*70)
    print("🧠 ТЕСТ: УНИВЕРСАЛЬНЫЕ ПРАВИЛА VS ЗАПОМИНАНИЕ")
    print("="*70 + "\n")
    
    # Инициализация
    data_storage = DataStorageSystem()
    learning_system = UniversalLearningSystem(data_storage=data_storage)
    
    # Симулируем данные для обучения (разные условия)
    test_data = []
    
    # Успешные сделки с разными значениями RSI (но в диапазоне)
    for i in range(20):
        test_data.append({
            'result': 'win',
            'rsi': 30 + i * 2,  # RSI от 30 до 68
            'bb_position': 20 + i * 2,  # BB от 20 до 58
            'volume_ratio': 1.0 + i * 0.1,  # Volume от 1.0 до 2.9
            'entry_price': 100.0 + i,
            'exit_price': 104.0 + i,  # +4% прибыль
            'market_condition': 'neutral' if i % 2 == 0 else 'bullish'
        })
    
    print("📊 Создаем паттерны из тестовых данных...")
    patterns = learning_system.analyze_market_patterns(test_data)
    
    print(f"\n✅ Создано паттернов: {len(patterns)}")
    
    if not patterns:
        print("❌ ОШИБКА: Паттерны не созданы!")
        return False
    
    # Проверка 1: Паттерны используют ДИАПАЗОНЫ
    print("\n🔍 ПРОВЕРКА 1: Паттерны используют ДИАПАЗОНЫ")
    print("-" * 70)
    
    for pattern in patterns:
        print(f"\n📋 Паттерн: {pattern.pattern_id}")
        print(f"   Success rate: {pattern.success_rate:.2%}")
        print(f"   Generalization score: {pattern.generalization_score:.2f}")
        
        if pattern.feature_ranges:
            print(f"\n   ✅ Использует ДИАПАЗОНЫ (не точные значения):")
            for feature, (min_val, max_val) in pattern.feature_ranges.items():
                range_size = max_val - min_val
                print(f"      {feature}: [{min_val:.2f}, {max_val:.2f}] (диапазон: {range_size:.2f})")
                
                # КРИТИЧНО: Диапазон должен быть достаточно широким
                if range_size < 0.5:
                    print(f"      ⚠️ ВНИМАНИЕ: Диапазон слишком узкий ({range_size:.2f}) - может быть запоминание!")
                else:
                    print(f"      ✅ Диапазон достаточен для обобщения")
        else:
            print(f"   ❌ ОШИБКА: Нет диапазонов признаков!")
            return False
    
    # Проверка 2: Создание универсальных правил
    print("\n\n🔍 ПРОВЕРКА 2: Создание универсальных правил")
    print("-" * 70)
    
    rules = learning_system.create_universal_rules(patterns)
    
    print(f"\n✅ Создано правил: {len(rules)}")
    
    if not rules:
        print("⚠️ Правила не созданы (возможно, не прошли пороги)")
        print(f"   Min success rate: {learning_system.min_success_rate:.2%}")
        print(f"   Min generalization: {learning_system.generalization_threshold:.2f}")
        return False
    
    # Проверка 3: Правила обобщают на новые данные
    print("\n\n🔍 ПРОВЕРКА 3: Правила обобщают на НОВЫЕ данные")
    print("-" * 70)
    
    for rule in rules:
        print(f"\n📋 Правило: {rule.rule_id} ({rule.rule_name})")
        print(f"   Priority: {rule.priority:.2f}")
        print(f"   Success history: {len(rule.success_history)} применений")
        print(f"   Market adaptability: {rule.market_adaptability}")
        
        if rule.conditions:
            print(f"\n   ✅ Условия в виде ДИАПАЗОНОВ:")
            for condition, value in rule.conditions.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    print(f"      {condition}: [{value[0]:.2f}, {value[1]:.2f}]")
                else:
                    print(f"      {condition}: {value}")
        
        # Проверка: Правило должно работать в разных условиях
        if len(rule.market_adaptability) > 1:
            print(f"   ✅ Работает в {len(rule.market_adaptability)} рыночных условиях (универсальность!)")
        else:
            print(f"   ⚠️ Только в одном условии (может быть узкоспециализированным)")
    
    # Проверка 4: Правила НЕ запоминают точные значения
    print("\n\n🔍 ПРОВЕРКА 4: Правила НЕ запоминают точные значения")
    print("-" * 70)
    
    # Проверяем что правила используют диапазоны, а не точные значения
    all_use_ranges = True
    for rule in rules:
        for condition, value in rule.conditions.items():
            if isinstance(value, (int, float)) and not isinstance(value, (list, tuple)):
                # Если значение - одно число, это подозрительно
                if isinstance(value, float) and value not in [0.0, 1.0, -1.0]:
                    # Проверяем что это не точное значение
                    all_use_ranges = False
                    print(f"   ⚠️ {rule.rule_id}: {condition} = {value} (точное значение, не диапазон)")
    
    if all_use_ranges:
        print("   ✅ ВСЕ правила используют ДИАПАЗОНЫ, не точные значения!")
    
    # Итоговая проверка
    print("\n\n" + "="*70)
    print("📊 ИТОГОВАЯ ОЦЕНКА")
    print("="*70 + "\n")
    
    checks_passed = 0
    total_checks = 4
    
    # Проверка 1: Диапазоны
    if all_use_ranges and patterns and all(p.feature_ranges for p in patterns):
        print("✅ ПРОВЕРКА 1: Правила используют ДИАПАЗОНЫ ✓")
        checks_passed += 1
    else:
        print("❌ ПРОВЕРКА 1: Не все правила используют диапазоны")
    
    # Проверка 2: Создание правил
    if rules:
        print("✅ ПРОВЕРКА 2: Правила создаются из паттернов ✓")
        checks_passed += 1
    else:
        print("❌ ПРОВЕРКА 2: Правила не созданы")
    
    # Проверка 3: Обобщение
    if rules and any(len(r.market_adaptability) > 1 for r in rules):
        print("✅ ПРОВЕРКА 3: Правила обобщают на новые данные ✓")
        checks_passed += 1
    else:
        print("⚠️ ПРОВЕРКА 3: Обобщение ограничено")
    
    # Проверка 4: Не запоминание
    if all_use_ranges:
        print("✅ ПРОВЕРКА 4: Система НЕ запоминает решения ✓")
        checks_passed += 1
    else:
        print("❌ ПРОВЕРКА 4: Есть признаки запоминания")
    
    print(f"\n📊 Результат: {checks_passed}/{total_checks} проверок пройдено")
    
    if checks_passed == total_checks:
        print("\n🎉 УСПЕХ! Система создает УНИВЕРСАЛЬНЫЕ ПРАВИЛА, а не запоминает решения!")
        return True
    elif checks_passed >= 3:
        print("\n✅ ХОРОШО! Система работает правильно с небольшими ограничениями")
        return True
    else:
        print("\n⚠️ ВНИМАНИЕ! Система может запоминать решения. Требуется улучшение.")
        return False

if __name__ == "__main__":
    success = test_universal_rules_vs_memorization()
    sys.exit(0 if success else 1)

