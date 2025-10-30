#!/usr/bin/env python3
"""
📁 ПРОВЕРКА СИСТЕМЫ ХРАНИЛИЩА И АВТОНОМНОГО ОБУЧЕНИЯ
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_directory_structure():
    """Проверка структуры папок"""
    print("📁 ПРОВЕРКА СТРУКТУРЫ ПАПОК")
    print("="*70)
    print()
    
    required_dirs = {
        'data/': 'База данных и торговые данные',
        'data/models/': 'ML модели для сохранения',
        'data/cache/': 'Кеш данных',
        'data/storage/': 'Исторические данные',
        'logs/': 'Логи системы',
        'logs/trading/': 'Логи торговли',
        'logs/system/': 'Логи системных событий',
        'logs/ml/': 'Логи ML/AI'
    }
    
    base_path = Path('.')
    status = []
    
    for dir_name, description in required_dirs.items():
        dir_path = base_path / dir_name
        exists = dir_path.exists()
        status.append((dir_name, description, exists))
        
        if exists:
            # Проверяем размер
            size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            files = len(list(dir_path.rglob('*')))
            print(f"✅ {dir_name:20s} - {description}")
            print(f"   Размер: {size/1024:.1f} KB, Файлов: {files}")
        else:
            print(f"❌ {dir_name:20s} - {description} - ОТСУТСТВУЕТ")
            # Создаем
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Создана!")
            except Exception as e:
                print(f"   ❌ Ошибка создания: {e}")
        print()
    
    print("="*70)
    existing = sum(1 for _, _, exists in status if exists)
    print(f"📊 Итого: {existing}/{len(required_dirs)} папок существуют")
    print()

def check_database():
    """Проверка базы данных"""
    print("💾 ПРОВЕРКА БАЗЫ ДАННЫХ")
    print("="*70)
    print()
    
    try:
        from data_storage_system import DataStorageSystem
        storage = DataStorageSystem()
        
        db_path = Path(storage.db_path)
        if db_path.exists():
            print(f"✅ База данных: {db_path}")
            print(f"   Размер: {db_path.stat().st_size / 1024:.1f} KB")
            
            import sqlite3
            conn = sqlite3.connect(storage.db_path)
            cursor = conn.cursor()
            
            # Проверяем таблицы
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = [
                'market_data', 'trade_decisions', 'universal_rules',
                'learning_patterns', 'performance_stats'
            ]
            
            print(f"\n📋 Таблицы ({len(tables)}):")
            for table in required_tables:
                if table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    print(f"  ✅ {table:20s} - {count} записей")
                else:
                    print(f"  ❌ {table:20s} - отсутствует")
            
            conn.close()
        else:
            print(f"⚠️ База данных не существует: {db_path}")
            print("   Будет создана при первом использовании")
            
    except Exception as e:
        print(f"❌ Ошибка проверки БД: {e}")

def check_universal_learning():
    """Проверка универсального обучения"""
    print("\n🧠 ПРОВЕРКА УНИВЕРСАЛЬНОГО ОБУЧЕНИЯ")
    print("="*70)
    print()
    
    code_file = Path('universal_learning_system.py')
    if code_file.exists():
        content = code_file.read_text(encoding='utf-8')
        
        # Проверяем что используются диапазоны
        checks = {
            'Использует диапазоны': 'range' in content.lower() or 'min(' in content or 'max(' in content,
            'Создает паттерны': '_create_pattern_from_data' in content,
            'Диапазоны уверенности': 'confidence_range' in content,
            'Универсальные правила': 'UniversalRule' in content,
            'НЕ использует точные значения': 'exact_value' not in content.lower() and '===' not in content
        }
        
        for check_name, passed in checks.items():
            icon = "✅" if passed else "❌"
            print(f"{icon} {check_name}")
        
        print()
        
        # Проверяем примеры использования диапазонов
        if '(min(' in content and ', max(' in content:
            print("✅ Найдено использование min/max для диапазонов (универсально)")
        else:
            print("⚠️ min/max диапазоны не найдены явно")
            
    else:
        print("❌ Файл universal_learning_system.py не найден")

def check_coin_selection():
    """Проверка выбора 145 монет"""
    print("\n🎯 ПРОВЕРКА ВЫБОРА 145 МОНЕТ")
    print("="*70)
    print()
    
    code_file = Path('smart_coin_selector.py')
    if code_file.exists():
        content = code_file.read_text(encoding='utf-8')
        
        # Проверяем что для NEUTRAL/NORMAL настроено 145
        if "145" in content:
            print("✅ Значение 145 найдено в коде")
            
            # Ищем контекст
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '145' in line and ('normal' in line.lower() or 'neutral' in line.lower()):
                    print(f"  Найдено: {line.strip()[:80]}")
                    break
        else:
            print("❌ 145 не найдено в коде")
        
        # Проверяем метод
        if 'get_smart_symbols' in content:
            print("✅ Метод get_smart_symbols существует")
        else:
            print("❌ Метод get_smart_symbols не найден")
            
        # Проверяем в логах
        log_file = Path('super_bot_v4_mtf.log')
        if log_file.exists():
            log_content = log_file.read_text(encoding='utf-8')
            if '145' in log_content or 'символ' in log_content.lower():
                print("✅ В логах есть упоминания выбора монет")
            else:
                print("⚠️ В логах нет упоминаний 145 (может быть рынок не NEUTRAL)")

def check_auto_update():
    """Проверка автоматического обновления"""
    print("\n🔄 ПРОВЕРКА АВТОМАТИЧЕСКОГО ОБНОВЛЕНИЯ")
    print("="*70)
    print()
    
    # Проверяем что данные сохраняются автоматически
    checks = {
        'Автосохранение рыночных данных': 'store_market_data' in open('super_bot_v4_mtf.py').read(),
        'Автосохранение решений': 'store_trade_decision' in open('super_bot_v4_mtf.py').read() or 'learn_from_decision' in open('super_bot_v4_mtf.py').read(),
        'Автоматическое обучение': 'learn_from_decision' in open('super_bot_v4_mtf.py').read() or 'analyze_market_patterns' in open('super_bot_v4_mtf.py').read(),
        'Автоочистка старых данных': 'cleanup_old_data' in open('data_storage_system.py').read(),
        'Обновление правил': 'last_updated' in open('data_storage_system.py').read()
    }
    
    for check_name, passed in checks.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {check_name}")

def main():
    """Главная функция"""
    print("="*70)
    print("📊 ПОЛНАЯ ПРОВЕРКА СИСТЕМЫ ХРАНИЛИЩА И АВТОНОМНОГО ОБУЧЕНИЯ")
    print("="*70)
    print()
    
    check_directory_structure()
    check_database()
    check_universal_learning()
    check_coin_selection()
    check_auto_update()
    
    print()
    print("="*70)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("="*70)

if __name__ == "__main__":
    main()


