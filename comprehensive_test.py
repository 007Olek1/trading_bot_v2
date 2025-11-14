"""
🔬 КОМПЛЕХСНОЕ ТЕСТИРОВАНИЕ DISCO57 BOT
Экспертная проверка всех компонентов системы
"""

import sys
import os
from pathlib import Path

# Добавляем путь к боту
sys.path.insert(0, str(Path(__file__).parent / "bybit_futures_bot"))

def test_imports():
    """Тест 1: Проверка импортов"""
    print("\n" + "="*70)
    print("ТЕСТ 1: ПРОВЕРКА ИМПОРТОВ")
    print("="*70)
    
    try:
        import config
        print("✅ config импортирован")
        
        from utils import round_quantity, calculate_position_size
        print("✅ utils импортирован")
        
        from indicators import MarketIndicators
        print("✅ indicators импортирован")
        
        from main import Disco57Bot
        print("✅ main импортирован")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

def test_config():
    """Тест 2: Проверка конфигурации"""
    print("\n" + "="*70)
    print("ТЕСТ 2: ПРОВЕРКА КОНФИГУРАЦИИ")
    print("="*70)
    
    try:
        import config
        
        checks = [
            ("BYBIT_API_KEY", bool(config.BYBIT_API_KEY)),
            ("BYBIT_API_SECRET", bool(config.BYBIT_API_SECRET)),
            ("TELEGRAM_TOKEN", bool(config.TELEGRAM_TOKEN)),
            ("TELEGRAM_CHAT_ID", bool(config.TELEGRAM_CHAT_ID)),
            ("POSITION_SIZE_USD", config.POSITION_SIZE_USD > 0),
            ("LEVERAGE", config.LEVERAGE > 0),
            ("MAX_CONCURRENT_POSITIONS", config.MAX_CONCURRENT_POSITIONS > 0),
            ("WATCHLIST", len(config.WATCHLIST) > 0),
        ]
        
        all_ok = True
        for name, check in checks:
            status = "✅" if check else "❌"
            print(f"{status} {name}: {check}")
            if not check:
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"❌ Ошибка проверки конфигурации: {e}")
        return False

def test_rounding():
    """Тест 3: Проверка округления количества"""
    print("\n" + "="*70)
    print("ТЕСТ 3: ПРОВЕРКА ОКРУГЛЕНИЯ КОЛИЧЕСТВА")
    print("="*70)
    
    try:
        from utils import round_quantity
        
        test_cases = [
            (1.0090000000000001, 0.001, 1.009),
            (19.881, 0.01, 19.88),
            (0.123456789, 0.001, 0.123),
            (100.5, 1.0, 100.0),  # round(100.5) = 100 в Python (banker's rounding)
            (101.5, 1.0, 102.0),
            (0.0001, 0.0001, 0.0001),
        ]
        
        all_ok = True
        for qty, step, expected in test_cases:
            result = round_quantity(qty, step)
            # Форматируем для сравнения
            result_str = f"{result:.10f}".rstrip('0').rstrip('.')
            expected_str = f"{expected:.10f}".rstrip('0').rstrip('.')
            
            passed = abs(result - expected) < 0.0001
            status = "✅" if passed else "❌"
            print(f"{status} qty={qty}, step={step} -> {result} (ожидалось {expected})")
            if not passed:
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"❌ Ошибка теста округления: {e}")
        return False

def test_bot_initialization():
    """Тест 4: Инициализация бота"""
    print("\n" + "="*70)
    print("ТЕСТ 4: ИНИЦИАЛИЗАЦИЯ БОТА")
    print("="*70)
    
    try:
        from main import Disco57Bot
        
        bot = Disco57Bot()
        print("✅ Бот инициализирован")
        
        # Проверка атрибутов
        checks = [
            ("client", hasattr(bot, 'client')),
            ("indicators_calculator", hasattr(bot, 'indicators_calculator')),
            ("active", hasattr(bot, 'active')),
            ("cycle_count", hasattr(bot, 'cycle_count')),
        ]
        
        all_ok = True
        for name, check in checks:
            status = "✅" if check else "❌"
            print(f"{status} {name}: {check}")
            if not check:
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"❌ Ошибка инициализации бота: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_connection():
    """Тест 5: Проверка подключения к API"""
    print("\n" + "="*70)
    print("ТЕСТ 5: ПРОВЕРКА ПОДКЛЮЧЕНИЯ К BYBIT API")
    print("="*70)
    
    try:
        from main import Disco57Bot
        
        bot = Disco57Bot()
        
        # Проверка баланса
        balance = bot.get_balance()
        print(f"✅ Баланс получен: ${balance:.2f}")
        
        if balance > 0:
            print("✅ Баланс > 0 - API работает")
            return True
        else:
            print("⚠️ Баланс = 0 (возможно нет средств или проблема с API)")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения к API: {e}")
        return False

def test_telegram():
    """Тест 6: Проверка Telegram"""
    print("\n" + "="*70)
    print("ТЕСТ 6: ПРОВЕРКА TELEGRAM")
    print("="*70)
    
    try:
        import config
        
        if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
            print("⚠️ Telegram не настроен")
            return False
        
        print(f"✅ TELEGRAM_TOKEN: {'установлен' if config.TELEGRAM_TOKEN else 'не установлен'}")
        print(f"✅ TELEGRAM_CHAT_ID: {config.TELEGRAM_CHAT_ID}")
        
        # Проверка формата токена
        if config.TELEGRAM_TOKEN and ':' in config.TELEGRAM_TOKEN:
            print("✅ Формат токена корректный")
            return True
        else:
            print("❌ Формат токена некорректный")
            return False
    except Exception as e:
        print(f"❌ Ошибка проверки Telegram: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("\n" + "="*70)
    print("🔬 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ DISCO57 BOT")
    print("   Экспертная проверка от команды разработчиков")
    print("="*70)
    
    tests = [
        ("Импорты", test_imports),
        ("Конфигурация", test_config),
        ("Округление", test_rounding),
        ("Инициализация бота", test_bot_initialization),
        ("API подключение", test_api_connection),
        ("Telegram", test_telegram),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте '{name}': {e}")
            results.append((name, False))
    
    # Итоговый отчет
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{status}: {name}")
    
    print("\n" + "="*70)
    print(f"📈 РЕЗУЛЬТАТ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к работе!")
        return 0
    else:
        print("⚠️ Некоторые тесты провалены. Проверьте ошибки выше.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

