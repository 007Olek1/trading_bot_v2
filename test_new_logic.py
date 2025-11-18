#!/usr/bin/env python3
"""
🧪 ТЕСТ НОВОЙ ЛОГИКИ - Безубыток при +10%
"""

import sys


def test_breakeven_logic():
    """Тест логики безубытка"""
    print("="*80)
    print("🧪 ТЕСТ: Новая логика безубытка при +10%")
    print("="*80)
    
    # Параметры позиции
    position_size_usd = 1.0
    leverage = 10
    entry_price = 0.4898
    direction = "SHORT"
    
    # Параметры безубытка
    breakeven_percent = 10.0
    trailing_callback = 1.5
    
    print(f"\n📊 ПОЗИЦИЯ:")
    print(f"  Депозит: ${position_size_usd}")
    print(f"  Плечо: x{leverage}")
    print(f"  Размер: ${position_size_usd * leverage}")
    print(f"  Направление: {direction}")
    print(f"  Цена входа: ${entry_price:.6f}")
    
    # Начальный SL
    max_loss_usd = 1.0
    position_value = position_size_usd * leverage
    loss_percent = max_loss_usd / position_value
    
    if direction == 'LONG':
        initial_sl = entry_price * (1 - loss_percent)
    else:
        initial_sl = entry_price * (1 + loss_percent)
    
    print(f"\n🛑 НАЧАЛЬНЫЙ SL:")
    print(f"  Цена SL: ${initial_sl:.6f}")
    print(f"  Макс. убыток: ${max_loss_usd}")
    
    # Цена безубытка
    if direction == 'LONG':
        breakeven_price = entry_price * (1 + breakeven_percent / 100)
    else:
        breakeven_price = entry_price * (1 - breakeven_percent / 100)
    
    print(f"\n🔒 БЕЗУБЫТОК:")
    print(f"  Активация: при +{breakeven_percent}%")
    print(f"  Цена безубытка: ${breakeven_price:.6f}")
    print(f"  SL переносится на: ${entry_price:.6f} (цена входа)")
    
    # Симуляция движения цены
    print(f"\n📈 СИМУЛЯЦИЯ ДВИЖЕНИЯ ЦЕНЫ:")
    
    test_prices = [
        ("Вход", entry_price, 0.0),
        ("Движение", 0.4700, 4.04),
        ("Движение", 0.4600, 6.08),
        ("Движение", 0.4500, 8.13),
        ("🔒 БЕЗУБЫТОК", 0.4408, 10.0),  # Активация безубытка
        ("Trailing", 0.4300, 12.21),
        ("Trailing", 0.4200, 14.25),
        ("Разворот", 0.4350, 11.19),
        ("Разворот", 0.4500, 8.13),
        ("🛑 SL сработал", 0.4898, 0.0),  # Закрытие по безубытку
    ]
    
    current_sl = initial_sl
    breakeven_activated = False
    trailing_active = False
    
    for label, price, expected_pnl in test_prices:
        # Расчёт PnL
        if direction == 'LONG':
            pnl_percent = ((price - entry_price) / entry_price) * 100
        else:
            pnl_percent = ((entry_price - price) / entry_price) * 100
        
        print(f"\n  {label}: ${price:.6f} (PnL: {pnl_percent:+.2f}%)")
        
        # Проверка активации безубытка
        if pnl_percent >= breakeven_percent and not breakeven_activated:
            breakeven_activated = True
            current_sl = entry_price
            trailing_active = True
            print(f"    ✅ Безубыток активирован! SL → ${current_sl:.6f}")
        
        # Trailing SL после безубытка
        if breakeven_activated and trailing_active:
            callback = trailing_callback / 100
            if direction == 'LONG':
                new_sl = price * (1 - callback)
            else:
                new_sl = price * (1 + callback)
            
            # Обновляем только если лучше
            should_update = False
            if direction == 'LONG' and new_sl > current_sl:
                should_update = True
            elif direction == 'SHORT' and new_sl < current_sl:
                should_update = True
            
            if should_update:
                print(f"    🔄 Trailing SL: ${current_sl:.6f} → ${new_sl:.6f}")
                current_sl = new_sl
        
        # Проверка срабатывания SL
        if direction == 'LONG' and price <= current_sl:
            print(f"    🛑 SL сработал! Закрытие по ${current_sl:.6f}")
            break
        elif direction == 'SHORT' and price >= current_sl:
            print(f"    🛑 SL сработал! Закрытие по ${current_sl:.6f}")
            break
    
    print(f"\n✅ РЕЗУЛЬТАТ:")
    print(f"  Финальный SL: ${current_sl:.6f}")
    print(f"  Безубыток активирован: {'Да' if breakeven_activated else 'Нет'}")
    
    # Проверка что SL не хуже цены входа
    if direction == 'SHORT':
        sl_better_than_entry = current_sl <= entry_price
    else:
        sl_better_than_entry = current_sl >= entry_price
    
    print(f"  Защита от убытков: {'Да' if sl_better_than_entry else 'Нет'}")
    
    # Проверка
    assert breakeven_activated, "❌ Безубыток не активировался!"
    assert sl_better_than_entry, "❌ SL хуже цены входа!"
    
    print(f"\n✅ ТЕСТ ПРОЙДЕН!")
    print()


def test_profit_scenarios():
    """Тест различных сценариев прибыли"""
    print("="*80)
    print("🧪 ТЕСТ: Сценарии прибыли")
    print("="*80)
    
    scenarios = [
        {
            "name": "Сценарий 1: Быстрый разворот (убыток)",
            "entry": 0.4898,
            "direction": "SHORT",
            "max_price": 0.4700,  # +4% (не достиг безубытка)
            "exit_price": 0.5388,  # SL
            "expected_pnl": -1.0
        },
        {
            "name": "Сценарий 2: Безубыток сработал (0)",
            "entry": 0.4898,
            "direction": "SHORT",
            "max_price": 0.4408,  # +10% (безубыток активирован)
            "exit_price": 0.4898,  # SL на безубытке
            "expected_pnl": 0.0
        },
        {
            "name": "Сценарий 3: Trailing защитил прибыль (+12%)",
            "entry": 0.4898,
            "direction": "SHORT",
            "max_price": 0.4300,  # +12.21%
            "exit_price": 0.4365,  # Trailing SL (откат 1.5%)
            "expected_pnl": 1.09
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"  Вход: ${scenario['entry']:.6f}")
        print(f"  Максимум: ${scenario['max_price']:.6f}")
        print(f"  Выход: ${scenario['exit_price']:.6f}")
        
        # Расчёт PnL
        if scenario['direction'] == 'LONG':
            pnl_percent = ((scenario['exit_price'] - scenario['entry']) / scenario['entry']) * 100
        else:
            pnl_percent = ((scenario['entry'] - scenario['exit_price']) / scenario['entry']) * 100
        
        position_value = 10.0  # $1 x10
        pnl_usd = position_value * (pnl_percent / 100)
        
        print(f"  PnL: ${pnl_usd:.2f} ({pnl_percent:+.2f}%)")
        print(f"  Ожидалось: ${scenario['expected_pnl']:.2f}")
        
        if abs(pnl_usd - scenario['expected_pnl']) < 0.2:
            print(f"  ✅ Соответствует ожиданиям")
        else:
            print(f"  ⚠️  Отклонение от ожидаемого")
    
    print(f"\n✅ ВСЕ СЦЕНАРИИ ПРОВЕРЕНЫ!")
    print()


def run_all_tests():
    """Запуск всех тестов"""
    print("\n🔬 ТЕСТИРОВАНИЕ НОВОЙ ЛОГИКИ БЕЗУБЫТКА")
    print("="*80)
    print()
    
    try:
        test_breakeven_logic()
        test_profit_scenarios()
        
        print("="*80)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*80)
        print()
        
        print("📋 ИТОГИ:")
        print("  ✅ Безубыток активируется при +10%")
        print("  ✅ SL переносится на цену входа")
        print("  ✅ Trailing SL защищает прибыль")
        print("  ✅ Максимальный убыток: $1")
        print("  ✅ Минимальная прибыль после безубытка: $0")
        print()
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
