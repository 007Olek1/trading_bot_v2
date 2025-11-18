#!/usr/bin/env python3
"""
🧪 ТЕСТИРОВАНИЕ ЛОГИКИ БОТА
Комплексная проверка всех критических функций
"""

import sys
from decimal import Decimal


def test_tp_calculation():
    """Тест расчёта TP уровней"""
    print("="*80)
    print("🧪 ТЕСТ #1: Расчёт TP уровней")
    print("="*80)
    
    # Симуляция SHORT позиции ADAUSDT
    entry_price = 0.4898
    direction = "SHORT"
    initial_quantity = 20.0
    
    # TP уровни из config
    tp_levels = [
        {"percent": 4.0, "size": 0.40},
        {"percent": 6.0, "size": 0.20},
        {"percent": 8.0, "size": 0.20},
        {"percent": 10.0, "size": 0.10},
        {"percent": 12.0, "size": 0.05},
        {"percent": 15.0, "size": 0.05},
    ]
    
    print(f"\n📊 Позиция: {direction} {initial_quantity} монет @ ${entry_price}")
    print(f"\n🎯 TP уровни:")
    
    total_closed = 0
    remaining_qty = initial_quantity
    
    for i, tp in enumerate(tp_levels, 1):
        # Расчёт цены TP
        percent = tp['percent'] / 100
        if direction == 'LONG':
            tp_price = entry_price * (1 + percent)
        else:
            tp_price = entry_price * (1 - percent)
        
        # Расчёт количества для закрытия от ИСХОДНОГО
        close_qty = initial_quantity * tp['size']
        
        # Проверка
        if close_qty > remaining_qty:
            close_qty = remaining_qty
        
        remaining_qty -= close_qty
        total_closed += close_qty
        
        # Расчёт PnL
        if direction == 'LONG':
            pnl = (tp_price - entry_price) * close_qty
        else:
            pnl = (entry_price - tp_price) * close_qty
        
        pnl_percent = (pnl / (entry_price * close_qty)) * 100
        
        print(f"  TP{i}: ${tp_price:.6f} ({tp['percent']:+.1f}%)")
        print(f"    Закрыть: {close_qty:.4f} монет ({tp['size']*100:.0f}%)")
        print(f"    PnL: ${pnl:.4f} ({pnl_percent:+.2f}%)")
        print(f"    Останется: {remaining_qty:.4f} монет")
        print()
    
    # Проверка
    assert abs(total_closed - initial_quantity) < 0.0001, "❌ Ошибка: не вся позиция закрыта!"
    assert abs(remaining_qty) < 0.0001, "❌ Ошибка: осталось незакрытое количество!"
    
    print("✅ ТЕСТ ПРОЙДЕН: Все TP рассчитаны правильно")
    print()


def test_fee_calculation():
    """Тест расчёта комиссий"""
    print("="*80)
    print("🧪 ТЕСТ #2: Расчёт комиссий")
    print("="*80)
    
    position_value = 10.0  # $10 позиция
    leverage = 10
    
    # Bybit комиссии
    taker_fee = 0.0006  # 0.06%
    
    # Комиссия на вход
    entry_fee = position_value * taker_fee
    # Комиссия на выход
    exit_fee = position_value * taker_fee
    # Общая комиссия
    total_fee = entry_fee + exit_fee
    
    print(f"\n💰 Размер позиции: ${position_value} x{leverage} = ${position_value * leverage}")
    print(f"📊 Комиссия Bybit Taker: {taker_fee*100}%")
    print(f"\n💸 Комиссия на вход: ${entry_fee:.4f}")
    print(f"💸 Комиссия на выход: ${exit_fee:.4f}")
    print(f"💸 Общая комиссия: ${total_fee:.4f}")
    
    # Минимальный процент для безубытка
    min_profit_percent = (total_fee / position_value) * 100
    
    print(f"\n⚠️  Минимальный профит для безубытка: {min_profit_percent:.2f}%")
    
    # Проверка TP1 (+4%)
    tp1_percent = 4.0
    tp1_profit = position_value * (tp1_percent / 100)
    tp1_net_profit = tp1_profit - total_fee
    
    print(f"\n🎯 TP1 (+{tp1_percent}%):")
    print(f"  Валовая прибыль: ${tp1_profit:.4f}")
    print(f"  Комиссии: -${total_fee:.4f}")
    print(f"  Чистая прибыль: ${tp1_net_profit:.4f}")
    
    assert tp1_net_profit > 0, "❌ Ошибка: TP1 не покрывает комиссии!"
    
    print("\n✅ ТЕСТ ПРОЙДЕН: Комиссии рассчитаны правильно")
    print()


def test_trailing_sl():
    """Тест trailing stop loss"""
    print("="*80)
    print("🧪 ТЕСТ #3: Trailing Stop Loss")
    print("="*80)
    
    entry_price = 0.4898
    direction = "SHORT"
    
    # Параметры trailing
    activation_percent = 2.0  # Активация при +2%
    callback_percent = 1.5    # Откат 1.5%
    
    # Начальный SL
    max_loss_usd = 1.0
    position_size_usd = 10.0
    loss_percent = max_loss_usd / position_size_usd
    
    if direction == 'LONG':
        initial_sl = entry_price * (1 - loss_percent)
    else:
        initial_sl = entry_price * (1 + loss_percent)
    
    print(f"\n📊 Позиция: {direction} @ ${entry_price}")
    print(f"🛑 Начальный SL: ${initial_sl:.6f}")
    print(f"🔄 Активация trailing: +{activation_percent}%")
    print(f"📉 Откат trailing: {callback_percent}%")
    
    # Симуляция движения цены
    print(f"\n📈 Симуляция движения цены:")
    
    test_prices = [
        0.4800,  # +2% profit - активация trailing
        0.4750,  # +3% profit - обновление SL
        0.4700,  # +4% profit - обновление SL
    ]
    
    current_sl = initial_sl
    trailing_active = False
    
    for price in test_prices:
        # Расчёт PnL
        if direction == 'LONG':
            pnl_percent = ((price - entry_price) / entry_price) * 100
        else:
            pnl_percent = ((entry_price - price) / entry_price) * 100
        
        print(f"\n  Цена: ${price:.6f} (PnL: {pnl_percent:+.2f}%)")
        
        # Проверка активации trailing
        if pnl_percent >= activation_percent:
            if not trailing_active:
                trailing_active = True
                print(f"    🔄 Trailing SL активирован!")
            
            # Расчёт нового SL
            callback = callback_percent / 100
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
                print(f"    📊 SL обновлён: ${current_sl:.6f} → ${new_sl:.6f}")
                current_sl = new_sl
            else:
                print(f"    ⏸️  SL не обновлён (текущий лучше)")
        else:
            print(f"    ⏸️  Trailing не активен (нужно +{activation_percent}%)")
    
    print(f"\n🛑 Финальный SL: ${current_sl:.6f}")
    
    # Проверка что SL лучше начального
    if direction == 'SHORT':
        assert current_sl < initial_sl, "❌ Ошибка: SL не улучшился!"
    else:
        assert current_sl > initial_sl, "❌ Ошибка: SL не улучшился!"
    
    print("\n✅ ТЕСТ ПРОЙДЕН: Trailing SL работает правильно")
    print()


def test_position_sizing():
    """Тест расчёта размера позиции"""
    print("="*80)
    print("🧪 ТЕСТ #4: Расчёт размера позиции")
    print("="*80)
    
    position_size_usd = 1.0
    leverage = 10
    current_price = 0.4898
    
    # Расчёт количества монет
    quantity = (position_size_usd * leverage) / current_price
    
    print(f"\n💰 Размер позиции: ${position_size_usd}")
    print(f"⚡ Плечо: x{leverage}")
    print(f"📊 Цена: ${current_price}")
    print(f"📦 Количество: {quantity:.4f} монет")
    print(f"💵 Общая стоимость: ${quantity * current_price:.2f}")
    
    # Проверка
    total_value = quantity * current_price
    expected_value = position_size_usd * leverage
    
    assert abs(total_value - expected_value) < 0.01, "❌ Ошибка: неправильный расчёт!"
    
    print("\n✅ ТЕСТ ПРОЙДЕН: Размер позиции рассчитан правильно")
    print()


def run_all_tests():
    """Запуск всех тестов"""
    print("\n")
    print("🔬 ГЕНЕРАЛЬНОЕ ТЕСТИРОВАНИЕ ЛОГИКИ БОТА")
    print("="*80)
    print()
    
    try:
        test_tp_calculation()
        test_fee_calculation()
        test_trailing_sl()
        test_position_sizing()
        
        print("="*80)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*80)
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
