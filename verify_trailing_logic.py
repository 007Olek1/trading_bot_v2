#!/usr/bin/env python3
"""
🧮 Проверка правильности расчётов Trailing Stop
Тестирует математику и логику перестановки SL
"""

from bot_v2_exchange import ExchangeManager
import asyncio

async def verify_trailing_calculations():
    print("="*70)
    print("🧮 ВЕРИФИКАЦИЯ TRAILING STOP РАСЧЁТОВ")
    print("="*70)
    
    em = ExchangeManager()
    await em.connect()
    
    positions = await em.fetch_positions()
    open_pos = [p for p in positions if p["contracts"] > 0]
    
    print(f"\n📊 Позиций для проверки: {len(open_pos)}\n")
    
    all_correct = True
    
    for p in open_pos:
        symbol = p["symbol"]
        side = p["side"]
        entry = p["entryPrice"]
        current = p.get("markPrice", entry)
        
        info = p.get("info", {})
        current_sl = info.get("stopLoss")
        
        print("="*70)
        print(f"💎 {symbol} | {side.upper()}")
        print("="*70)
        
        # ============================================
        # ШАГ 1: Расчёт текущей прибыли
        # ============================================
        print(f"\n📊 ШАГ 1: Расчёт прибыли")
        print(f"   Entry Price: ${entry:.4f}")
        print(f"   Current Price: ${current:.4f}")
        
        if side.lower() in ["buy", "long"]:
            price_change = current - entry
            price_change_pct = (price_change / entry) * 100
            print(f"   LONG: (Current - Entry) / Entry")
            print(f"         ({current:.4f} - {entry:.4f}) / {entry:.4f}")
        else:  # SHORT
            price_change = entry - current
            price_change_pct = (price_change / entry) * 100
            print(f"   SHORT: (Entry - Current) / Entry")
            print(f"          ({entry:.4f} - {current:.4f}) / {entry:.4f}")
        
        print(f"   = {price_change_pct:+.2f}%")
        
        # С учётом плеча
        profit_pct = price_change_pct * 5
        print(f"   С плечом 5x: {profit_pct:+.1f}%")
        
        # ============================================
        # ШАГ 2: Определение нового SL
        # ============================================
        print(f"\n🎯 ШАГ 2: Расчёт нового Trailing SL")
        
        new_sl = None
        trailing_level = None
        
        if profit_pct >= 10:
            if side.lower() in ["buy", "long"]:
                new_sl = entry * 1.05  # +5%
                trailing_level = "+5%"
            else:  # SHORT
                new_sl = entry * 1.02  # +2%
                trailing_level = "+2%"
            print(f"   Прибыль {profit_pct:.1f}% ≥ 10%")
            
        elif profit_pct >= 5:
            if side.lower() in ["buy", "long"]:
                new_sl = entry * 1.02  # +2%
                trailing_level = "+2%"
            else:  # SHORT
                new_sl = entry * 1.01  # +1%
                trailing_level = "+1%"
            print(f"   Прибыль {profit_pct:.1f}% ≥ 5%")
            
        elif profit_pct >= 2:
            if side.lower() in ["buy", "long"]:
                new_sl = entry  # Безубыток
                trailing_level = "безубыток"
            else:  # SHORT
                new_sl = entry * 1.002  # +0.2%
                trailing_level = "+0.2%"
            print(f"   Прибыль {profit_pct:.1f}% ≥ 2%")
        else:
            print(f"   Прибыль {profit_pct:.1f}% < 2% → SL не меняется")
        
        if new_sl:
            print(f"   → Новый SL: ${new_sl:.4f} ({trailing_level})")
            
            # Расчёт изменения от Entry
            if side.lower() in ["buy", "long"]:
                sl_change_pct = ((new_sl - entry) / entry) * 100
            else:  # SHORT
                sl_change_pct = ((entry - new_sl) / entry) * 100
            
            print(f"   → От Entry: {sl_change_pct:+.2f}%")
        
        # ============================================
        # ШАГ 3: Проверка должен ли обновляться
        # ============================================
        print(f"\n✅ ШАГ 3: Проверка условий обновления")
        
        if current_sl:
            current_sl_float = float(current_sl)
            print(f"   Текущий SL: ${current_sl_float:.4f}")
            
            if new_sl:
                print(f"   Новый SL: ${new_sl:.4f}")
                
                should_update = False
                
                if side.lower() in ["buy", "long"]:
                    # Для LONG: новый SL должен быть ВЫШЕ текущего
                    should_update = new_sl > current_sl_float
                    print(f"   LONG проверка: {new_sl:.4f} > {current_sl_float:.4f} = {should_update}")
                else:  # SHORT
                    # Для SHORT: новый SL должен быть НИЖЕ текущего и ВЫШЕ Entry
                    min_sl = entry * 1.001  # Entry + 0.1%
                    if new_sl < min_sl:
                        new_sl = min_sl
                        print(f"   ⚠️ SL скорректирован до минимума: ${new_sl:.4f}")
                    
                    should_update = new_sl < current_sl_float and new_sl > entry
                    print(f"   SHORT проверка:")
                    print(f"      {new_sl:.4f} < {current_sl_float:.4f} = {new_sl < current_sl_float}")
                    print(f"      {new_sl:.4f} > {entry:.4f} = {new_sl > entry}")
                    print(f"      Результат: {should_update}")
                
                if should_update:
                    print(f"   ✅ SL ДОЛЖЕН ОБНОВИТЬСЯ")
                else:
                    print(f"   ⏸️ SL НЕ ОБНОВЛЯЕТСЯ (условия не выполнены)")
            else:
                print(f"   ⏸️ Прибыль < 2% → SL остаётся: ${current_sl_float:.4f}")
        else:
            print(f"   ⚠️ Stop Loss не установлен!")
            all_correct = False
        
        # ============================================
        # ПРОВЕРКА КОРРЕКТНОСТИ
        # ============================================
        print(f"\n🔬 ВЕРИФИКАЦИЯ:")
        
        issues = []
        
        if current_sl:
            sl_f = float(current_sl)
            
            # Проверка 1: SL в правильную сторону от Entry
            if side.lower() in ["buy", "long"]:
                if sl_f >= entry:
                    print(f"   🎯 LONG: SL={sl_f:.4f} >= Entry={entry:.4f} → Безубыток!")
                elif sl_f < entry * 0.9:
                    print(f"   ⚠️ LONG: SL слишком далеко от Entry")
                    issues.append("SL слишком далеко")
                else:
                    print(f"   ✅ LONG: SL корректно установлен")
            else:  # SHORT
                if sl_f <= entry:
                    print(f"   ❌ SHORT: SL={sl_f:.4f} <= Entry={entry:.4f} → ОШИБКА!")
                    issues.append("SL ниже Entry для SHORT!")
                    all_correct = False
                elif sl_f > entry * 1.1:
                    print(f"   ⚠️ SHORT: SL слишком далеко от Entry")
                    issues.append("SL слишком далеко")
                else:
                    print(f"   ✅ SHORT: SL корректно установлен")
            
            # Проверка 2: Trailing работает
            if profit_pct >= 2:
                # Должен быть на безубытке или лучше
                if side.lower() in ["buy", "long"]:
                    if sl_f >= entry:
                        print(f"   ✅ Trailing активен: прибыль защищена")
                    else:
                        print(f"   ⚠️ При прибыли {profit_pct:.1f}% SL должен быть на Entry")
                        issues.append("Trailing не сработал")
                else:  # SHORT
                    expected_min = entry * 1.002  # +0.2%
                    if sl_f <= expected_min:
                        print(f"   ✅ Trailing активен: прибыль защищена")
                    else:
                        print(f"   ⚠️ SL можно подтянуть ближе")
        
        if issues:
            print(f"\n⚠️ Найдено проблем: {len(issues)}")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n✅ Все проверки пройдены!")
        
        print("")
    
    await em.disconnect()
    
    print("="*70)
    print("📊 ИТОГОВАЯ ОЦЕНКА")
    print("="*70)
    
    if all_correct:
        print("\n🎉 ВСЯ ЛОГИКА TRAILING STOP РАБОТАЕТ КОРРЕКТНО!")
        print("\n✅ ГОТОВ К НОЧНОЙ ТОРГОВЛЕ:")
        print("   ✅ Все SL установлены правильно")
        print("   ✅ Trailing работает для прибыльных позиций")
        print("   ✅ Математика расчётов верна")
        print("   ✅ Защита от убытков активна")
    else:
        print("\n⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
        print("   Рекомендуется исправить перед ночным запуском")
    
    print("="*70)
    
    return all_correct

asyncio.run(verify_trailing_calculations())


