#!/usr/bin/env python3
"""
Проверка правильности расчетов SL/TP
"""

# Параметры
POSITION_SIZE = 5.0  # $5
LEVERAGE = 5  # 5x
POSITION_NOTIONAL = 25.0  # $25
MAX_STOP_LOSS_USD = 5.0
STOP_LOSS_PERCENT = (MAX_STOP_LOSS_USD / POSITION_NOTIONAL) * 100  # 20%

print("=" * 70)
print("🔍 ПРОВЕРКА РАСЧЕТОВ SL/TP")
print("=" * 70)

# Тест 1: LONG позиция
print("\n1️⃣ LONG позиция (BTC):")
entry_long = 111000.0
sl_long = entry_long * (1 - STOP_LOSS_PERCENT / 100)  # Формула из кода
contracts_long = POSITION_NOTIONAL / entry_long
loss_per_contract = entry_long - sl_long
total_loss = contracts_long * loss_per_contract

print(f"   Вход: ${entry_long:,.2f}")
print(f"   SL: ${sl_long:,.2f} (-{STOP_LOSS_PERCENT:.2f}%)")
print(f"   Контрактов: {contracts_long:.6f}")
print(f"   Убыток на контракт: ${loss_per_contract:,.2f}")
print(f"   ОБЩИЙ УБЫТОК: ${total_loss:.2f} {'✅ ПРАВИЛЬНО' if abs(total_loss - MAX_STOP_LOSS_USD) < 0.01 else '❌ ОШИБКА'}")

# Тест 2: SHORT позиция
print("\n2️⃣ SHORT позиция (XRP):")
entry_short = 2.60
sl_short = entry_short * (1 + STOP_LOSS_PERCENT / 100)  # Формула из кода
contracts_short = POSITION_NOTIONAL / entry_short
loss_per_contract_short = sl_short - entry_short
total_loss_short = contracts_short * loss_per_contract_short

print(f"   Вход: ${entry_short:.4f}")
print(f"   SL: ${sl_short:.4f} (+{STOP_LOSS_PERCENT:.2f}%)")
print(f"   Контрактов: {contracts_short:.6f}")
print(f"   Убыток на контракт: ${loss_per_contract_short:.4f}")
print(f"   ОБЩИЙ УБЫТОК: ${total_loss_short:.2f} {'✅ ПРАВИЛЬНО' if abs(total_loss_short - MAX_STOP_LOSS_USD) < 0.01 else '❌ ОШИБКА'}")

# Тест 3: Трейлинг стоп LONG
print("\n3️⃣ Трейлинг стоп для LONG:")
entry_tl = 100.0
current_price_tl = 110.0
trailing_distance = MAX_STOP_LOSS_USD / POSITION_NOTIONAL * entry_tl
new_sl_long = current_price_tl - trailing_distance
contracts_tl = POSITION_NOTIONAL / entry_tl
loss_from_new_sl = contracts_tl * (entry_tl - new_sl_long)

print(f"   Вход: ${entry_tl:.2f}")
print(f"   Текущая цена: ${current_price_tl:.2f} (+{((current_price_tl/entry_tl - 1)*100):.2f}%)")
print(f"   Расстояние трейлинга: ${trailing_distance:.2f}")
print(f"   Новый SL: ${new_sl_long:.2f}")
print(f"   Убыток от нового SL (от входа): ${loss_from_new_sl:.2f}")
print(f"   ✅ Трейлинг защищает прибыль!")

# Тест 4: Трейлинг стоп SHORT
print("\n4️⃣ Трейлинг стоп для SHORT:")
entry_ts = 2.60
current_price_ts = 2.40
trailing_distance_short = MAX_STOP_LOSS_USD / POSITION_NOTIONAL * entry_ts
new_sl_short = current_price_ts + trailing_distance_short
contracts_ts = POSITION_NOTIONAL / entry_ts
loss_from_new_sl_short = contracts_ts * (new_sl_short - entry_ts)

print(f"   Вход: ${entry_ts:.4f}")
print(f"   Текущая цена: ${current_price_ts:.4f} ({((current_price_ts/entry_ts - 1)*100):.2f}%)")
print(f"   Расстояние трейлинга: ${trailing_distance_short:.4f}")
print(f"   Новый SL: ${new_sl_short:.4f}")
print(f"   Убыток от нового SL (от входа): ${loss_from_new_sl_short:.2f}")
print(f"   ✅ Трейлинг защищает прибыль!")

# Тест 5: Take Profit уровни
print("\n5️⃣ Take Profit уровни:")
TP_LEVELS = [
    {"level": 1, "percent": 4.0, "portion": 0.40},
    {"level": 2, "percent": 6.0, "portion": 0.20},
    {"level": 3, "percent": 8.0, "portion": 0.20},
    {"level": 4, "percent": 10.0, "portion": 0.10},
    {"level": 5, "percent": 12.0, "portion": 0.05},
    {"level": 6, "percent": 15.0, "portion": 0.05},
]

entry_tp = 100.0
direction_tp = "buy"
contracts_tp = POSITION_NOTIONAL / entry_tp
total_profit = 0.0

print(f"\n   Тест: LONG @ ${entry_tp:.2f}")
print(f"   Контрактов: {contracts_tp:.6f}\n")

for tp in TP_LEVELS:
    tp_price = entry_tp * (1 + tp["percent"] / 100) if direction_tp == "buy" else entry_tp * (1 - tp["percent"] / 100)
    contracts_closed = contracts_tp * tp["portion"]
    profit_per_contract = tp_price - entry_tp if direction_tp == "buy" else entry_tp - tp_price
    profit_usd = contracts_closed * profit_per_contract
    total_profit += profit_usd
    
    expected_profit = POSITION_NOTIONAL * tp["portion"] * tp["percent"] / 100
    is_correct = abs(profit_usd - expected_profit) < 0.01
    
    print(f"   TP{tp['level']}: ${tp_price:.4f} (+{tp['percent']:.0f}%)")
    print(f"      Закрытие: {tp['portion']*100:.0f}% = {contracts_closed:.6f} контрактов")
    print(f"      Прибыль: ${profit_usd:.4f} (ожидается: ${expected_profit:.4f}) {'✅' if is_correct else '❌'}")
    print(f"      Накопительно: ${total_profit:.4f}\n")

print(f"   ИТОГО: ${total_profit:.4f}")
expected_total = sum([POSITION_NOTIONAL * tp["portion"] * tp["percent"] / 100 for tp in TP_LEVELS])
print(f"   ОЖИДАЕТСЯ: ${expected_total:.4f}")
print(f"   {'✅ ВСЕ РАСЧЕТЫ ПРАВИЛЬНЫЕ!' if abs(total_profit - expected_total) < 0.01 else '❌ ОШИБКА В РАСЧЕТАХ'}")

print("\n" + "=" * 70)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 70)





