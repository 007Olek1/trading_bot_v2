#!/usr/bin/env python3
"""Проверка позиции HYPEUSDT SHORT"""
import sys
sys.path.insert(0, '/opt/bot')

from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path("/opt/bot/.env"))
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

session = HTTP(api_key=api_key, api_secret=api_secret, testnet=False)

print("="*70)
print("📊 ПРОВЕРКА ПОЗИЦИИ HYPEUSDT SHORT")
print("="*70)

pos = session.get_positions(category="linear", symbol="HYPEUSDT", settleCoin="USDT")
if pos and pos.get("retCode") == 0:
    positions = pos.get("result", {}).get("list", [])
    found = False
    for p in positions:
        size = float(p.get("size", 0))
        if size > 0:
            found = True
            symbol = p.get("symbol", "N/A")
            side = p.get("side", "N/A")
            entry = float(p.get("avgPrice", 0))
            mark = float(p.get("markPrice", 0))
            tp_str = p.get("takeProfit", "")
            sl_str = p.get("stopLoss", "")
            leverage = p.get("leverage", "N/A")
            upnl = float(p.get("unrealisedPnl", 0))
            size_val = float(p.get("size", 0))
            
            print(f"\n📊 ПОЗИЦИЯ НА БИРЖЕ:")
            print(f"   Символ: {symbol}")
            print(f"   Направление: {side}")
            print(f"   Вход: ${entry:.5f}")
            print(f"   Текущая: ${mark:.5f}")
            print(f"   Размер: {size_val}")
            print(f"   Плечо: {leverage}x")
            print(f"   uPnL: ${upnl:.4f}")
            tp_status = tp_str if tp_str else "НЕ УСТАНОВЛЕН"
            print(f"   TP на бирже: {tp_status}")
            sl_status = sl_str if sl_str else "НЕ УСТАНОВЛЕН"
            print(f"   SL на бирже: {sl_status}")
            
            if entry > 0 and side == "Sell":
                # Расчет PnL
                pnl_pct = ((entry - mark) / entry) * 100
                position_notional = entry * size_val
                
                print(f"\n💰 ТЕКУЩИЙ PnL:")
                print(f"   Процент: {pnl_pct:+.2f}%")
                print(f"   USDT: ${upnl:.4f}")
                print(f"   Нотиональная стоимость: ${position_notional:.2f}")
                
                # Проверка TP
                print(f"\n🎯 ПРОВЕРКА TP:")
                if tp_str:
                    tp_val = float(tp_str)
                    tp_pct = ((entry - tp_val) / entry) * 100
                    print(f"   ✅ TP установлен: ${tp_val:.5f}")
                    print(f"   TP процент: +{tp_pct:.2f}%")
                    
                    # Ожидаемые значения
                    expected_tp_start = 1.15  # Стартовый TP +1.15%
                    expected_tp_guaranteed = 4.0  # Гарантированный +4% = +$1
                    
                    if tp_pct < 1.0:
                        print(f"   ❌ TP СЛИШКОМ МАЛ! Должен быть минимум +{expected_tp_start:.2f}% (стартовый)")
                    elif tp_pct < expected_tp_start:
                        print(f"   ⚠️ TP меньше стартового +{expected_tp_start:.2f}% (сейчас +{tp_pct:.2f}%)")
                        print(f"   💡 Монитор должен обновить TP до +{expected_tp_start:.2f}% или выше")
                    elif tp_pct >= expected_tp_start and tp_pct < expected_tp_guaranteed:
                        print(f"   ✅ TP в диапазоне трейлинга (+{expected_tp_start:.2f}% → +{expected_tp_guaranteed:.2f}%)")
                        if pnl_pct >= 0:
                            steps = int((pnl_pct - expected_tp_start) / 0.5)
                            expected_tp_current = expected_tp_start + steps * 0.5
                            if abs(tp_pct - expected_tp_current) > 0.2:
                                print(f"   ⚠️ TP должен быть около +{expected_tp_current:.2f}% при текущем PnL {pnl_pct:.2f}%")
                    elif tp_pct >= expected_tp_guaranteed:
                        print(f"   🎉 TP на гарантированном уровне (+{tp_pct:.2f}% = +$1+)")
                        profit_usd = position_notional * (tp_pct / 100)
                        print(f"   💰 Ожидаемая прибыль: ${profit_usd:.2f}")
                else:
                    print(f"   ❌ TP НЕ УСТАНОВЛЕН на бирже!")
                    print(f"   ⚠️ Монитор должен установить TP немедленно")
                
                # Проверка SL
                print(f"\n🛑 ПРОВЕРКА SL:")
                if sl_str:
                    sl_val = float(sl_str)
                    sl_pct = ((sl_val - entry) / entry) * 100
                    print(f"   ✅ SL установлен: ${sl_val:.5f}")
                    print(f"   SL процент: {sl_pct:+.2f}%")
                    
                    if sl_pct > 0:
                        print(f"   ✅ SL выше входа - возможно уже в BE или трейлится")
                        if upnl >= 1.0:
                            print(f"   🎉 Прибыль >= $1 - SL должен быть в BE (около входа)")
                    else:
                        expected_sl_pct = -4.0  # -$1 на $25 позиции = -4%
                        if abs(sl_pct - expected_sl_pct) < 1.0:
                            print(f"   ✅ SL соответствует ожидаемому (-$1 максимум)")
                        else:
                            print(f"   ⚠️ SL отличается от ожидаемого -$1 ({expected_sl_pct:.2f}%)")
                else:
                    print(f"   ❌ SL НЕ УСТАНОВЛЕН на бирже!")
                    print(f"   ⚠️ Монитор должен установить SL немедленно")
            
            print(f"\n📋 РЕКОМЕНДАЦИИ:")
            print(f"   1. Проверить работает ли monitor_trailing_tp_universal.py")
            print(f"   2. Убедиться что TP трейлится каждые 30 сек")
            print(f"   3. Проверить соответствие логике входа (45m+1h+4h подтверждение)")
            break
    
    if not found:
        print("\n⚠️ Позиция HYPEUSDT не найдена на бирже")
else:
    ret_msg = pos.get('retMsg', 'Unknown') if pos else 'No response'
    print(f"\n❌ Ошибка получения позиций: {ret_msg}")








