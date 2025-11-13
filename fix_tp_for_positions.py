#!/usr/bin/env python3
"""Исправление TP для открытых позиций до минимального +1.15%"""
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
print("🔧 ИСПРАВЛЕНИЕ TP ДО МИНИМАЛЬНОГО +1.15%")
print("="*70)

pos = session.get_positions(category="linear", settleCoin="USDT")
if pos and pos.get("retCode") == 0:
    positions = pos.get("result", {}).get("list", [])
    open_positions = [p for p in positions if float(p.get("size", 0)) > 0]
    
    fixed_count = 0
    
    for p in open_positions:
        symbol = p.get("symbol", "N/A")
        side = p.get("side", "N/A")
        entry = float(p.get("avgPrice", 0))
        tp_str = p.get("takeProfit", "")
        size_val = float(p.get("size", 0))
        
        if entry > 0 and size_val > 0:
            # Рассчитываем правильный TP +1.15%
            tp_percent = 1.15
            
            if side == "Buy":
                new_tp = entry * (1 + tp_percent / 100.0)
            else:  # Sell
                new_tp = entry * (1 - tp_percent / 100.0)
            
            # Проверяем нужно ли обновить
            need_update = False
            if not tp_str:
                need_update = True
                print(f"\n{symbol} {side}: TP НЕ УСТАНОВЛЕН")
            else:
                current_tp = float(tp_str)
                if side == "Buy":
                    current_tp_pct = ((current_tp - entry) / entry) * 100
                else:
                    current_tp_pct = ((entry - current_tp) / entry) * 100
                
                if current_tp_pct < tp_percent:
                    need_update = True
                    print(f"\n{symbol} {side}: TP {current_tp_pct:.2f}% < {tp_percent:.2f}%")
            
            if need_update:
                try:
                    result = session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        takeProfit=f"{new_tp:.8f}",
                        tpslMode="Full",
                        positionIdx=0
                    )
                    
                    if result.get("retCode") == 0:
                        print(f"   ✅ TP обновлен до ${new_tp:.8f} (+{tp_percent:.2f}%)")
                        fixed_count += 1
                    else:
                        print(f"   ❌ Ошибка: {result.get('retMsg', 'Unknown')}")
                except Exception as e:
                    print(f"   ❌ Ошибка обновления TP: {e}")
            else:
                if tp_str:
                    current_tp = float(tp_str)
                    if side == "Buy":
                        current_tp_pct = ((current_tp - entry) / entry) * 100
                    else:
                        current_tp_pct = ((entry - current_tp) / entry) * 100
                    print(f"\n{symbol} {side}: TP уже правильный (+{current_tp_pct:.2f}%)")
    
    print("\n" + "="*70)
    print(f"✅ Исправлено позиций: {fixed_count}/{len(open_positions)}")
    print("="*70)
else:
    ret_msg = pos.get('retMsg', 'Unknown') if pos else 'No response'
    print(f"\n❌ Ошибка получения позиций: {ret_msg}")








