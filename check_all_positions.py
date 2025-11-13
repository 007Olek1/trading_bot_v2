#!/usr/bin/env python3
"""Проверка всех открытых позиций"""
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
print("📊 ПРОВЕРКА ВСЕХ ОТКРЫТЫХ ПОЗИЦИЙ")
print("="*70)

pos = session.get_positions(category="linear", settleCoin="USDT")
if pos and pos.get("retCode") == 0:
    positions = pos.get("result", {}).get("list", [])
    open_positions = [p for p in positions if float(p.get("size", 0)) > 0]
    
    print(f"\n📈 Найдено открытых позиций: {len(open_positions)}")
    print("="*70)
    
    for idx, p in enumerate(open_positions, 1):
        symbol = p.get("symbol", "N/A")
        side = p.get("side", "N/A")
        entry = float(p.get("avgPrice", 0))
        mark = float(p.get("markPrice", 0))
        tp_str = p.get("takeProfit", "")
        sl_str = p.get("stopLoss", "")
        leverage = p.get("leverage", "N/A")
        upnl = float(p.get("unrealisedPnl", 0))
        size_val = float(p.get("size", 0))
        
        print(f"\n{idx}. {symbol} {side}")
        print(f"   Вход: ${entry:.8f} | Текущая: ${mark:.8f}")
        print(f"   Размер: {size_val} | Плечо: {leverage}x")
        print(f"   uPnL: ${upnl:.4f}")
        
        if entry > 0:
            if side == "Sell":
                pnl_pct = ((entry - mark) / entry) * 100
            else:
                pnl_pct = ((mark - entry) / entry) * 100
            print(f"   PnL: {pnl_pct:+.2f}%")
            
            # TP
            tp_status = tp_str if tp_str else "НЕ УСТАНОВЛЕН"
            print(f"   TP: {tp_status}", end="")
            if tp_str:
                tp_val = float(tp_str)
                if side == "Sell":
                    tp_pct = ((entry - tp_val) / entry) * 100
                else:
                    tp_pct = ((tp_val - entry) / entry) * 100
                
                expected_tp_start = 1.15
                if tp_pct < expected_tp_start:
                    print(f" (⚠️ +{tp_pct:.2f}% < +{expected_tp_start:.2f}%)")
                elif tp_pct >= 1.15 and tp_pct < 4.0:
                    print(f" (✅ +{tp_pct:.2f}% в трейлинге)")
                else:
                    print(f" (🎉 +{tp_pct:.2f}% гарантировано)")
            else:
                print(" ❌")
            
            # SL
            sl_status = sl_str if sl_str else "НЕ УСТАНОВЛЕН"
            print(f"   SL: {sl_status}")
            if sl_str:
                sl_val = float(sl_str)
                if side == "Sell":
                    sl_pct = ((sl_val - entry) / entry) * 100
                else:
                    sl_pct = ((entry - sl_val) / entry) * 100
                
                if sl_pct > 0 and upnl >= 1.0:
                    print(f"      (✅ В BE - прибыль >= $1)")
                elif abs(sl_pct - 4.0) < 1.0:
                    print(f"      (✅ -$1 максимум)")
                else:
                    print(f"      ({sl_pct:+.2f}%)")
    
    print("\n" + "="*70)
    print("📋 СТАТУС МОНИТОРОВ:")
    
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    if 'monitor_trailing_tp' in result.stdout:
        print("   ✅ monitor_trailing_tp_universal.py: РАБОТАЕТ")
    else:
        print("   ❌ monitor_trailing_tp_universal.py: НЕ РАБОТАЕТ")
    
    if 'super_bot_v4_mtf.py' in result.stdout:
        print("   ✅ super_bot_v4_mtf.py: РАБОТАЕТ")
    else:
        print("   ❌ super_bot_v4_mtf.py: НЕ РАБОТАЕТ")
    
    print("="*70)
else:
    ret_msg = pos.get('retMsg', 'Unknown') if pos else 'No response'
    print(f"\n❌ Ошибка получения позиций: {ret_msg}")








