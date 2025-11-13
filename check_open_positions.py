#!/usr/bin/env python3
"""Проверка всех открытых позиций на бирже"""
import sys
sys.path.insert(0, '/opt/bot')

from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import pytz

load_dotenv(Path("/opt/bot/.env"))
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

session = HTTP(api_key=api_key, api_secret=api_secret, testnet=False)

print("="*70)
print("📊 ОТКРЫТЫЕ ПОЗИЦИИ НА БИРЖЕ")
print("="*70)

# Получаем открытые позиции
positions = session.get_positions(
    category="linear",
    settleCoin="USDT"
)

if positions and positions.get("retCode") == 0:
    pos_list = positions.get("result", {}).get("list", [])
    
    # Фильтруем только позиции с размером > 0
    open_positions = [p for p in pos_list if float(p.get("size", 0) or 0) > 0]
    
    if not open_positions:
        print("\n✅ Открытых позиций нет")
        print("="*70)
    else:
        print(f"\n📌 Найдено открытых позиций: {len(open_positions)}\n")
        
        total_upnl = 0.0
        total_notional = 0.0
        
        for i, pos in enumerate(open_positions, 1):
            symbol = pos.get("symbol", "N/A")
            side = pos.get("side", "")
            size = float(pos.get("size", 0) or 0)
            entry_price = float(pos.get("avgPrice", 0) or pos.get("entryPrice", 0) or 0)
            mark_price = float(pos.get("markPrice", 0) or 0)
            upnl = float(pos.get("unrealisedPnl", 0) or 0)
            leverage = float(pos.get("leverage", 0) or 1)
            tp_price = pos.get("takeProfit")
            sl_price = pos.get("stopLoss")
            created_time = pos.get("createdTime", "")
            updated_time = pos.get("updatedTime", "")
            
            # Расчет PnL в процентах
            if entry_price > 0:
                if side == "Buy":
                    pnl_pct = ((mark_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - mark_price) / entry_price) * 100
            else:
                pnl_pct = 0.0
            
            # Нотиональная стоимость
            notional = entry_price * size if entry_price > 0 else 0
            
            total_upnl += upnl
            total_notional += notional
            
            # Расчет TP/SL в процентах
            tp_pct = None
            sl_pct = None
            if entry_price > 0:
                if tp_price:
                    tp_val = float(tp_price)
                    if side == "Buy":
                        tp_pct = ((tp_val - entry_price) / entry_price) * 100
                    else:
                        tp_pct = ((entry_price - tp_val) / entry_price) * 100
                
                if sl_price:
                    sl_val = float(sl_price)
                    if side == "Buy":
                        sl_pct = ((entry_price - sl_val) / entry_price) * 100
                    else:
                        sl_pct = ((sl_val - entry_price) / entry_price) * 100
            
            # Время открытия
            duration_str = ""
            if created_time:
                try:
                    created_dt = datetime.fromtimestamp(int(created_time) / 1000, tz=pytz.timezone("Europe/Warsaw"))
                    duration = (datetime.now(pytz.timezone("Europe/Warsaw")) - created_dt).total_seconds()
                    hours = int(duration // 3600)
                    minutes = int((duration % 3600) // 60)
                    duration_str = f"{hours}ч {minutes}м"
                except:
                    duration_str = "N/A"
            
            print(f"{i}. 🔖 {symbol} {side.upper()}")
            print(f"   Вход: ${entry_price:.8f} | Текущая: ${mark_price:.8f}")
            print(f"   Размер: {size:.0f} | Левередж: {leverage}x")
            print(f"   uPnL: ${upnl:.4f} ({pnl_pct:+.2f}%)")
            print(f"   Нотиональ: ${notional:.2f}")
            
            if tp_price:
                tp_val = float(tp_price)
                tp_str = f"${tp_val:.8f}"
                if tp_pct is not None:
                    tp_str += f" ({tp_pct:+.2f}%)"
                print(f"   🎯 TP: {tp_str}")
            else:
                print(f"   🎯 TP: НЕ УСТАНОВЛЕН ⚠️")
            
            if sl_price:
                sl_val = float(sl_price)
                sl_str = f"${sl_val:.8f}"
                if sl_pct is not None:
                    sl_str += f" ({sl_pct:+.2f}%)"
                print(f"   🛑 SL: {sl_str}")
            else:
                print(f"   🛑 SL: НЕ УСТАНОВЛЕН ⚠️")
            
            if duration_str:
                print(f"   ⏰ Открыта: {duration_str} назад")
            
            print()
        
        print("="*70)
        print(f"💰 ИТОГО:")
        print(f"   Суммарный uPnL: ${total_upnl:.4f}")
        print(f"   Общая нотиональ: ${total_notional:.2f}")
        print(f"   Позиций: {len(open_positions)}")
        print("="*70)
else:
    ret_msg = positions.get('retMsg', 'Unknown') if positions else 'No response'
    print(f"\n❌ Ошибка получения позиций: {ret_msg}")






