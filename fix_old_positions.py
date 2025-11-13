#!/usr/bin/env python3
"""Исправление старых позиций: установка правильного TP и проверка автозакрытия"""
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
print("🔧 ИСПРАВЛЕНИЕ СТАРЫХ ПОЗИЦИЙ")
print("="*70)

# Получаем открытые позиции
positions = session.get_positions(
    category="linear",
    settleCoin="USDT"
)

if positions and positions.get("retCode") == 0:
    pos_list = positions.get("result", {}).get("list", [])
    open_positions = [p for p in pos_list if float(p.get("size", 0) or 0) > 0]
    
    if not open_positions:
        print("\n✅ Открытых позиций нет")
    else:
        print(f"\n📌 Найдено позиций: {len(open_positions)}\n")
        
        MIN_TP_PERCENT = 1.15  # Минимальный TP +1.15%
        MAX_HOLD_HOURS = 24  # Максимальное время удержания
        
        for pos in open_positions:
            symbol = pos.get("symbol", "N/A")
            side = pos.get("side", "")
            entry_price = float(pos.get("avgPrice", 0) or pos.get("entryPrice", 0) or 0)
            tp_price_str = pos.get("takeProfit")
            created_time = pos.get("createdTime", "")
            
            if entry_price <= 0:
                continue
            
            # Проверка времени открытия
            should_close_time = False
            if created_time:
                try:
                    created_dt = datetime.fromtimestamp(int(created_time) / 1000, tz=pytz.timezone("Europe/Warsaw"))
                    duration_hours = (datetime.now(pytz.timezone("Europe/Warsaw")) - created_dt).total_seconds() / 3600
                    if duration_hours > MAX_HOLD_HOURS:
                        should_close_time = True
                        print(f"⚠️ {symbol} {side.upper()}: Открыта {duration_hours:.1f} часов назад (>24ч) - должна быть закрыта!")
                except:
                    pass
            
            # Проверка и исправление TP
            needs_tp_fix = False
            if tp_price_str:
                tp_price = float(tp_price_str)
                if side == "Buy":
                    tp_pct = ((tp_price - entry_price) / entry_price) * 100
                else:
                    tp_pct = ((entry_price - tp_price) / entry_price) * 100
                
                if tp_pct < MIN_TP_PERCENT:
                    needs_tp_fix = True
                    print(f"⚠️ {symbol} {side.upper()}: TP = +{tp_pct:.2f}% < {MIN_TP_PERCENT}% - нужно исправить!")
            else:
                needs_tp_fix = True
                print(f"⚠️ {symbol} {side.upper()}: TP не установлен!")
            
            # Исправление TP
            if needs_tp_fix:
                if side == "Buy":
                    new_tp = entry_price * (1 + MIN_TP_PERCENT / 100.0)
                else:
                    new_tp = entry_price * (1 - MIN_TP_PERCENT / 100.0)
                
                print(f"   🔧 Устанавливаем TP: ${new_tp:.8f} (+{MIN_TP_PERCENT}%)")
                
                # Устанавливаем TP через set_trading_stop
                try:
                    result = session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        takeProfit=new_tp,
                        tpTriggerBy="LastPrice" if side == "Buy" else "LastPrice"
                    )
                    
                    if result.get("retCode") == 0:
                        print(f"   ✅ TP успешно установлен")
                    else:
                        print(f"   ❌ Ошибка установки TP: {result.get('retMsg', 'Unknown')}")
                except Exception as e:
                    print(f"   ❌ Исключение при установке TP: {e}")
        
        print("\n" + "="*70)
        print("📋 РЕКОМЕНДАЦИИ:")
        print("   1. Проверить логику автозакрытия через 24 часа в monitor_positions")
        print("   2. Убедиться, что монитор запущен и работает")
        print("   3. Проверить, почему позиции не были закрыты автоматически")
        print("="*70)








