#!/usr/bin/env python3
"""Анализ закрытой позиции AI16ZUSDT"""
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
print("🔍 АНАЛИЗ ЗАКРЫТИЯ AI16ZUSDT LONG")
print("="*70)

closed = session.get_closed_pnl(
    category="linear",
    symbol="AI16ZUSDT",
    limit=3
)

if closed and closed.get("retCode") == 0:
    closed_positions = closed.get("result", {}).get("list", [])
    if closed_positions:
        latest = closed_positions[0]
        
        entry = float(latest.get("avgEntryPrice", 0))
        exit_price = float(latest.get("avgExitPrice", 0))
        closed_pnl = float(latest.get("closedPnl", 0))
        size = float(latest.get("qty", 0))
        side = latest.get("side", "")
        
        print(f"\n📊 ДАННЫЕ ЗАКРЫТОЙ ПОЗИЦИИ:")
        print(f"   Символ: AI16ZUSDT")
        print(f"   Направление: {side}")
        print(f"   Размер: {size}")
        print(f"   Вход: ${entry:.8f}")
        print(f"   Выход: ${exit_price:.8f}")
        print(f"   Closed PnL: ${closed_pnl:.4f}")
        
        # Расчет ожидаемого PnL
        if side == "Buy":
            pnl_pct = ((exit_price - entry) / entry) * 100
            position_notional = entry * size
            expected_loss_max = -1.0  # Максимальный убыток -$1
            
            print(f"\n💰 РАСЧЕТЫ:")
            print(f"   PnL процент: {pnl_pct:.2f}%")
            print(f"   Нотиональная стоимость: ${position_notional:.2f}")
            print(f"   Ожидаемый максимальный убыток: ${expected_loss_max:.2f}")
            
            # Проверка почему превышен лимит
            if closed_pnl < expected_loss_max:
                excess = abs(closed_pnl - expected_loss_max)
                print(f"\n❌ ПРОБЛЕМА:")
                print(f"   Фактический убыток (${closed_pnl:.4f}) > Максимум (${expected_loss_max:.2f})")
                print(f"   Превышение: ${excess:.4f}")
                
                # Рассчитываем какой должен был быть SL
                notional = 25.0  # $25 позиция
                risk_usd = 1.0   # -$1 максимум
                risk_pct = (risk_usd / notional) * 100  # -4%
                expected_sl = entry * (1 - risk_pct / 100.0)
                
                print(f"\n🔍 АНАЛИЗ SL:")
                print(f"   Ожидаемый SL: ${expected_sl:.8f} (-4% от входа)")
                print(f"   Фактический выход: ${exit_price:.8f}")
                
                sl_pct = ((entry - exit_price) / entry) * 100
                print(f"   SL процент: {sl_pct:.2f}%")
                
                if exit_price < expected_sl:
                    slippage = ((expected_sl - exit_price) / expected_sl) * 100
                    slippage_usd = (expected_sl - exit_price) * size
                    print(f"\n⚠️ ПРИЧИНЫ ПРЕВЫШЕНИЯ:")
                    print(f"   1. Проскальзывание (slippage): {slippage:.2f}%")
                    print(f"   2. Дополнительный убыток от slippage: ${slippage_usd:.4f}")
                    print(f"   3. Возможно комиссии биржи")
                    print(f"\n💡 РЕКОМЕНДАЦИЯ:")
                    print(f"   Установить SL немного выше (меньше риск), чтобы")
                    print(f"   компенсировать проскальзывание при закрытии")
                else:
                    print(f"   ⚠️ Выход выше ожидаемого SL")
                    print(f"   Возможно SL был установлен неправильно")
                
                # Расчет правильного SL с учетом slippage
                slippage_buffer = 0.1  # 0.1% буфер на проскальзывание
                safe_risk_pct = risk_pct - slippage_buffer  # -3.9% вместо -4%
                safe_sl = entry * (1 - safe_risk_pct / 100.0)
                print(f"\n🔧 РЕКОМЕНДУЕМЫЙ SL (с учетом slippage):")
                print(f"   Безопасный SL: ${safe_sl:.8f} (-{abs(safe_risk_pct):.2f}%)")
                print(f"   Максимальный убыток: ${abs(risk_usd):.2f} + slippage")
        
        # Время
        created_time = latest.get("createdTime", "")
        updated_time = latest.get("updatedTime", "")
        if created_time:
            created_dt = datetime.fromtimestamp(int(created_time) / 1000, tz=pytz.timezone("Europe/Warsaw"))
            print(f"\n⏰ Открыта: {created_dt.strftime('%H:%M:%S %d.%m.%Y')}")
        if updated_time:
            updated_dt = datetime.fromtimestamp(int(updated_time) / 1000, tz=pytz.timezone("Europe/Warsaw"))
            print(f"   Закрыта: {updated_dt.strftime('%H:%M:%S %d.%m.%Y')}")
    else:
        print("\n⚠️ История закрытых позиций пуста")
else:
    ret_msg = closed.get('retMsg', 'Unknown') if closed else 'No response'
    print(f"\n❌ Ошибка получения истории: {ret_msg}")










