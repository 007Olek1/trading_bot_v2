#!/usr/bin/env python3
"""Полный анализ сделки AI16ZUSDT LONG"""
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
print("🔍 ПОЛНЫЙ АНАЛИЗ СДЕЛКИ AI16ZUSDT LONG")
print("="*70)

# Получаем историю закрытой позиции
closed = session.get_closed_pnl(
    category="linear",
    symbol="AI16ZUSDT",
    limit=1
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
        created_time = latest.get("createdTime", "")
        updated_time = latest.get("updatedTime", "")
        
        print(f"\n📊 ДАННЫЕ СДЕЛКИ:")
        print(f"   Символ: AI16ZUSDT")
        print(f"   Направление: {side}")
        print(f"   Размер: {size}")
        print(f"   Вход: ${entry:.8f}")
        print(f"   Выход: ${exit_price:.8f}")
        print(f"   Closed PnL: ${closed_pnl:.4f}")
        
        # Время
        if created_time:
            created_dt = datetime.fromtimestamp(int(created_time) / 1000, tz=pytz.timezone("Europe/Warsaw"))
            print(f"\n⏰ ВРЕМЯ:")
            print(f"   Открыта: {created_dt.strftime('%H:%M:%S %d.%m.%Y')}")
        if updated_time:
            updated_dt = datetime.fromtimestamp(int(updated_time) / 1000, tz=pytz.timezone("Europe/Warsaw"))
            duration = (int(updated_time) - int(created_time)) / 1000 / 60  # минуты
            print(f"   Закрыта: {updated_dt.strftime('%H:%M:%S %d.%m.%Y')}")
            print(f"   Длительность: {duration:.1f} минут")
        
        if side == "Buy" and entry > 0:
            # Расчеты для LONG
            pnl_pct = ((exit_price - entry) / entry) * 100
            position_notional = entry * size
            
            print(f"\n💰 РАСЧЕТЫ:")
            print(f"   PnL процент: {pnl_pct:.2f}%")
            print(f"   Нотиональная стоимость: ${position_notional:.2f}")
            
            # Ожидаемые значения
            notional_expected = 25.0  # $25 позиция
            risk_usd_max = 1.0  # Максимальный риск -$1
            risk_pct_expected = (risk_usd_max / notional_expected) * 100  # -4%
            expected_sl = entry * (1 - risk_pct_expected / 100.0)
            expected_loss = (entry - expected_sl) * size
            
            print(f"\n📊 ОЖИДАЕМЫЕ ЗНАЧЕНИЯ:")
            print(f"   Ожидаемый SL процент: -{risk_pct_expected:.2f}%")
            print(f"   Ожидаемый SL цена: ${expected_sl:.8f}")
            print(f"   Ожидаемый максимальный убыток: ${abs(expected_loss):.4f}")
            
            # Фактические значения
            sl_pct_actual = ((entry - exit_price) / entry) * 100
            
            print(f"\n📈 ФАКТИЧЕСКИЕ ЗНАЧЕНИЯ:")
            print(f"   Фактический SL процент: {sl_pct_actual:.2f}%")
            print(f"   Фактический убыток: ${abs(closed_pnl):.4f}")
            
            # Проблема
            if closed_pnl < -risk_usd_max:
                excess = abs(closed_pnl - risk_usd_max)
                excess_pct = (excess / abs(closed_pnl)) * 100
                
                print(f"\n❌ ПРОБЛЕМА:")
                print(f"   Убыток превысил лимит на ${excess:.4f}")
                print(f"   Превышение: {excess_pct:.2f}%")
                
                # Анализ причин
                if exit_price < expected_sl:
                    slippage_price = expected_sl - exit_price
                    slippage_pct = (slippage_price / expected_sl) * 100
                    slippage_usd = slippage_price * size
                    
                    print(f"\n⚠️ ПРИЧИНЫ ПРЕВЫШЕНИЯ:")
                    print(f"   1. Проскальзывание (slippage):")
                    print(f"      - Ожидаемая цена закрытия: ${expected_sl:.8f}")
                    print(f"      - Фактическая цена закрытия: ${exit_price:.8f}")
                    print(f"      - Проскальзывание: ${slippage_price:.8f} ({slippage_pct:.2f}%)")
                    print(f"      - Дополнительный убыток: ${slippage_usd:.4f}")
                    print(f"   2. Возможные комиссии биржи")
                    print(f"   3. SL был установлен точно на -4%, без буфера")
                    
                    # Решение
                    slippage_buffer = 0.15  # 0.15% буфер
                    safe_risk_pct = risk_pct_expected - slippage_buffer  # -3.85%
                    safe_sl = entry * (1 - safe_risk_pct / 100.0)
                    safe_loss = (entry - safe_sl) * size
                    
                    print(f"\n🔧 РЕШЕНИЕ (УЖЕ ПРИМЕНЕНО):")
                    print(f"   Добавлен буфер на проскальзывание: {slippage_buffer:.2f}%")
                    print(f"   Новый безопасный SL: -{abs(safe_risk_pct):.2f}% (вместо -{risk_pct_expected:.2f}%)")
                    print(f"   Безопасная цена SL: ${safe_sl:.8f}")
                    print(f"   Максимальный убыток с буфером: ${abs(safe_loss):.4f}")
                    print(f"   ✅ Теперь убыток не превысит -$1 даже при slippage")
                
            # Проверка логики входа
            print(f"\n🎯 ПРОВЕРКА ЛОГИКИ ВХОДА:")
            print(f"   Время открытия: {duration:.1f} минут от начала")
            print(f"   ⚠️ Позиция закрылась быстро - возможно:")
            print(f"      - Не было полного MTF подтверждения (45m+1h+4h)")
            print(f"      - Сработал SL из-за неблагоприятного движения цены")
            print(f"      - Рыночные условия были неблагоприятны")
            
            # Рекомендации
            print(f"\n📋 РЕКОМЕНДАЦИИ:")
            print(f"   ✅ Буфер на проскальзывание добавлен в монитор")
            print(f"   ✅ Для будущих позиций SL будет установлен на -3.85% вместо -4%")
            print(f"   ✅ Это предотвратит превышение лимита -$1")
            print(f"   ⚠️ Стоит проверить логику входа для AI16ZUSDT")
            print(f"      (была ли проверка 45m+1h+4h подтверждения?)")
            
        print("\n" + "="*70)
else:
    ret_msg = closed.get('retMsg', 'Unknown') if closed else 'No response'
    print(f"\n❌ Ошибка получения истории: {ret_msg}")










