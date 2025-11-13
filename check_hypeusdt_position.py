#!/usr/bin/env python3
"""Проверка позиции HYPEUSDT SHORT"""
import sys
sys.path.insert(0, '/opt/bot')

import asyncio
from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(Path("/opt/bot/.env"))
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

async def check_hypeusdt_position():
    """Проверка позиции HYPEUSDT"""
    session = HTTP(api_key=api_key, api_secret=api_secret, testnet=False)
    
    print("="*70)
    print("📊 ПРОВЕРКА ПОЗИЦИИ HYPEUSDT SHORT")
    print("="*70)
    
    # Получаем позицию
    pos = session.get_positions(category="linear", symbol="HYPEUSDT", settleCoin="USDT")
    
    if pos and pos.get("retCode") == 0:
        positions = pos.get("result", {}).get("list", [])
        for p in positions:
            size = float(p.get("size", 0))
            if size > 0:
                symbol = p.get("symbol", "N/A")
                side = p.get("side", "N/A")
                entry = float(p.get("avgPrice", 0))
                mark = float(p.get("markPrice", 0))
                tp_str = p.get("takeProfit", "")
                sl_str = p.get("stopLoss", "")
                leverage = p.get("leverage", "N/A")
                created_time = p.get("createdTime", "")
                
                print(f"\n📊 ПОЗИЦИЯ НА БИРЖЕ:")
                print(f"   Символ: {symbol}")
                print(f"   Направление: {side}")
                print(f"   Вход: ${entry:.5f}")
                print(f"   Текущая: ${mark:.5f}")
                print(f"   Размер: {size}")
                print(f"   Плечо: {leverage}x")
                print(f"   TP на бирже: {tp_str if tp_str else 'НЕ УСТАНОВЛЕН'}")
                print(f"   SL на бирже: {sl_str if sl_str else 'НЕ УСТАНОВЛЕН'}")
                
                # Расчет PnL
                if side == "Sell":
                    pnl_pct = ((entry - mark) / entry) * 100
                    pnl_usd = pnl_pct / 100 * (entry * size)
                else:
                    pnl_pct = ((mark - entry) / entry) * 100
                    pnl_usd = pnl_pct / 100 * (entry * size)
                
                print(f"\n💰 ТЕКУЩИЙ PnL:")
                print(f"   Процент: {pnl_pct:+.2f}%")
                print(f"   USDT: ${pnl_usd:+.2f}")
                
                # Проверка TP/SL
                if entry > 0:
                    print(f"\n🎯 ПРОВЕРКА TP/SL:")
                    
                    # Ожидаемые значения
                    expected_tp_start = entry * 0.9885  # -1.15% для SHORT
                    expected_tp_guaranteed = entry * 0.96  # -4% для гарантированного +$1
                    expected_sl = entry + (1.0 / (entry * size / 25.0)) if size > 0 else entry * 1.04
                    
                    if tp_str:
                        tp_val = float(tp_str)
                        if side == "Sell":
                            tp_pct = ((entry - tp_val) / entry) * 100
                            print(f"   ✅ TP установлен: ${tp_val:.5f}")
                            print(f"   TP процент: +{tp_pct:.2f}%")
                            
                            if tp_pct < 1.0:
                                print(f"   ⚠️ TP слишком мал! Должен быть минимум +1.15% (стартовый)")
                            elif tp_pct < 1.15:
                                print(f"   ⚠️ TP меньше стартового +1.15% (сейчас +{tp_pct:.2f}%)")
                            elif tp_pct >= 1.15 and tp_pct < 4.0:
                                print(f"   ✅ TP в диапазоне трейлинга (+1.15% → +4%)")
                            elif tp_pct >= 4.0:
                                print(f"   ✅ TP на гарантированном уровне (+{tp_pct:.2f}% = +$1+)")
                        else:
                            tp_pct = ((tp_val - entry) / entry) * 100
                            print(f"   ✅ TP установлен: ${tp_val:.5f} (+{tp_pct:.2f}%)")
                    else:
                        print(f"   ❌ TP НЕ УСТАНОВЛЕН на бирже!")
                    
                    if sl_str:
                        sl_val = float(sl_str)
                        if side == "Sell":
                            sl_pct = ((sl_val - entry) / entry) * 100
                            print(f"   ✅ SL установлен: ${sl_val:.5f}")
                            print(f"   SL процент: -{sl_pct:.2f}%")
                            
                            # Проверка что SL = -$1 или BE
                            if sl_pct > 0:
                                print(f"   ⚠️ SL выше входа - возможно уже в BE или трейлится")
                            else:
                                expected_sl_pct = -1.0 / (entry * size / 25.0) * 100 if size > 0 else -4.0
                                if abs(sl_pct - expected_sl_pct) > 1.0:
                                    print(f"   ⚠️ SL отличается от ожидаемого -$1 ({expected_sl_pct:.2f}%)")
                        else:
                            sl_pct = ((entry - sl_val) / entry) * 100
                            print(f"   ✅ SL установлен: ${sl_val:.5f} (-{sl_pct:.2f}%)")
                    else:
                        print(f"   ❌ SL НЕ УСТАНОВЛЕН на бирже!")
                
                # Проверка монитора
                print(f"\n⏰ ИНФОРМАЦИЯ:")
                if created_time:
                    from datetime import datetime
                    import pytz
                    created_dt = datetime.fromtimestamp(int(created_time) / 1000, tz=pytz.timezone('Europe/Warsaw'))
                    print(f"   Открыта: {created_dt.strftime('%H:%M:%S %d.%m.%Y')}")
                
                print(f"\n📋 СЛЕДУЮЩИЕ ШАГИ:")
                print(f"   1. Проверить работает ли monitor_trailing_tp_universal.py")
                print(f"   2. Убедиться что TP трейлится каждые 30 сек")
                print(f"   3. Проверить логику входа (45m+1h+4h подтверждение)")
                
                break
        else:
            print("\n⚠️ Позиция HYPEUSDT не найдена на бирже")
    else:
        print(f"\n❌ Ошибка получения позиций: {pos.get('retMsg', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(check_hypeusdt_position())








