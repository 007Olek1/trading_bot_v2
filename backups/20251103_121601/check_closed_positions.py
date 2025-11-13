#!/usr/bin/env python3
import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
load_dotenv(Path("/opt/bot/.env"), override=True)
s = HTTP(testnet=False, api_key=os.getenv("BYBIT_API_KEY"), api_secret=os.getenv("BYBIT_API_SECRET"))
closed = s.get_closed_pnl(category="linear", symbol="AIAUSDT", limit=1).get("result", {}).get("list", [])
if closed:
    p = closed[0]
    created_ms = int(p.get("createdTime", 0))
    updated_ms = int(p.get("updatedTime", 0))
    created_dt = datetime.fromtimestamp(created_ms / 1000) if created_ms > 0 else None
    updated_dt = datetime.fromtimestamp(updated_ms / 1000) if updated_ms > 0 else None
    pnl = float(p.get("closedPnl", 0))
    entry = float(p.get("avgEntryPrice", 0))
    exit_price = float(p.get("avgExitPrice", 0))
    side = p.get("side", "")
    reason = p.get("closeType", "UNKNOWN")
    print(f"{side} | Entry: ${entry:.6f} | Exit: ${exit_price:.6f} | PnL: ${pnl:.2f}")
    if created_dt:
        fmt_str = "%Y-%m-%d %H:%M:%S"
        print(f"Создана: {created_dt.strftime(fmt_str)}")
    if updated_dt:
        fmt_str = "%Y-%m-%d %H:%M:%S"
        print(f"Закрыта: {updated_dt.strftime(fmt_str)}")
    print(f"Причина: {reason}")
else:
    print("Закрытых позиций AIAUSDT не найдено")




