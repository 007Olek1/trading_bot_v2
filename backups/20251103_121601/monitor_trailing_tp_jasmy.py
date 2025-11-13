#!/usr/bin/env python3
import os, time, math
from pybit.unified_trading import HTTP

SYMBOL = 'JASMYUSDT'
POSITION_NOTIONAL = 25.0
BASE_TARGET = 1.0  # старт +1%
STEP = 0.5         # шаг 0.5%
MAX_TARGET = 5.0   # до +5%
INTERVAL_SEC = 60

def calc_profit_pct(entry: float, mark: float, side: str) -> float:
    if entry <= 0 or mark <= 0:
        return 0.0
    if side == 'Sell':
        return (entry - mark) / entry * 100.0
    return (mark - entry) / entry * 100.0

def main():
    api_key=os.getenv('BYBIT_API_KEY') or os.getenv('API_KEY')
    api_secret=os.getenv('BYBIT_API_SECRET') or os.getenv('API_SECRET')
    if not api_key or not api_secret:
        print('No API keys'); return
    s=HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=15)
    last_target_applied = None
    while True:
        try:
            r=s.get_positions(category='linear', symbol=SYMBOL)
            pos=(r.get('result',{}).get('list',[]) or [None])[0]
            if not pos or float(pos.get('size') or 0) <= 0:
                print('No open position; sleeping...')
                time.sleep(INTERVAL_SEC)
                continue
            side=pos.get('side')
            entry=float(pos.get('avgPrice') or 0)
            mark=float(pos.get('markPrice') or 0)
            profit_pct = calc_profit_pct(entry, mark, side)
            # вычисляем целевой таргет
            if profit_pct < BASE_TARGET:
                target = BASE_TARGET
            else:
                steps = math.floor((profit_pct - BASE_TARGET) / STEP)
                target = min(BASE_TARGET + steps * STEP, MAX_TARGET)
            # Не уменьшать уже выставленный таргет
            if last_target_applied is not None and target <= last_target_applied:
                time.sleep(INTERVAL_SEC)
                continue
            if side == 'Sell':
                tp_price = entry * (1 - target/100.0)
            else:
                tp_price = entry * (1 + target/100.0)
            resp=s.set_trading_stop(category='linear', symbol=SYMBOL, takeProfit=f"{tp_price:.8f}", tpslMode='Full', positionIdx=0)
            print(f"Applied trailing TP target={target:.2f}% -> tp={tp_price:.8f} retCode={resp.get('retCode')}")
            last_target_applied = target
        except Exception as e:
            print('Error:', e)
        time.sleep(INTERVAL_SEC)

if __name__ == '__main__':
    main()


