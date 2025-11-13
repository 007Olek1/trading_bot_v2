#!/usr/bin/env python3
import os
import datetime as dt
from pybit.unified_trading import HTTP

def main():
    api_key=os.getenv("BYBIT_API_KEY") or os.getenv("API_KEY")
    api_secret=os.getenv("BYBIT_API_SECRET") or os.getenv("API_SECRET")
    if not api_key or not api_secret:
        print("No API keys"); return
    s=HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=15)
    sym='JASMYUSDT'
    # Closed PnL за последние ~36 часов
    now=dt.datetime.utcnow()
    start=int((now - dt.timedelta(hours=36)).timestamp()*1000)
    try:
        pnl=s.get_closed_pnl(category='linear', symbol=sym, startTime=start, limit=100)
        lst=pnl.get('result',{}).get('list',[]) or []
        print('ClosedPnL count:', len(lst))
        for x in lst[-3:]:
            print('ClosedPnL:', {k:x.get(k) for k in ['symbol','side','avgEntryPrice','avgExitPrice','closedPnl','updatedTime','orderType','execType']})
    except Exception as e:
        print('closed_pnl error:', e)
    # Executions (fills) последние 50
    try:
        exc=s.get_execution_list(category='linear', symbol=sym, limit=50)
        el=exc.get('result',{}).get('list',[]) or []
        print('Exec count:', len(el))
        for x in el[:5]:
            print('Exec:', {k:x.get(k) for k in ['symbol','side','execQty','execPrice','execType','orderType','stopOrderType','execTime']})
    except Exception as e:
        print('exec error:', e)

if __name__ == '__main__':
    main()


