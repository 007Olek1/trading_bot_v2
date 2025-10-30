#!/usr/bin/env python3
import os
from pybit.unified_trading import HTTP

def main():
    api_key=os.getenv("BYBIT_API_KEY") or os.getenv("API_KEY")
    api_secret=os.getenv("BYBIT_API_SECRET") or os.getenv("API_SECRET")
    if not api_key or not api_secret:
        print('No API keys'); return
    s=HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=15)
    sym='JASMYUSDT'
    p=s.get_positions(category='linear', symbol=sym)
    pos=(p.get('result',{}).get('list',[]) or [None])[0]
    if not pos:
        print('No open position for', sym); return
    print('symbol=', pos.get('symbol'))
    print('side=', pos.get('side'))
    print('size=', pos.get('size'))
    print('entry(avgPrice)=', pos.get('avgPrice'))
    print('mark=', pos.get('markPrice'))
    print('takeProfit=', pos.get('takeProfit'))
    print('stopLoss=', pos.get('stopLoss'))

if __name__ == '__main__':
    main()


