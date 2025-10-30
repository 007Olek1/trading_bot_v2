#!/usr/bin/env python3
import os
from pybit.unified_trading import HTTP

def main():
    api_key=os.getenv("BYBIT_API_KEY") or os.getenv("API_KEY")
    api_secret=os.getenv("BYBIT_API_SECRET") or os.getenv("API_SECRET")
    if not api_key or not api_secret:
        print('No API keys'); return
    s=HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=15)
    symbol='JASMYUSDT'
    r=s.get_positions(category='linear', settleCoin='USDT', limit=200)
    lst=r.get('result',{}).get('list',[]) or []
    pos=None
    for x in lst:
        try:
            if x.get('symbol')==symbol and float(x.get('size') or 0)>0:
                pos=x; break
        except Exception:
            pass
    if not pos:
        print('No open JASMYUSDT position'); return
    side=pos.get('side')
    entry=float(pos.get('avgPrice') or 0)
    mark=float(pos.get('markPrice') or 0)
    notional=25.0
    risk_usd=1.0
    risk_pct=(risk_usd/notional)
    if side=='Sell':
        tp_price=mark*0.99
        sl_price=entry*(1+risk_pct)
    else:
        tp_price=mark*1.01
        sl_price=entry*(1-risk_pct)
    u_pnl=float(pos.get('unrealisedPnl') or 0)
    be_buffer=0.001
    if u_pnl>=1.0:
        if side=='Sell':
            sl_price=entry*(1+be_buffer)
        else:
            sl_price=entry*(1-be_buffer)
    resp=s.set_trading_stop(category='linear', symbol=symbol, takeProfit=f"{tp_price:.8f}", stopLoss=f"{sl_price:.8f}", tpslMode='Full', positionIdx=0)
    print('set_trading_stop retCode=', resp.get('retCode'), 'tp=', f"{tp_price:.8f}", 'sl=', f"{sl_price:.8f}", 'side=', side)

if __name__ == '__main__':
    main()


