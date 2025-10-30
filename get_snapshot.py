#!/usr/bin/env python3
import os
from pybit.unified_trading import HTTP

def main():
    api_key=os.getenv("BYBIT_API_KEY") or os.getenv("API_KEY")
    api_secret=os.getenv("BYBIT_API_SECRET") or os.getenv("API_SECRET")
    if not api_key or not api_secret:
        print('No API keys'); return
    S=HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=15)
    # Balance
    wb=S.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    lst=(wb.get('result',{}).get('list',[]) or [{}])
    coin=(lst[0].get('coin',[]) or [{}])[0]
    wallet=coin.get('walletBalance','?')
    avail=coin.get('availableToWithdraw','?')
    print(f"Баланс: {wallet} USDT (свободно: {avail})")
    # Positions
    plist=S.get_positions(category='linear', settleCoin='USDT', limit=200).get('result',{}).get('list',[]) or []
    open_positions=[p for p in plist if float(p.get('size') or 0)>0]
    print(f"Открытых позиций: {len(open_positions)}")
    for p in open_positions:
        sym=p.get('symbol')
        side=p.get('side')
        size=p.get('size')
        entry=p.get('avgPrice')
        mark=p.get('markPrice')
        tp=p.get('takeProfit')
        sl=p.get('stopLoss')
        upnl=p.get('unrealisedPnl')
        print(f"• {sym} {side} size={size} | entry={entry} mark={mark} uPnL={upnl} TP={tp} SL={sl}")

if __name__ == '__main__':
    main()


