#!/usr/bin/env python3
import os, re
from statistics import mean
from pybit.unified_trading import HTTP

keys={}
for p in ('/opt/bot/api.env','/opt/bot/.env'):
    if os.path.isfile(p):
        for line in open(p,'r',encoding='utf-8',errors='ignore'):
            line=line.strip()
            if not line or line.startswith('#'):
                continue
            m=re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
            if not m:
                continue
            k,v=m.group(1), m.group(2)
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v=v[1:-1]
            keys[k]=v
api_key=keys.get('BYBIT_API_KEY') or keys.get('API_KEY')
api_secret=keys.get('BYBIT_API_SECRET') or keys.get('API_SECRET')
S=HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=30)

r=S.get_closed_pnl(category='linear', settleCoin='USDT', limit=200)
rows=(r.get('result',{}) or {}).get('list',[])
seen=set(); pairs=[]
for x in rows:
    s=x.get('symbol'); ex_side=str(x.get('side') or '').lower()
    if not s or s in seen:
        continue
    orig='sell' if ex_side=='buy' else 'buy'
    pairs.append((s, orig))
    seen.add(s)
    if len(pairs)>=100:
        break

if not pairs:
    print('NO_INPUT'); raise SystemExit(0)

NOTIONAL=25.0; TP_USD=1.0; SL_USD=1.0; STEP=0.005; MAX_TARGET=0.05; BASE_PCT=TP_USD/NOTIONAL
intervals=[15,30,45,60,240]

def run_bt(interval):
    out=[]
    for sym, side in pairs:
        try:
            k=S.get_kline(category='linear', symbol=sym, interval=interval, limit=600)
            kl=(k.get('result',{}) or {}).get('list',[])
            if not kl:
                out.append((sym,side,None)); continue
            bars=[[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in kl]
            entry=bars[-2][3]
            risk_pct=SL_USD/NOTIONAL
            sl= entry*(1+risk_pct) if side=='sell' else entry*(1-risk_pct)
            best=BASE_PCT
            tp= entry*(1-best) if side=='sell' else entry*(1+best)
            pnl=None
            for o,h,l,c in bars[-300:]:
                if side=='sell':
                    if l<=tp:
                        while best+STEP<=MAX_TARGET and l <= entry*(1-(best+STEP)):
                            best+=STEP
                        pnl= round(best*NOTIONAL, 4); break
                    if h>=sl:
                        pnl= -SL_USD; break
                else:
                    if h>=tp:
                        while best+STEP<=MAX_TARGET and h >= entry*(1+(best+STEP)):
                            best+=STEP
                        pnl= round(best*NOTIONAL, 4); break
                    if l<=sl:
                        pnl= -SL_USD; break
            if pnl is None:
                pnl=0.0
            out.append((sym,side,pnl))
        except Exception:
            out.append((sym,side,None))
    valid=[p for _,_,p in out if p is not None]
    wins=sum(1 for p in valid if p>0); losses=sum(1 for p in valid if p<0)
    hit=(wins/len(valid)*100 if valid else 0.0)
    sump=round(sum(valid),4) if valid else 0.0
    avgp=round(mean(valid),4) if valid else 0.0
    print(f"{interval}m: tested={len(out)} valid={len(valid)} hit={hit:.2f}% sum={sump} avg={avgp}")

for iv in intervals:
    run_bt(iv)
