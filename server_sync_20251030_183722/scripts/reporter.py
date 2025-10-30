#!/usr/bin/env python3
import os
import requests
from datetime import datetime

# load env files
for path in ('/opt/bot/.env','/opt/bot/api.env'):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k,v=line.split('=',1)
                os.environ.setdefault(k.strip(), v.strip())

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID') or os.getenv('TG_CHAT_ID')


def tg_send(text: str):
    if not (BOT_TOKEN and CHAT_ID):
        print('No telegram creds; printing report\n'+text)
        return False
    try:
        r = requests.post('https://api.telegram.org/bot'+BOT_TOKEN+'/sendMessage',
                          json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=20)
        print('Telegram status:', r.status_code)
        return r.ok
    except Exception as e:
        print('Telegram send error:', e)
        return False

# positions via pybit (optional)
positions=[]
try:
    from pybit.unified_trading import HTTP
    api_key=os.getenv('BYBIT_API_KEY') or os.getenv('API_KEY')
    api_secret=os.getenv('BYBIT_API_SECRET') or os.getenv('API_SECRET')
    if api_key and api_secret:
        s=HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=15)
        r=s.get_positions(category='linear', settleCoin='USDT', limit=200)
        positions=r.get('result',{}).get('list',[]) or []
except Exception:
    positions=[]

openp=[p for p in positions if float(p.get('size') or 0)>0]
upnl_sum=0.0
for p in openp:
    try:
        upnl_sum += float(p.get('unrealisedPnl') or 0)
    except Exception:
        pass

now=datetime.now().strftime('%d.%m.%Y %H:%M:%S')
lines=[
    f"📊 Авто-отчёт | {now}",
    f"Открытых позиций: {len(openp)}",
    f"Суммарный uPnL: {upnl_sum:.2f} USDT",
]

for p in openp[:10]:
    try:
        sym=p.get('symbol'); side=p.get('side'); size=float(p.get('size') or 0)
        entry=float(p.get('avgPrice') or 0); mark=float(p.get('markPrice') or 0)
        tp=p.get('takeProfit') or '-'; sl=p.get('stopLoss') or '-'
        upnl=float(p.get('unrealisedPnl') or 0)
        lines.append(f"• {sym} {side} size={size} uPnL={upnl:.2f} | entry={entry} mark={mark} TP={tp} SL={sl}")
    except Exception:
        continue

report='\n'.join(lines)

tg_send(report)


