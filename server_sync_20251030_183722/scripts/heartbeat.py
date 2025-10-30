#!/usr/bin/env python3
import os
import requests
from datetime import datetime

ENV_FILE = '/opt/bot/.env'
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        line=line.strip()
        if line and '=' in line and not line.startswith('#'):
            k,v=line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip())

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID') or os.getenv('TG_CHAT_ID')

def tg(text):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post('https://api.telegram.org/bot'+BOT_TOKEN+'/sendMessage', json={chat_id: CHAT_ID, text: text}, timeout=10)
    except Exception:
        pass

now = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
tg('HEARTBEAT: bot alive ' + now)
