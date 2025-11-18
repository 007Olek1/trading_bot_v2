#!/usr/bin/env python3
"""
Проверка истории сделок на бирже
"""

from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

client = HTTP(
    testnet=False,
    api_key=os.getenv('BYBIT_API_KEY'),
    api_secret=os.getenv('BYBIT_API_SECRET')
)

print('='*80)
print('📊 ИСТОРИЯ СДЕЛОК ЗА ПОСЛЕДНИЕ 10 ДНЕЙ')
print('='*80)
print()

try:
    # Получаем закрытые позиции
    response = client.get_closed_pnl(
        category='linear',
        limit=50
    )
    
    if response['retCode'] == 0:
        trades = response['result']['list']
        
        if not trades:
            print('📭 Нет закрытых сделок за последние 10 дней')
        else:
            print(f'📈 Найдено сделок: {len(trades)}')
            print()
            
            total_pnl = 0
            winning = 0
            losing = 0
            
            for i, trade in enumerate(trades[:20], 1):  # Показываем последние 20
                symbol = trade['symbol']
                pnl = float(trade['closedPnl'])
                side = trade['side']
                qty = float(trade['qty'])
                entry = float(trade['avgEntryPrice'])
                exit_price = float(trade['avgExitPrice'])
                created = datetime.fromtimestamp(int(trade['createdTime'])/1000)
                updated = datetime.fromtimestamp(int(trade['updatedTime'])/1000)
                
                duration = updated - created
                hours = duration.total_seconds() / 3600
                
                total_pnl += pnl
                if pnl > 0:
                    winning += 1
                    emoji = '✅'
                else:
                    losing += 1
                    emoji = '❌'
                
                print(f'{i:2d}. {emoji} {symbol} {side}')
                print(f'    💰 PnL: ${pnl:.2f}')
                print(f'    📊 Вход: ${entry:.4f} → Выход: ${exit_price:.4f}')
                print(f'    📦 Объём: {qty}')
                print(f'    ⏱️  Время: {created.strftime("%d.%m %H:%M")} → {updated.strftime("%d.%m %H:%M")} ({hours:.1f}ч)')
                print()
            
            print('='*80)
            print('📊 СТАТИСТИКА')
            print('='*80)
            print(f'💰 Общий PnL: ${total_pnl:.2f}')
            print(f'✅ Прибыльных: {winning}')
            print(f'❌ Убыточных: {losing}')
            if winning + losing > 0:
                win_rate = (winning / (winning + losing)) * 100
                print(f'📈 Win Rate: {win_rate:.1f}%')
                avg_pnl = total_pnl / (winning + losing)
                print(f'📊 Средний PnL: ${avg_pnl:.2f}')
    else:
        print(f'❌ Ошибка: {response.get("retMsg")}')
        
except Exception as e:
    print(f'❌ Ошибка получения истории: {e}')
    import traceback
    traceback.print_exc()
