#!/usr/bin/env python3
"""
Глубокий анализ паттернов торговли
"""

from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from collections import defaultdict

load_dotenv()

client = HTTP(
    testnet=False,
    api_key=os.getenv('BYBIT_API_KEY'),
    api_secret=os.getenv('BYBIT_API_SECRET')
)

print('='*80)
print('🔍 ГЛУБОКИЙ АНАЛИЗ ПАТТЕРНОВ ТОРГОВЛИ')
print('='*80)
print()

try:
    response = client.get_closed_pnl(
        category='linear',
        limit=50
    )
    
    if response['retCode'] == 0:
        trades = response['result']['list']
        
        # Структуры для анализа
        by_symbol = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []})
        by_side = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0})
        by_duration = {'quick': [], 'short': [], 'medium': [], 'long': []}
        
        all_wins = []
        all_losses = []
        
        for trade in trades:
            symbol = trade['symbol']
            pnl = float(trade['closedPnl'])
            side = trade['side']
            qty = float(trade['qty'])
            entry = float(trade['avgEntryPrice'])
            exit_price = float(trade['avgExitPrice'])
            created = datetime.fromtimestamp(int(trade['createdTime'])/1000)
            updated = datetime.fromtimestamp(int(trade['updatedTime'])/1000)
            
            duration_hours = (updated - created).total_seconds() / 3600
            
            # Анализ по символам
            by_symbol[symbol]['count'] += 1
            by_symbol[symbol]['total_pnl'] += pnl
            by_symbol[symbol]['trades'].append({
                'pnl': pnl,
                'duration': duration_hours,
                'side': side,
                'entry': entry,
                'exit': exit_price
            })
            
            if pnl > 0:
                by_symbol[symbol]['wins'] += 1
                all_wins.append(pnl)
            else:
                by_symbol[symbol]['losses'] += 1
                all_losses.append(pnl)
            
            # Анализ по направлению
            by_side[side]['count'] += 1
            by_side[side]['total_pnl'] += pnl
            if pnl > 0:
                by_side[side]['wins'] += 1
            else:
                by_side[side]['losses'] += 1
            
            # Анализ по длительности
            trade_data = {'pnl': pnl, 'symbol': symbol, 'duration': duration_hours}
            if duration_hours < 0.5:
                by_duration['quick'].append(trade_data)
            elif duration_hours < 4:
                by_duration['short'].append(trade_data)
            elif duration_hours < 12:
                by_duration['medium'].append(trade_data)
            else:
                by_duration['long'].append(trade_data)
        
        # ═══════════════════════════════════════════════════════════════
        # 1. АНАЛИЗ ПО СИМВОЛАМ
        # ═══════════════════════════════════════════════════════════════
        print('📊 1. АНАЛИЗ ПО МОНЕТАМ')
        print('-'*80)
        
        sorted_symbols = sorted(by_symbol.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for symbol, data in sorted_symbols[:10]:
            win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
            avg_pnl = data['total_pnl'] / data['count']
            
            status = '✅' if data['total_pnl'] > 0 else '❌'
            print(f'{status} {symbol:15s} | Сделок: {data["count"]:2d} | Win Rate: {win_rate:5.1f}% | PnL: ${data["total_pnl"]:+6.2f} | Avg: ${avg_pnl:+5.2f}')
        
        print()
        
        # ═══════════════════════════════════════════════════════════════
        # 2. АНАЛИЗ ПО НАПРАВЛЕНИЮ (LONG/SHORT)
        # ═══════════════════════════════════════════════════════════════
        print('📊 2. АНАЛИЗ ПО НАПРАВЛЕНИЮ')
        print('-'*80)
        
        for side, data in by_side.items():
            win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
            avg_pnl = data['total_pnl'] / data['count']
            direction = 'LONG' if side == 'Buy' else 'SHORT'
            
            print(f'{direction:6s} | Сделок: {data["count"]:2d} | Win Rate: {win_rate:5.1f}% | PnL: ${data["total_pnl"]:+6.2f} | Avg: ${avg_pnl:+5.2f}')
        
        print()
        
        # ═══════════════════════════════════════════════════════════════
        # 3. АНАЛИЗ ПО ДЛИТЕЛЬНОСТИ
        # ═══════════════════════════════════════════════════════════════
        print('📊 3. АНАЛИЗ ПО ДЛИТЕЛЬНОСТИ УДЕРЖАНИЯ')
        print('-'*80)
        
        duration_labels = {
            'quick': 'Быстрые (<30мин)',
            'short': 'Короткие (30м-4ч)',
            'medium': 'Средние (4ч-12ч)',
            'long': 'Долгие (>12ч)'
        }
        
        for key, label in duration_labels.items():
            trades_list = by_duration[key]
            if trades_list:
                count = len(trades_list)
                wins = sum(1 for t in trades_list if t['pnl'] > 0)
                total_pnl = sum(t['pnl'] for t in trades_list)
                avg_pnl = total_pnl / count
                win_rate = (wins / count * 100) if count > 0 else 0
                
                status = '✅' if total_pnl > 0 else '❌'
                print(f'{status} {label:20s} | Сделок: {count:2d} | Win Rate: {win_rate:5.1f}% | PnL: ${total_pnl:+6.2f} | Avg: ${avg_pnl:+5.2f}')
        
        print()
        
        # ═══════════════════════════════════════════════════════════════
        # 4. СТАТИСТИКА ПРИБЫЛЕЙ И УБЫТКОВ
        # ═══════════════════════════════════════════════════════════════
        print('📊 4. СТАТИСТИКА ПРИБЫЛЕЙ И УБЫТКОВ')
        print('-'*80)
        
        if all_wins:
            avg_win = sum(all_wins) / len(all_wins)
            max_win = max(all_wins)
            print(f'✅ Прибыльные сделки: {len(all_wins)}')
            print(f'   Средняя прибыль: ${avg_win:.2f}')
            print(f'   Максимальная прибыль: ${max_win:.2f}')
        
        print()
        
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            max_loss = min(all_losses)
            print(f'❌ Убыточные сделки: {len(all_losses)}')
            print(f'   Средний убыток: ${avg_loss:.2f}')
            print(f'   Максимальный убыток: ${max_loss:.2f}')
        
        print()
        
        # ═══════════════════════════════════════════════════════════════
        # 5. РЕКОМЕНДАЦИИ
        # ═══════════════════════════════════════════════════════════════
        print('='*80)
        print('💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ')
        print('='*80)
        print()
        
        # Анализ Win Rate
        total_trades = len(all_wins) + len(all_losses)
        overall_win_rate = (len(all_wins) / total_trades * 100) if total_trades > 0 else 0
        
        print('1. 📊 ОБЩАЯ ПРОИЗВОДИТЕЛЬНОСТЬ')
        print(f'   Win Rate: {overall_win_rate:.1f}% (целевой: >50%)')
        if overall_win_rate < 50:
            print('   ⚠️  ПРОБЛЕМА: Низкий Win Rate')
            print('   ✅ РЕШЕНИЕ: Увеличить минимальную уверенность для входа')
            print('   📝 Текущая: 65% → Рекомендуемая: 70-75%')
        print()
        
        # Анализ Risk/Reward
        if all_wins and all_losses:
            avg_win = sum(all_wins) / len(all_wins)
            avg_loss = abs(sum(all_losses) / len(all_losses))
            rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            print('2. 💰 RISK/REWARD RATIO')
            print(f'   Текущий R/R: {rr_ratio:.2f}:1')
            print(f'   Средняя прибыль: ${avg_win:.2f}')
            print(f'   Средний убыток: ${avg_loss:.2f}')
            
            if rr_ratio < 1.5:
                print('   ⚠️  ПРОБЛЕМА: Низкий R/R (прибыли не покрывают убытки)')
                print('   ✅ РЕШЕНИЕ: Увеличить минимальную цель прибыли')
                print('   📝 Текущая: 10% → Рекомендуемая: 15-20%')
            print()
        
        # Анализ быстрых закрытий
        quick_trades = by_duration['quick']
        if quick_trades:
            quick_losses = sum(1 for t in quick_trades if t['pnl'] < 0)
            quick_loss_rate = (quick_losses / len(quick_trades) * 100)
            
            print('3. ⚡ БЫСТРЫЕ ЗАКРЫТИЯ (<30 минут)')
            print(f'   Количество: {len(quick_trades)}')
            print(f'   Убыточных: {quick_losses} ({quick_loss_rate:.1f}%)')
            
            if quick_loss_rate > 60:
                print('   ⚠️  ПРОБЛЕМА: Слишком много быстрых убытков (SL срабатывает рано)')
                print('   ✅ РЕШЕНИЕ: Уже исправлено! Trailing SL теперь активируется после +2%')
                print('   📝 Это должно снизить количество преждевременных закрытий')
            print()
        
        # Анализ по направлению
        print('4. 📈 НАПРАВЛЕНИЕ СДЕЛОК')
        for side, data in by_side.items():
            direction = 'LONG' if side == 'Buy' else 'SHORT'
            win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
            
            if win_rate < 40:
                print(f'   ⚠️  {direction}: Win Rate {win_rate:.1f}% - слишком низкий')
                print(f'   ✅ РЕШЕНИЕ: Усилить фильтрацию {direction} сигналов')
        print()
        
        # Рекомендуемые параметры
        print('='*80)
        print('⚙️  РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ')
        print('='*80)
        print()
        print('config.py:')
        print('-'*80)
        
        # Определяем рекомендации на основе анализа
        if overall_win_rate < 45:
            rec_confidence = 0.75
        elif overall_win_rate < 50:
            rec_confidence = 0.70
        else:
            rec_confidence = 0.65
        
        if rr_ratio < 1.5:
            rec_target = 15.0
        elif rr_ratio < 2.0:
            rec_target = 12.0
        else:
            rec_target = 10.0
        
        print(f'SIGNAL_THRESHOLDS = {{')
        print(f'    "min_confidence": {rec_confidence},  # Минимальная уверенность (текущая: 0.65)')
        print(f'    "strong_confidence": 0.85,  # Сильный сигнал')
        print(f'    "min_volume_ratio": 1.5,  # Минимальное отношение объема (увеличено с 1.3)')
        print(f'    "max_spread_percent": 0.3,')
        print(f'}}')
        print()
        print(f'MIN_PROFIT_TARGET_PERCENT = {rec_target}  # Минимальная цель (текущая: 10.0%)')
        print()
        print(f'TRAILING_SL_ACTIVATION_PERCENT = 2.5  # Активация после прибыли (текущая: 2.0%)')
        print(f'TRAILING_SL_CALLBACK_PERCENT = 2.0    # Откат от максимума (текущая: 1.5%)')
        print()
        print(f'MIN_TIMEFRAME_ALIGNMENT = 4  # Минимум таймфреймов для подтверждения (текущая: 3)')
        print()
        
    else:
        print(f'❌ Ошибка: {response.get("retMsg")}')
        
except Exception as e:
    print(f'❌ Ошибка: {e}')
    import traceback
    traceback.print_exc()
