"""
📊 АНАЛИЗ ПРОБЛЕМ С ПРИБЫЛЬЮ
Детальный разбор почему прибыль такая маленькая
"""

import re
from datetime import datetime
from collections import defaultdict

def analyze_trades_from_log():
    """Анализ сделок из лога"""
    
    trades = []
    trailing_stops = []
    
    # Паттерны для поиска
    position_opened = re.compile(r'ПОЗИЦИЯ ОТКРЫТА.*?💎 ([\w/]+).*?Entry: \$([\d.]+).*?Инвестировано: \$([\d.]+)', re.DOTALL)
    position_closed = re.compile(r'ПОЗИЦИЯ ЗАКРЫТА.*?💎 ([\w/]+).*?P&L: \$([-\d.]+)', re.DOTALL)
    trailing_stop = re.compile(r'TRAILING STOP.*?💎 ([\w/]+).*?Прибыль: \+([\d.]+)%.*?Новый SL: \$([\d.]+)', re.DOTALL)
    
    print("="*80)
    print("🔍 АНАЛИЗ ПРОБЛЕМ С ПРИБЫЛЬЮ")
    print("="*80)
    
    # Читаем лог файл
    try:
        with open('logs/bot_v2.log', 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        print("❌ Не могу прочитать лог файл")
        return
    
    # Находим открытые позиции
    opened_matches = position_opened.findall(content)
    for match in opened_matches:
        symbol, entry, invested = match
        trades.append({
            'symbol': symbol,
            'entry': float(entry),
            'invested': float(invested),
            'status': 'opened'
        })
    
    # Находим закрытые позиции
    closed_matches = position_closed.findall(content)
    for match in closed_matches:
        symbol, pnl = match
        trades.append({
            'symbol': symbol,
            'pnl': float(pnl),
            'status': 'closed'
        })
    
    # Находим trailing stops
    trailing_matches = trailing_stop.findall(content)
    for match in trailing_matches:
        symbol, profit_pct, new_sl = match
        trailing_stops.append({
            'symbol': symbol,
            'profit_pct': float(profit_pct),
            'new_sl': float(new_sl)
        })
    
    print(f"\n📊 СТАТИСТИКА СДЕЛОК:")
    print(f"   Открыто позиций: {len(opened_matches)}")
    print(f"   Закрыто позиций: {len(closed_matches)}")
    print(f"   Trailing Stop событий: {len(trailing_matches)}")
    
    # Анализ закрытых сделок
    if closed_matches:
        print(f"\n💰 ЗАКРЫТЫЕ СДЕЛКИ:")
        total_pnl = 0
        wins = 0
        losses = 0
        
        for match in closed_matches:
            symbol, pnl = match
            pnl_float = float(pnl)
            total_pnl += pnl_float
            
            if pnl_float > 0:
                wins += 1
                emoji = "✅"
            else:
                losses += 1
                emoji = "❌"
            
            print(f"   {emoji} {symbol:20} PnL: ${pnl_float:+.2f}")
        
        print(f"\n   📈 Итого PnL: ${total_pnl:.2f}")
        print(f"   ✅ Прибыльных: {wins}")
        print(f"   ❌ Убыточных: {losses}")
        if wins + losses > 0:
            win_rate = wins / (wins + losses) * 100
            print(f"   📊 Win Rate: {win_rate:.1f}%")
    
    # Анализ Trailing Stop
    if trailing_matches:
        print(f"\n🎯 АНАЛИЗ TRAILING STOP:")
        
        # Группируем по символам
        symbol_trailing = defaultdict(list)
        for ts in trailing_matches:
            symbol_trailing[ts['symbol']].append(ts)
        
        print(f"\n   Сделок с Trailing Stop: {len(symbol_trailing)}")
        
        for symbol, events in symbol_trailing.items():
            print(f"\n   💎 {symbol}:")
            for i, event in enumerate(events, 1):
                print(f"      {i}. Прибыль: +{event['profit_pct']}% → SL: ${event['new_sl']:.4f}")
            
            # Анализ проблемы
            if len(events) > 1:
                max_profit = max(e['profit_pct'] for e in events)
                print(f"      ⚠️ ПРОБЛЕМА: Максимальная прибыль была +{max_profit:.1f}%")
                print(f"         но Trailing Stop сработал слишком рано!")
    
    # Анализ открытых позиций
    if opened_matches:
        print(f"\n📊 АНАЛИЗ ИНВЕСТИЦИЙ:")
        
        symbol_investments = defaultdict(list)
        for match in opened_matches:
            symbol, entry, invested = match
            symbol_investments[symbol].append(float(invested))
        
        for symbol, investments in symbol_investments.items():
            total_invested = sum(investments)
            count = len(investments)
            avg = total_invested / count
            
            print(f"   💎 {symbol:20} Сделок: {count} | Инвестировано: ${total_invested:.2f} | Средний: ${avg:.2f}")
            
            if count > 1:
                print(f"      ⚠️ ПРОБЛЕМА: {count} сделки по одной монете - cooldown не работает!")
    
    # ВЫВОДЫ И РЕКОМЕНДАЦИИ
    print(f"\n{'='*80}")
    print(f"🚨 НАЙДЕННЫЕ ПРОБЛЕМЫ:")
    print(f"{'='*80}")
    
    problems = []
    
    # 1. Маленькие инвестиции
    if opened_matches:
        avg_investment = sum(float(m[2]) for m in opened_matches) / len(opened_matches)
        if avg_investment < 5:
            problems.append(f"1. ❌ МАЛЕНЬКИЕ ИНВЕСТИЦИИ: Средний размер ${avg_investment:.2f}")
            print(f"\n   💡 РЕШЕНИЕ: Увеличить размер позиций для ТОП монет до $10-20")
    
    # 2. Trailing Stop слишком агрессивный
    if trailing_matches:
        early_stops = [ts for ts in trailing_matches if ts['profit_pct'] < 10]
        if len(early_stops) > len(trailing_matches) * 0.5:
            problems.append(f"2. ❌ TRAILING STOP СЛИШКОМ АГРЕССИВНЫЙ: {len(early_stops)} из {len(trailing_matches)} сработали при <10% прибыли")
            print(f"\n   💡 РЕШЕНИЕ: Убрать Trailing Stop до достижения минимум 10% прибыли")
    
    # 3. Повторные сделки
    if opened_matches:
        symbol_counts = defaultdict(int)
        for match in opened_matches:
            symbol_counts[match[0]] += 1
        
        repeated = {s: c for s, c in symbol_counts.items() if c > 1}
        if repeated:
            problems.append(f"3. ❌ ПОВТОРНЫЕ СДЕЛКИ: {len(repeated)} монет торговались несколько раз")
            print(f"\n   💡 РЕШЕНИЕ: Увеличить cooldown до 24 часов")
    
    # 4. Убыточные сделки
    if closed_matches:
        loss_trades = [m for m in closed_matches if float(m[1]) < 0]
        if loss_trades:
            total_loss = sum(float(m[1]) for m in loss_trades)
            problems.append(f"4. ❌ УБЫТОЧНЫЕ СДЕЛКИ: {len(loss_trades)} сделок, убыток ${total_loss:.2f}")
            print(f"\n   💡 РЕШЕНИЕ: Ужесточить фильтры сигналов (95% → 98% уверенность)")
    
    if not problems:
        print("\n   ✅ Критических проблем не найдено")
    
    print(f"\n{'='*80}")
    print(f"📋 РЕКОМЕНДАЦИИ:")
    print(f"{'='*80}")
    print(f"\n1. 💰 УВЕЛИЧИТЬ РАЗМЕР ПОЗИЦИЙ:")
    print(f"   - BTC, ETH, BNB, SOL: $20-30 на сделку")
    print(f"   - Остальные ТОП-50: $10-15 на сделку")
    print(f"\n2. 🎯 ИЗМЕНИТЬ TRAILING STOP:")
    print(f"   - НЕ активировать до +10% прибыли")
    print(f"   - При +10-15%: SL на +5% от входа")
    print(f"   - При +15-20%: SL на +8% от входа")
    print(f"   - При +20%+: SL на +12% от входа")
    print(f"\n3. ⏰ УВЕЛИЧИТЬ COOLDOWN:")
    print(f"   - С 12 часов до 24 часов")
    print(f"   - Блокировка по symbol + side")
    print(f"\n4. 📊 УЖЕСТОЧИТЬ ФИЛЬТРЫ:")
    print(f"   - Минимальная уверенность: 98%")
    print(f"   - Минимум 5 индикаторов в согласии")
    print(f"   - Только монеты с объёмом >$10M/24h")
    print(f"\n5. ⏰ ЖДАТЬ ЗАКРЫТИЯ СВЕЧИ:")
    print(f"   - Анализировать только закрытые свечи")
    print(f"   - Не открывать на неподтверждённых сигналах")

if __name__ == "__main__":
    analyze_trades_from_log()

