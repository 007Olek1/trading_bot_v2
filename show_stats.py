#!/usr/bin/env python3
"""
📊 ПОКАЗАТЬ СТАТИСТИКУ СДЕЛОК
"""

import json
from pathlib import Path
from datetime import datetime
from trade_logger import TradeLogger

def main():
    trade_logger = TradeLogger(logs_dir="logs")
    
    # Статистика
    stats = trade_logger.get_stats()
    
    print("="*80)
    print("📊 СТАТИСТИКА ТОРГОВЛИ")
    print("="*80)
    print()
    
    print(f"📈 Всего сделок: {stats['total_trades']}")
    print(f"✅ Прибыльных: {stats['wins']}")
    print(f"❌ Убыточных: {stats['losses']}")
    print(f"📊 Win Rate: {stats['win_rate']:.1f}%")
    print()
    
    print(f"💰 Общий P&L: ${stats['total_pnl_usd']:.2f}")
    print(f"📊 Средняя прибыль: ${stats['avg_win']:.2f}")
    print(f"📊 Средний убыток: ${stats['avg_loss']:.2f}")
    print(f"🏆 Лучшая сделка: ${stats['best_trade']:.2f}")
    print(f"💔 Худшая сделка: ${stats['worst_trade']:.2f}")
    print()
    
    if 'avg_duration_minutes' in stats:
        print(f"⏱️ Средняя длительность: {stats['avg_duration_minutes']:.0f} минут")
    
    print(f"🕐 Обновлено: {stats['last_updated']}")
    print()
    
    # Последние сделки
    closed_trades = trade_logger.get_closed_trades(limit=10)
    
    if closed_trades:
        print("="*80)
        print("📋 ПОСЛЕДНИЕ 10 СДЕЛОК")
        print("="*80)
        print()
        
        for i, trade in enumerate(closed_trades, 1):
            result = "✅" if trade['pnl_usd'] > 0 else "❌"
            mode = "⚡" if trade['mode'] == 'scalping' else "📊"
            
            print(f"{i}. {result} {mode} {trade['symbol']} {trade['direction']}")
            print(f"   Entry: ${trade['entry_price']:.4f} → Exit: ${trade['exit_price']:.4f}")
            print(f"   P&L: ${trade['pnl_usd']:.2f} ({trade['pnl_percent']:+.2f}%) | ROE: {trade['roe_percent']:+.1f}%")
            print(f"   Duration: {trade['duration_minutes']:.0f}m | Reason: {trade['close_reason']}")
            print()
    
    # Открытые позиции
    open_trades = trade_logger.get_open_trades()
    
    if open_trades:
        print("="*80)
        print("🔓 ОТКРЫТЫЕ ПОЗИЦИИ")
        print("="*80)
        print()
        
        for trade in open_trades:
            mode = "⚡" if trade['mode'] == 'scalping' else "📊"
            open_time = datetime.fromisoformat(trade['open_time'])
            duration = (datetime.now(open_time.tzinfo) - open_time).total_seconds() / 60
            
            print(f"{mode} {trade['symbol']} {trade['direction']}")
            print(f"   Entry: ${trade['entry_price']:.4f} | Leverage: {trade['leverage']}x")
            print(f"   Confidence: {trade['confidence']:.0%} | Duration: {duration:.0f}m")
            print()
    
    print("="*80)

if __name__ == "__main__":
    main()
