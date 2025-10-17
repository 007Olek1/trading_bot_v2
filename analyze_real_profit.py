#!/usr/bin/env python3
"""
Анализ реальной прибыли с учётом комиссий
"""
import asyncio
from bot_v2_exchange import ExchangeManager
from datetime import datetime, timedelta

async def analyze_real_profit():
    em = ExchangeManager()
    await em.connect()
    
    # Получаем историю сделок за последние 24 часа
    since = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
    
    try:
        trades = await em.exchange.fetch_my_trades(since=since, limit=200)
        
        print("="*70)
        print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ СДЕЛОК")
        print("="*70)
        print(f"\n💰 Всего сделок: {len(trades)}")
        
        # Группируем по символам и считаем комиссии
        symbols = {}
        total_fees = 0
        
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in symbols:
                symbols[symbol] = {
                    "buy_amount": 0, 
                    "sell_amount": 0, 
                    "fees": 0,
                    "trades": 0
                }
            
            side = trade["side"]
            amount = float(trade["amount"])
            price = float(trade["price"])
            fee = float(trade.get("fee", {}).get("cost", 0))
            
            symbols[symbol]["trades"] += 1
            symbols[symbol]["fees"] += fee
            total_fees += fee
            
            if side == "buy":
                symbols[symbol]["buy_amount"] += amount * price
            else:
                symbols[symbol]["sell_amount"] += amount * price
        
        # Считаем реальный PnL с учётом комиссий
        real_pnl = 0
        closed_positions = []
        
        for symbol, data in symbols.items():
            if data["buy_amount"] > 0 and data["sell_amount"] > 0:
                gross_pnl = data["sell_amount"] - data["buy_amount"]
                net_pnl = gross_pnl - data["fees"]
                real_pnl += net_pnl
                
                closed_positions.append({
                    "symbol": symbol,
                    "gross_pnl": gross_pnl,
                    "fees": data["fees"],
                    "net_pnl": net_pnl,
                    "trades": data["trades"]
                })
        
        print(f"📊 Закрытых позиций: {len(closed_positions)}")
        print(f"💸 Общие комиссии: ${total_fees:.2f}")
        print(f"💰 ЧИСТАЯ ПРИБЫЛЬ: ${real_pnl:.2f}")
        print()
        
        # Сортируем по чистой прибыли
        closed_positions.sort(key=lambda x: x["net_pnl"], reverse=True)
        
        print("📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        for pos in closed_positions:
            emoji = "✅" if pos["net_pnl"] > 0 else "❌"
            trades_count = pos["trades"]
            print(f"{emoji} {pos['symbol']:20} | "
                  f"Gross: ${pos['gross_pnl']:+.2f} | "
                  f"Fees: ${pos['fees']:.2f} | "
                  f"Net: ${pos['net_pnl']:+.2f} | "
                  f"Trades: {trades_count}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    await em.disconnect()

if __name__ == "__main__":
    asyncio.run(analyze_real_profit())
