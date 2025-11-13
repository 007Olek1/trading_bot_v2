#!/usr/bin/env python3
"""
Скрипт для добавления SL/TP используя методы класса бота
"""

import asyncio
import sys
import os

sys.path.insert(0, '/opt/bot')
os.chdir('/opt/bot')
from dotenv import load_dotenv
load_dotenv('/opt/bot/api.env')

from super_bot_v4_mtf import SuperBotV4MTF

POSITION_SIZE = 5.0
LEVERAGE = 5
MAX_STOP_LOSS_USD = 5.0
TP_PERCENT = 20.0


async def main():
    try:
        # Создаем экземпляр бота
        bot = SuperBotV4MTF()
        await bot.initialize()
        
        print("🔍 Получаем открытые позиции...")
        print("=" * 70)
        
        # Получаем позиции через биржу бота
        positions = await bot.exchange.fetch_positions(params={'category': 'linear'})
        open_positions = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
        
        print(f"📊 Найдено открытых позиций: {len(open_positions)}\n")
        
        if not open_positions:
            print("✅ Нет открытых позиций")
            return
        
        for pos in open_positions:
            symbol = pos.get('symbol', '')
            side = pos.get('side', '').lower()
            entry_price = float(pos.get('entryPrice', 0) or pos.get('entry', 0))
            size = float(pos.get('contracts', 0) or pos.get('size', 0))
            
            print(f"\n📌 Обрабатываем {symbol} {side.upper()}:")
            print(f"   Вход: ${entry_price:.8f}")
            print(f"   Размер: {size}")
            
            # Рассчитываем SL и TP
            position_notional = POSITION_SIZE * LEVERAGE
            stop_loss_percent = (MAX_STOP_LOSS_USD / position_notional) * 100
            
            if side in ['long', 'buy']:
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100.0)
                tp_price = entry_price * (1 + TP_PERCENT / 100.0)
                direction = 'buy'
            else:
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100.0)
                tp_price = entry_price * (1 - TP_PERCENT / 100.0)
                direction = 'sell'
            
            print(f"   🛑 SL: ${stop_loss_price:.8f} (-${MAX_STOP_LOSS_USD:.2f} максимум)")
            print(f"   🎯 TP: ${tp_price:.8f} (+{TP_PERCENT:.0f}%)")
            
            # Используем новый метод бота для добавления SL/TP к существующей позиции
            print(f"\n   ⚙️ Устанавливаем Stop Loss и Take Profit...")
            
            try:
                # Используем специальный метод для существующих позиций
                success = await bot.add_sl_tp_to_existing_position(
                    symbol=symbol,
                    side=direction,
                    entry_price=entry_price
                )
                
                if success:
                    print(f"      ✅ SL и TP установлены!")
                else:
                    print(f"      ⚠️ SL/TP не установлены, контролируются через мониторинг бота")
            except Exception as e:
                print(f"      ❌ Ошибка: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("✅ Обработка завершена!")
        
        # Закрываем соединение
        if bot.exchange:
            await bot.exchange.close()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

