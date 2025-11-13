#!/usr/bin/env python3
"""Тест интеграции TradingView для получения 45m данных"""
import sys
sys.path.insert(0, '/opt/bot')

async def test_tradingview_45m():
    """Тест получения 45m данных из различных источников"""
    import asyncio
    import ccxt
    import pandas as pd
    from super_bot_v4_mtf import SuperBotV4MTF
    
    bot = SuperBotV4MTF()
    await bot.initialize()
    
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print("="*70)
    print("📊 ТЕСТ ПОЛУЧЕНИЯ 45M ДАННЫХ ИЗ РАЗЛИЧНЫХ ИСТОЧНИКОВ")
    print("="*70)
    
    for symbol in test_symbols:
        print(f"\n🔍 Тестируем {symbol}:")
        
        # Метод 1: Bybit (текущий основной)
        try:
            df_bybit = await bot._fetch_ohlcv(symbol, '45m', 50)
            if not df_bybit.empty:
                print(f"   ✅ Bybit: {len(df_bybit)} свечей | Последняя цена: ${df_bybit['close'].iloc[-1]:.2f}")
            else:
                print(f"   ❌ Bybit: пусто")
        except Exception as e:
            print(f"   ❌ Bybit: ошибка - {e}")
        
        # Метод 2: OKX
        try:
            okx = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
            symbol_okx = symbol.replace('USDT', '/USDT:USDT')
            ohlcv_okx = await okx.fetch_ohlcv(symbol_okx, '45m', 50)
            if ohlcv_okx:
                df_okx = pd.DataFrame(ohlcv_okx, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                print(f"   ✅ OKX: {len(df_okx)} свечей | Последняя цена: ${df_okx['close'].iloc[-1]:.2f}")
            else:
                print(f"   ❌ OKX: пусто")
        except Exception as e:
            print(f"   ⚠️ OKX: {e}")
        
        # Метод 3: Синтез из 15m
        try:
            df15 = await bot._fetch_ohlcv(symbol, '15m', 150)
            if not df15.empty:
                df15 = df15.sort_values('timestamp').reset_index(drop=True)
                import numpy as np
                idx = np.arange(len(df15)) // 3
                agg = df15.groupby(idx).agg({
                    'timestamp': 'last',
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index(drop=True)
                print(f"   ✅ Синтез 15m→45m: {len(agg)} свечей | Последняя цена: ${agg['close'].iloc[-1]:.2f}")
            else:
                print(f"   ❌ Синтез: нет данных 15m")
        except Exception as e:
            print(f"   ❌ Синтез: ошибка - {e}")
    
    print("\n" + "="*70)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*70)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tradingview_45m())








