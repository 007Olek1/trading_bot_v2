#!/usr/bin/env python3
import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import requests

load_dotenv(Path("/opt/bot/.env"), override=True)

def get_kline(symbol: str, interval: str, limit: int = 100):
    """Получить свечи с Bybit API"""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if data.get('retCode') == 0:
        klines = data.get('result', {}).get('list', [])
        # Конвертируем в DataFrame
        df_data = []
        for k in klines:
            df_data.append({
                'timestamp': pd.to_datetime(int(k[0]), unit='ms'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    return pd.DataFrame()

def calculate_indicators(df):
    """Рассчитать индикаторы"""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    
    ema_9 = talib.EMA(closes, timeperiod=9)[-1]
    ema_21 = talib.EMA(closes, timeperiod=21)[-1]
    ema_50 = talib.EMA(closes, timeperiod=50)[-1] if len(closes) >= 50 else ema_21
    rsi = talib.RSI(closes, timeperiod=14)[-1]
    macd, macd_signal, _ = talib.MACD(closes)
    macd_val = macd[-1]
    macd_sig = macd_signal[-1]
    bb_upper, bb_middle, bb_lower = talib.BBANDS(closes)
    bb_position = ((closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) * 100 if bb_upper[-1] != bb_lower[-1] else 50
    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
    atr_pct = (atr / closes[-1]) * 100
    avg_volume = np.mean(volumes[-20:])
    current_volume = volumes[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    return {
        'price': closes[-1],
        'ema_9': ema_9,
        'ema_21': ema_21,
        'ema_50': ema_50,
        'rsi': rsi,
        'macd': macd_val,
        'macd_signal': macd_sig,
        'bb_position': bb_position,
        'atr': atr,
        'atr_pct': atr_pct,
        'volume_ratio': volume_ratio
    }

s = HTTP(testnet=False, api_key=os.getenv("BYBIT_API_KEY"), api_secret=os.getenv("BYBIT_API_SECRET"))
pos = s.get_positions(category="linear", settleCoin="USDT").get("result", {}).get("list", [])

print("="*70)
print("📊 АНАЛИЗ ВОЗМОЖНОСТИ ЗАРАБОТКА ПО СИГНАЛАМ")
print("="*70)

for p in pos:
    size = float(p.get("size", 0))
    if size > 0:
        symbol = p.get("symbol", "")
        side = p.get("side", "")
        entry = float(p.get("avgPrice", 0))
        mark = float(p.get("markPrice", 0))
        tp = float(p.get("takeProfit", 0) or 0)
        sl = float(p.get("stopLoss", 0) or 0)
        
        print(f"\n🔹 {symbol} | {side}")
        print(f"   Вход: ${entry:.6f} | Текущая: ${mark:.6f}")
        if side == "Buy":
            pnl_pct = ((mark - entry) / entry) * 100
        else:
            pnl_pct = ((entry - mark) / entry) * 100
        print(f"   PnL: {pnl_pct:+.2f}%")
        if tp > 0:
            tp_pct = ((tp - entry) / entry) * 100 if side == "Buy" else ((entry - tp) / entry) * 100
            tp_dist = ((tp - mark) / mark) * 100 if side == "Buy" else ((mark - tp) / mark) * 100
            print(f"   🎯 TP: ${tp:.6f} (+{tp_pct:.2f}%) | До TP: {tp_dist:.2f}%")
        if sl > 0:
            sl_pct = ((entry - sl) / entry) * 100 if side == "Buy" else ((sl - entry) / entry) * 100
            sl_dist = ((mark - sl) / mark) * 100 if side == "Buy" else ((sl - mark) / mark) * 100
            print(f"   🛑 SL: ${sl:.6f} (-{sl_pct:.2f}%) | До SL: {sl_dist:.2f}%")
        
        # Получаем MTF данные
        print(f"\n📊 MTF АНАЛИЗ:")
        mtf_results = {}
        for tf in ['15m', '30m', '45m', '1h', '4h']:
            try:
                interval_map = {'15m': '15', '30m': '30', '45m': '45', '1h': '60', '4h': '240'}
                interval = interval_map.get(tf, tf)
                df = get_kline(symbol, interval, 100)
                if not df.empty and len(df) > 50:
                    indicators = calculate_indicators(df)
                    mtf_results[tf] = indicators
                    ema_check = (indicators['ema_9'] > indicators['ema_21']) if side == "Buy" else (indicators['ema_9'] < indicators['ema_21'])
                    print(f"   {tf:>4s}: EMA9/21={'✅' if ema_check else '❌'} | RSI={indicators['rsi']:.1f} | BB={indicators['bb_position']:.1f}% | Vol={indicators['volume_ratio']:.2f}x")
                else:
                    # Для 45m пробуем синтезировать из 15m
                    if tf == '45m':
                        try:
                            df15 = get_kline(symbol, '15', 300)
                            if not df15.empty and len(df15) >= 90:
                                # Агрегируем 15m в 45m (3 свечи = 1 свеча 45m)
                                df15 = df15.sort_values('timestamp').reset_index(drop=True)
                                idx = np.arange(len(df15)) // 3
                                agg = df15.groupby(idx).agg({
                                    'timestamp': 'last',
                                    'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last',
                                    'volume': 'sum'
                                }).reset_index(drop=True)
                                if len(agg) > 50:
                                    indicators = calculate_indicators(agg)
                                    mtf_results[tf] = indicators
                                    ema_check = (indicators['ema_9'] > indicators['ema_21']) if side == "Buy" else (indicators['ema_9'] < indicators['ema_21'])
                                    print(f"   {tf:>4s}: EMA9/21={'✅' if ema_check else '❌'} | RSI={indicators['rsi']:.1f} | BB={indicators['bb_position']:.1f}% | Vol={indicators['volume_ratio']:.2f}x (синтез из 15m)")
                        except:
                            print(f"   {tf:>4s}: ❌ Данные не получены")
                    else:
                        print(f"   {tf:>4s}: ❌ Данные не получены")
            except Exception as e:
                print(f"   {tf:>4s}: ❌ Ошибка: {e}")
        
        # Проверка MTF подтверждения
        print(f"\n✅ ПРОВЕРКА MTF ПОДТВЕРЖДЕНИЯ:")
        checks = {}
        
        if '45m' in mtf_results and '1h' in mtf_results and '4h' in mtf_results:
            data_45m = mtf_results['45m']
            data_1h = mtf_results['1h']
            data_4h = mtf_results['4h']
            
            if side == "Buy":
                confirm_45m = data_45m['ema_9'] > data_45m['ema_21']
                confirm_1h = data_1h['ema_9'] > data_1h['ema_21']
                confirm_4h = data_4h['ema_9'] > data_4h['ema_21']
            else:
                confirm_45m = data_45m['ema_9'] < data_45m['ema_21']
                confirm_1h = data_1h['ema_9'] < data_1h['ema_21']
                confirm_4h = data_4h['ema_9'] < data_4h['ema_21']
            
            checks['MTF_45m_1h_4h'] = confirm_45m and confirm_1h and confirm_4h
            print(f"   45m: {'✅' if confirm_45m else '❌'} | 1h: {'✅' if confirm_1h else '❌'} | 4h: {'✅' if confirm_4h else '❌'}")
            print(f"   ИТОГО: {'✅ ПОДТВЕРЖДЕНО' if checks['MTF_45m_1h_4h'] else '❌ НЕ ПОДТВЕРЖДЕНО'}")
        else:
            print(f"   ⚠️ Недостаточно MTF данных для проверки")
        
        if '45m' in mtf_results:
            data_45m = mtf_results['45m']
            checks['ATR'] = data_45m['atr_pct'] >= 1.2
            checks['Volume'] = data_45m['volume_ratio'] >= 1.2
            print(f"\n   ATR (45m): {data_45m['atr_pct']:.2f}% ({'✅' if checks['ATR'] else '❌'})")
            print(f"   Volume (45m): {data_45m['volume_ratio']:.2f}x ({'✅' if checks['Volume'] else '❌'})")
        
        if '15m' in mtf_results and '30m' in mtf_results:
            data_15m = mtf_results['15m']
            data_30m = mtf_results['30m']
            if side == "Buy":
                impulse_15m = data_15m['ema_9'] > data_15m['ema_21']
                impulse_30m = data_30m['ema_9'] > data_30m['ema_21']
            else:
                impulse_15m = data_15m['ema_9'] < data_15m['ema_21']
                impulse_30m = data_30m['ema_9'] < data_30m['ema_21']
            checks['Impulse'] = impulse_15m and impulse_30m
            print(f"   Импульс 15m/30m: {'✅' if checks['Impulse'] else '❌'}")
        
        # Итоговая оценка
        print(f"\n🎯 ИТОГОВАЯ ОЦЕНКА:")
        passed = sum(checks.values())
        total = len(checks)
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        if checks.get('MTF_45m_1h_4h', False):
            print(f"   ✅ MTF подтверждение: ДА")
            print(f"   📊 Возможность заработка: ВЫСОКАЯ")
            if pnl_pct < 0:
                print(f"   ⚠️ ВНИМАНИЕ: Позиция в убытке ({pnl_pct:+.2f}%), но MTF подтвержден - возможен отскок")
        else:
            print(f"   ❌ MTF подтверждение: НЕТ")
            print(f"   📊 Возможность заработка: НИЗКАЯ")
            if pnl_pct < 0:
                print(f"   🚨 КРИТИЧНО: Позиция в убытке ({pnl_pct:+.2f}%) БЕЗ MTF подтверждения!")
            print(f"   ⚠️ БОТ ДОЛЖЕН ЗАКРЫТЬ ПОЗИЦИЮ (нет подтверждения 45m+1h+4h)")
        
        print(f"   Пройдено проверок: {passed}/{total} ({pass_rate:.0f}%)")
        
        print("="*70)

