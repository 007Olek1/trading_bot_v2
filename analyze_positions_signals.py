#!/usr/bin/env python3
"""
Анализ текущих позиций и оценка возможности заработка согласно сигналам
"""
import os
import sys
import asyncio
import ccxt
from datetime import datetime
import pytz
from dotenv import load_dotenv
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import talib

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WARSAW_TZ = pytz.timezone('Europe/Warsaw')

# Загружаем переменные окружения
env_file = Path("/opt/bot/.env")
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv()

async def analyze_position_profitability(exchange, symbol: str, side: str, entry_price: float, current_price: float):
    """Анализ возможности заработка для позиции"""
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 АНАЛИЗ ПРИБЫЛЬНОСТИ: {symbol} {side.upper()}")
        logger.info(f"{'='*70}")
        
        # Получаем MTF данные
        timeframes = ['15m', '30m', '45m', '1h', '4h']
        mtf_data = {}
        
        for tf in timeframes:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, tf, limit=100)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Рассчитываем индикаторы
                    closes = df['close'].values
                    highs = df['high'].values
                    lows = df['low'].values
                    volumes = df['volume'].values
                    
                    # EMA
                    ema_9 = talib.EMA(closes, timeperiod=9)[-1]
                    ema_21 = talib.EMA(closes, timeperiod=21)[-1]
                    ema_50 = talib.EMA(closes, timeperiod=50)[-1]
                    
                    # RSI
                    rsi = talib.RSI(closes, timeperiod=14)[-1]
                    
                    # MACD
                    macd, macd_signal, macd_hist = talib.MACD(closes)
                    macd_val = macd[-1]
                    macd_sig = macd_signal[-1]
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(closes)
                    bb_position = ((closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) * 100 if bb_upper[-1] != bb_lower[-1] else 50
                    
                    # ATR
                    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                    atr_pct = (atr / closes[-1]) * 100
                    
                    # Volume Ratio
                    avg_volume = np.mean(volumes[-20:])
                    current_volume = volumes[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    mtf_data[tf] = {
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
            except Exception as e:
                logger.debug(f"⚠️ Ошибка получения данных {tf}: {e}")
                continue
        
        # Анализируем условия входа и текущее состояние
        logger.info(f"\n💰 ЦЕНОВЫЕ УРОВНИ:")
        logger.info(f"   Вход: ${entry_price:.6f}")
        logger.info(f"   Текущая: ${current_price:.6f}")
        if side == 'Buy':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        logger.info(f"   PnL: {pnl_pct:+.2f}%")
        
        # Проверка MTF подтверждения
        logger.info(f"\n📊 MTF ПОДТВЕРЖДЕНИЕ:")
        checks = {
            'MTF_1h_4h_confirm': False,
            'MTF_45m_confirm': False,
            'ATR_min': False,
            'Volume_min': False,
            '15m_30m_impulse': False
        }
        
        if '1h' in mtf_data and '4h' in mtf_data:
            data_1h = mtf_data['1h']
            data_4h = mtf_data['4h']
            
            if side == 'Buy':
                confirm_1h = data_1h['ema_9'] > data_1h['ema_21']
                confirm_4h = data_4h['ema_9'] > data_4h['ema_21']
            else:
                confirm_1h = data_1h['ema_9'] < data_1h['ema_21']
                confirm_4h = data_4h['ema_9'] < data_4h['ema_21']
            
            checks['MTF_1h_4h_confirm'] = confirm_1h and confirm_4h
            logger.info(f"   ✅ 1h+4h: {'✅ ДА' if checks['MTF_1h_4h_confirm'] else '❌ НЕТ'}")
            logger.info(f"      1h: EMA9={data_1h['ema_9']:.6f} vs EMA21={data_1h['ema_21']:.6f} ({'✅' if confirm_1h else '❌'})")
            logger.info(f"      4h: EMA9={data_4h['ema_9']:.6f} vs EMA21={data_4h['ema_21']:.6f} ({'✅' if confirm_4h else '❌'})")
        
        if '45m' in mtf_data:
            data_45m = mtf_data['45m']
            if side == 'Buy':
                confirm_45m = data_45m['ema_9'] > data_45m['ema_21']
            else:
                confirm_45m = data_45m['ema_9'] < data_45m['ema_21']
            checks['MTF_45m_confirm'] = confirm_45m
            logger.info(f"   ✅ 45m: {'✅ ДА' if checks['MTF_45m_confirm'] else '❌ НЕТ'}")
        
        # Проверка ATR
        if '45m' in mtf_data:
            data_45m = mtf_data['45m']
            atr_pct = data_45m.get('atr_pct', 0)
            checks['ATR_min'] = atr_pct >= 1.2
            logger.info(f"   ✅ ATR (45m): {atr_pct:.2f}% ({'✅ >= 1.2%' if checks['ATR_min'] else '❌ < 1.2%'})")
        
        # Проверка Volume
        if '45m' in mtf_data:
            data_45m = mtf_data['45m']
            vol_ratio = data_45m.get('volume_ratio', 0)
            checks['Volume_min'] = vol_ratio >= 1.2
            logger.info(f"   ✅ Volume Ratio (45m): {vol_ratio:.2f}x ({'✅ >= 1.2x' if checks['Volume_min'] else '❌ < 1.2x'})")
        
        # Проверка импульса 15m/30m
        if '15m' in mtf_data and '30m' in mtf_data:
            data_15m = mtf_data['15m']
            data_30m = mtf_data['30m']
            
            if side == 'Buy':
                impulse_15m = data_15m['ema_9'] > data_15m['ema_21']
                impulse_30m = data_30m['ema_9'] > data_30m['ema_21']
            else:
                impulse_15m = data_15m['ema_9'] < data_15m['ema_21']
                impulse_30m = data_30m['ema_9'] < data_30m['ema_21']
            
            checks['15m_30m_impulse'] = impulse_15m and impulse_30m
            logger.info(f"   ✅ Импульс 15m/30m: {'✅ ДА' if checks['15m_30m_impulse'] else '❌ НЕТ'}")
        
        # Показываем все индикаторы
        logger.info(f"\n📈 ИНДИКАТОРЫ ПО ТАЙМФРЕЙМАМ:")
        for tf in ['15m', '30m', '45m', '1h', '4h']:
            if tf in mtf_data:
                data = mtf_data[tf]
                ema_check = (data['ema_9'] > data['ema_21']) if side == 'Buy' else (data['ema_9'] < data['ema_21'])
                logger.info(f"   {tf:>4s}: RSI={data['rsi']:.1f} | EMA9/21={'✅' if ema_check else '❌'} | BB={data['bb_position']:.1f}% | Vol={data['volume_ratio']:.2f}x")
        
        # Оценка вероятности достижения TP
        logger.info(f"\n🎯 ОЦЕНКА ВЕРОЯТНОСТИ ДОСТИЖЕНИЯ TP:")
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        pass_rate = (passed_checks / total_checks) * 100
        
        base_probability = 50
        if side == 'Buy' and '45m' in mtf_data:
            rsi = mtf_data['45m']['rsi']
            if 30 < rsi < 70:
                base_probability += 10
            elif rsi < 30:
                base_probability += 15
        
        probability_bonus = 0
        if checks['MTF_1h_4h_confirm']:
            probability_bonus += 15
        if checks['MTF_45m_confirm']:
            probability_bonus += 10
        if checks['ATR_min']:
            probability_bonus += 5
        if checks['Volume_min']:
            probability_bonus += 5
        if checks['15m_30m_impulse']:
            probability_bonus += 5
        
        final_probability = min(95, base_probability + probability_bonus)
        
        logger.info(f"   Пройдено проверок: {passed_checks}/{total_checks} ({pass_rate:.0f}%)")
        logger.info(f"   Базовая вероятность: {base_probability}%")
        logger.info(f"   Бонусы за проверки: +{probability_bonus}%")
        logger.info(f"   🎯 ИТОГОВАЯ ВЕРОЯТНОСТЬ ДОСТИЖЕНИЯ TP: ~{final_probability}%")
        
        # Оценка возможности заработка
        logger.info(f"\n💰 ОЦЕНКА ВОЗМОЖНОСТИ ЗАРАБОТКА:")
        if pnl_pct >= 0:
            logger.info(f"   ✅ Позиция в прибыли ({pnl_pct:+.2f}%)")
        else:
            logger.info(f"   ⚠️ Позиция в убытке ({pnl_pct:+.2f}%)")
        
        if pass_rate >= 80:
            logger.info(f"   ✅ ВЫСОКАЯ - Большинство проверок пройдено ({pass_rate:.0f}%)")
        elif pass_rate >= 60:
            logger.info(f"   ⚠️ СРЕДНЯЯ - Часть проверок не пройдена ({pass_rate:.0f}%)")
        else:
            logger.info(f"   ❌ НИЗКАЯ - Многие проверки не пройдены ({pass_rate:.0f}%)")
        
        if final_probability >= 75:
            logger.info(f"   ✅ Высокая вероятность достижения TP (~{final_probability}%)")
        elif final_probability >= 60:
            logger.info(f"   ⚠️ Средняя вероятность достижения TP (~{final_probability}%)")
        else:
            logger.info(f"   ❌ Низкая вероятность достижения TP (~{final_probability}%)")
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа {symbol}: {e}", exc_info=True)

async def main():
    try:
        # Инициализация биржи
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear', 'accountType': 'UNIFIED'}
        })
        
        # Получаем открытые позиции
        import asyncio as asyncio_check
        if asyncio_check.iscoroutinefunction(exchange.fetch_positions):
            positions = await exchange.fetch_positions(params={'category': 'linear'})
        else:
            positions = exchange.fetch_positions(params={'category': 'linear'})
        open_positions = [p for p in positions if (p.get('contracts', 0) or p.get('size', 0)) > 0]
        
        if not open_positions:
            logger.info("📊 Открытых позиций не найдено")
            return
        
        logger.info(f"📊 Найдено открытых позиций: {len(open_positions)}\n")
        
        # Анализируем каждую позицию
        for pos in open_positions:
            symbol = pos.get('symbol', '')
            side = pos.get('side', '').lower()
            entry_price = float(pos.get('entryPrice', 0))
            current_price = float(pos.get('markPrice', 0))
            
            if side == 'buy':
                side_normalized = 'Buy'
            elif side == 'sell':
                side_normalized = 'Sell'
            else:
                side_normalized = side
            
            await analyze_position_profitability(exchange, symbol, side_normalized, entry_price, current_price)
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ АНАЛИЗ ЗАВЕРШЕН")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())










