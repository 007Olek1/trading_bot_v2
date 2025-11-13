#!/usr/bin/env python3
"""
Анализ логики входа для проверки гарантии заработка в текущих сделках
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

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WARSAW_TZ = pytz.timezone('Europe/Warsaw')

# Загружаем переменные окружения
env_file = Path("/opt/bot/.env")
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv()

async def analyze_position_entry(exchange, symbol: str):
    """Анализ логики входа для позиции"""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 АНАЛИЗ ВХОДА: {symbol}")
        logger.info(f"{'='*60}")
        
        # Получаем текущую позицию
        import asyncio
        if asyncio.iscoroutinefunction(exchange.fetch_positions):
            positions = await exchange.fetch_positions([symbol], params={'category': 'linear'})
        else:
            positions = exchange.fetch_positions([symbol], params={'category': 'linear'})
        position = None
        for pos in positions:
            size = pos.get('contracts', 0) or pos.get('size', 0)
            if size > 0:
                position = pos
                break
        
        if not position:
            logger.warning(f"⚠️ Позиция {symbol} не найдена")
            return
        
        entry_price = float(position.get('entryPrice', 0))
        side = position.get('side', '').lower()
        mark_price = float(position.get('markPrice', 0))
        current_pnl = float(position.get('unrealisedPnl', 0))
        current_pnl_pct = ((mark_price - entry_price) / entry_price * 100) if side == 'buy' else ((entry_price - mark_price) / entry_price * 100)
        
        logger.info(f"💰 Вход: ${entry_price:.6f} | Текущая: ${mark_price:.6f}")
        logger.info(f"📈 PnL: ${current_pnl:.2f} ({current_pnl_pct:.2f}%)")
        
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
                    import talib
                    import numpy as np
                    
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
                    current_price = closes[-1]
                    bb_position = ((current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) * 100
                    
                    # ATR
                    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                    atr_pct = (atr / current_price) * 100
                    
                    # Volume Ratio
                    avg_volume = np.mean(volumes[-20:])
                    current_volume = volumes[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    mtf_data[tf] = {
                        'price': current_price,
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
        
        # Анализируем условия входа
        logger.info(f"\n📊 MTF АНАЛИЗ:")
        logger.info(f"{'='*60}")
        
        checks = {
            'MTF_1h_4h_confirm': False,
            'ATR_min': False,
            'Volume_min': False,
            '15m_30m_impulse': False
        }
        
        # Проверка 1: MTF подтверждение 1h + 4h
        if '1h' in mtf_data and '4h' in mtf_data:
            data_1h = mtf_data['1h']
            data_4h = mtf_data['4h']
            
            if side == 'buy':
                confirm_1h = data_1h['ema_9'] > data_1h['ema_21']
                confirm_4h = data_4h['ema_9'] > data_4h['ema_21']
                checks['MTF_1h_4h_confirm'] = confirm_1h and confirm_4h
            else:
                confirm_1h = data_1h['ema_9'] < data_1h['ema_21']
                confirm_4h = data_4h['ema_9'] < data_4h['ema_21']
                checks['MTF_1h_4h_confirm'] = confirm_1h and confirm_4h
            
            logger.info(f"✅ MTF подтверждение 1h+4h: {'✅ ДА' if checks['MTF_1h_4h_confirm'] else '❌ НЕТ'}")
            logger.info(f"   1h: EMA9={data_1h['ema_9']:.6f} vs EMA21={data_1h['ema_21']:.6f} ({'✅' if confirm_1h else '❌'})")
            logger.info(f"   4h: EMA9={data_4h['ema_9']:.6f} vs EMA21={data_4h['ema_21']:.6f} ({'✅' if confirm_4h else '❌'})")
        
        # Проверка 2: ATR минимум
        if '45m' in mtf_data:
            data_45m = mtf_data['45m']
            atr_pct = data_45m.get('atr_pct', 0)
            checks['ATR_min'] = atr_pct >= 1.2
            logger.info(f"✅ ATR (45m): {atr_pct:.2f}% ({'✅ >= 1.2%' if checks['ATR_min'] else '❌ < 1.2%'})")
        
        # Проверка 3: Volume минимум
        if '45m' in mtf_data:
            data_45m = mtf_data['45m']
            vol_ratio = data_45m.get('volume_ratio', 0)
            checks['Volume_min'] = vol_ratio >= 1.2
            logger.info(f"✅ Volume Ratio (45m): {vol_ratio:.2f}x ({'✅ >= 1.2x' if checks['Volume_min'] else '❌ < 1.2x'})")
        
        # Проверка 4: Импульс на 15m/30m
        if '15m' in mtf_data and '30m' in mtf_data:
            data_15m = mtf_data['15m']
            data_30m = mtf_data['30m']
            
            if side == 'buy':
                impulse_15m = data_15m['ema_9'] > data_15m['ema_21']
                impulse_30m = data_30m['ema_9'] > data_30m['ema_21']
            else:
                impulse_15m = data_15m['ema_9'] < data_15m['ema_21']
                impulse_30m = data_30m['ema_9'] < data_30m['ema_21']
            
            checks['15m_30m_impulse'] = impulse_15m and impulse_30m
            logger.info(f"✅ Импульс 15m/30m: {'✅ ДА' if checks['15m_30m_impulse'] else '❌ НЕТ'}")
            logger.info(f"   15m: EMA9={data_15m['ema_9']:.6f} vs EMA21={data_15m['ema_21']:.6f} ({'✅' if impulse_15m else '❌'})")
            logger.info(f"   30m: EMA9={data_30m['ema_9']:.6f} vs EMA21={data_30m['ema_21']:.6f} ({'✅' if impulse_30m else '❌'})")
        
        # Показываем все индикаторы по таймфреймам
        logger.info(f"\n📈 ИНДИКАТОРЫ ПО ТАЙМФРЕЙМАМ:")
        logger.info(f"{'='*60}")
        for tf in ['15m', '30m', '45m', '1h', '4h']:
            if tf in mtf_data:
                data = mtf_data[tf]
                logger.info(f"{tf:>4s}: RSI={data['rsi']:.1f} | EMA9/21={'✅' if (data['ema_9'] > data['ema_21'] if side == 'buy' else data['ema_9'] < data['ema_21']) else '❌'} | BB={data['bb_position']:.1f}% | Vol={data['volume_ratio']:.2f}x")
        
        # Итоговая оценка
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        pass_rate = (passed_checks / total_checks) * 100
        
        logger.info(f"\n🎯 ИТОГОВАЯ ОЦЕНКА:")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Пройдено проверок: {passed_checks}/{total_checks} ({pass_rate:.0f}%)")
        
        if pass_rate == 100:
            logger.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ - ВХОД КОРРЕКТНЫЙ")
        elif pass_rate >= 75:
            logger.info("⚠️ БОЛЬШИНСТВО ПРОВЕРОК ПРОЙДЕНО - ВХОД УСЛОВНО КОРРЕКТНЫЙ")
        else:
            logger.warning("❌ МНОГИЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ - ВХОД РИСКОВАННЫЙ")
        
        # Прогноз вероятности прибыли на основе текущих условий
        logger.info(f"\n🔮 ПРОГНОЗ ВЕРОЯТНОСТИ ПРИБЫЛИ:")
        logger.info(f"{'='*60}")
        
        # TP/SL уровни
        tp_price = float(position.get('takeProfit', 0))
        sl_price = float(position.get('stopLoss', 0))
        
        if tp_price > 0 and sl_price > 0:
            if side == 'buy':
                tp_distance = ((tp_price - mark_price) / mark_price) * 100
                sl_distance = ((mark_price - sl_price) / mark_price) * 100
            else:
                tp_distance = ((mark_price - tp_price) / mark_price) * 100
                sl_distance = ((sl_price - mark_price) / mark_price) * 100
            
            logger.info(f"🎯 TP: ${tp_price:.6f} (расстояние: {tp_distance:.2f}%)")
            logger.info(f"🛑 SL: ${sl_price:.6f} (расстояние: {sl_distance:.2f}%)")
            
            # Вероятность достижения TP на основе текущих индикаторов
            probability_bonus = 0
            if checks['MTF_1h_4h_confirm']:
                probability_bonus += 10
            if checks['ATR_min']:
                probability_bonus += 5
            if checks['Volume_min']:
                probability_bonus += 5
            if checks['15m_30m_impulse']:
                probability_bonus += 5
            
            base_probability = 50  # Базовая вероятность
            if side == 'buy' and '45m' in mtf_data:
                rsi = mtf_data['45m']['rsi']
                if 30 < rsi < 70:  # Здоровая зона
                    base_probability += 10
                elif rsi < 30:  # Перепродано - хороший вход для лонга
                    base_probability += 15
            
            final_probability = min(95, base_probability + probability_bonus)
            
            logger.info(f"📊 Вероятность достижения TP: ~{final_probability}%")
            logger.info(f"   Базовая: {base_probability}%")
            logger.info(f"   Бонусы за проверки: +{probability_bonus}%")
        
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
        import asyncio
        if asyncio.iscoroutinefunction(exchange.fetch_positions):
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
            await analyze_position_entry(exchange, symbol)
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ АНАЛИЗ ЗАВЕРШЕН")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    # Импортируем pandas и talib только здесь
    import pandas as pd
    import talib
    import numpy as np
    
    asyncio.run(main())

