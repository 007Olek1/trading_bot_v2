#!/usr/bin/env python3
"""
🔍 ПОКАЗАТЬ РАБОТУ БОТА В РЕАЛЬНОМ ВРЕМЕНИ
Демонстрирует как бот ищет монеты и анализирует сигналы
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

load_dotenv()

import config
from indicators import MarketIndicators
from strategies import TrendVolumeStrategy, ManipulationDetector, GlobalTrendAnalyzer
from market_scanner import MarketScanner
from utils import setup_logging

# Настройка логирования
logger = setup_logging(Path("logs/analysis_demo.log"), "INFO")


def get_mtf_data(client, symbol: str):
    """Получает данные по всем таймфреймам"""
    mtf_data = {}
    
    for tf_name, tf_value in config.TIMEFRAMES.items():
        try:
            response = client.get_kline(
                category="linear",
                symbol=symbol,
                interval=tf_value,
                limit=200
            )
            
            if response['retCode'] == 0:
                klines = response['result']['list']
                
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                mtf_data[tf_name] = df
        except Exception as e:
            print(f"   ⚠️ Ошибка получения {tf_name}: {e}")
    
    return mtf_data


def analyze_symbol_detailed(symbol: str):
    """Детальный анализ монеты"""
    
    print("="*80)
    print(f"🔍 ДЕТАЛЬНЫЙ АНАЛИЗ: {symbol}")
    print("="*80)
    print()
    
    # Инициализация
    client = HTTP(
        testnet=config.USE_TESTNET,
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET,
    )
    
    indicators = MarketIndicators(config.INDICATOR_PARAMS)
    trend_volume = TrendVolumeStrategy(config.INDICATOR_PARAMS)
    manipulation = ManipulationDetector()
    global_trend = GlobalTrendAnalyzer()
    
    # ═══════════════════════════════════════════════════════════════
    # 1. ПОЛУЧЕНИЕ ДАННЫХ ПО ВСЕМ ТАЙМФРЕЙМАМ
    # ═══════════════════════════════════════════════════════════════
    
    print("📊 1. ПОЛУЧЕНИЕ MTF ДАННЫХ")
    print("-"*80)
    
    mtf_data = get_mtf_data(client, symbol)
    
    if not mtf_data:
        print("❌ Не удалось получить данные")
        return
    
    for tf_name in config.TIMEFRAMES.keys():
        if tf_name in mtf_data:
            df = mtf_data[tf_name]
            current_price = df['close'].iloc[-1]
            print(f"   ✅ {tf_name:4s}: {len(df)} свечей | Цена: ${current_price:.6f}")
        else:
            print(f"   ❌ {tf_name:4s}: Нет данных")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 2. РАСЧЁТ ИНДИКАТОРОВ (PRIMARY TIMEFRAME)
    # ═══════════════════════════════════════════════════════════════
    
    print("📈 2. РАСЧЁТ ИНДИКАТОРОВ (15m)")
    print("-"*80)
    
    primary_df = mtf_data[config.PRIMARY_TIMEFRAME]
    
    # EMA
    ema_20 = primary_df['close'].ewm(span=20).mean().iloc[-1]
    ema_50 = primary_df['close'].ewm(span=50).mean().iloc[-1]
    ema_200 = primary_df['close'].ewm(span=200).mean().iloc[-1]
    current_price = primary_df['close'].iloc[-1]
    
    print(f"   EMA 20:  ${ema_20:.6f}")
    print(f"   EMA 50:  ${ema_50:.6f}")
    print(f"   EMA 200: ${ema_200:.6f}")
    print(f"   Цена:    ${current_price:.6f}")
    
    # Тренд по EMA
    if current_price > ema_20 > ema_50:
        print(f"   📈 Тренд: ВОСХОДЯЩИЙ")
    elif current_price < ema_20 < ema_50:
        print(f"   📉 Тренд: НИСХОДЯЩИЙ")
    else:
        print(f"   ↔️  Тренд: БОКОВОЙ")
    
    print()
    
    # RSI
    delta = primary_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    print(f"   RSI: {current_rsi:.1f}")
    if current_rsi > 70:
        print(f"   ⚠️  Перекупленность (>70)")
    elif current_rsi < 30:
        print(f"   ⚠️  Перепроданность (<30)")
    else:
        print(f"   ✅ Нейтральная зона")
    
    print()
    
    # Объём
    avg_volume = primary_df['volume'].rolling(window=20).mean().iloc[-1]
    current_volume = primary_df['volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume
    
    print(f"   Объём текущий: {current_volume:.0f}")
    print(f"   Объём средний: {avg_volume:.0f}")
    print(f"   Отношение: {volume_ratio:.2f}x")
    
    if volume_ratio > 1.5:
        print(f"   🔥 Всплеск объёма!")
    else:
        print(f"   ✅ Нормальный объём")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 3. АНАЛИЗ СТРАТЕГИЙ
    # ═══════════════════════════════════════════════════════════════
    
    print("🎯 3. АНАЛИЗ СТРАТЕГИЙ")
    print("-"*80)
    
    # Стратегия 1: Тренд + Объём + BB
    print("   📊 Стратегия 1: Тренд + Объём + Bollinger (40%)")
    signal1 = trend_volume.analyze(mtf_data)
    
    if signal1:
        print(f"      Направление: {signal1['direction']}")
        print(f"      Уверенность: {signal1['confidence']:.1%}")
        print(f"      Причина: {signal1.get('reason', 'N/A')}")
    else:
        print(f"      ❌ Нет сигнала")
    
    print()
    
    # Стратегия 2: Детектор манипуляций
    print("   🎭 Стратегия 2: Детектор манипуляций (30%)")
    signal2 = manipulation.analyze(mtf_data)
    
    if signal2:
        print(f"      Направление: {signal2['direction']}")
        print(f"      Уверенность: {signal2['confidence']:.1%}")
        print(f"      Причина: {signal2.get('reason', 'N/A')}")
    else:
        print(f"      ❌ Нет сигнала")
    
    print()
    
    # Стратегия 3: Глобальный тренд
    print("   🌍 Стратегия 3: Глобальный тренд 4h+1D (30%)")
    signal3 = global_trend.analyze(mtf_data)
    
    if signal3:
        print(f"      Направление: {signal3['direction']}")
        print(f"      Уверенность: {signal3['confidence']:.1%}")
        print(f"      Причина: {signal3.get('reason', 'N/A')}")
    else:
        print(f"      ❌ Нет сигнала")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 4. ВЗВЕШЕННАЯ ОЦЕНКА
    # ═══════════════════════════════════════════════════════════════
    
    print("⚖️  4. ВЗВЕШЕННАЯ ОЦЕНКА")
    print("-"*80)
    
    signals = []
    if signal1:
        signals.append(('Тренд+Объём+BB', signal1, 0.40))
    if signal2:
        signals.append(('Манипуляции', signal2, 0.30))
    if signal3:
        signals.append(('Глобальный тренд', signal3, 0.30))
    
    if not signals:
        print("   ❌ Нет сигналов от стратегий")
        print()
        return
    
    # Проверяем согласованность направлений
    directions = [s[1]['direction'] for s in signals]
    if len(set(directions)) > 1:
        print(f"   ⚠️  Стратегии не согласованы: {directions}")
        print(f"   ❌ Сигнал отклонён")
        print()
        return
    
    # Рассчитываем взвешенную уверенность
    final_direction = directions[0]
    final_confidence = sum(s[1]['confidence'] * s[2] for s in signals)
    
    print(f"   Направление: {final_direction}")
    print(f"   Взвешенная уверенность: {final_confidence:.1%}")
    print()
    
    for name, signal, weight in signals:
        print(f"      {name:20s}: {signal['confidence']:.1%} × {weight:.0%} = {signal['confidence']*weight:.1%}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 5. ПРОВЕРКА MTF ВЫРАВНИВАНИЯ
    # ═══════════════════════════════════════════════════════════════
    
    print("🔄 5. MTF ВЫРАВНИВАНИЕ")
    print("-"*80)
    
    aligned_count = 0
    
    for tf_name in config.TIMEFRAMES.keys():
        if tf_name not in mtf_data:
            continue
        
        df = mtf_data[tf_name]
        
        # Простая проверка тренда
        ema_20_tf = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50_tf = df['close'].ewm(span=50).mean().iloc[-1]
        current_price_tf = df['close'].iloc[-1]
        
        if final_direction == "LONG":
            if current_price_tf > ema_20_tf > ema_50_tf:
                aligned_count += 1
                print(f"   ✅ {tf_name:4s}: Подтверждает LONG")
            else:
                print(f"   ❌ {tf_name:4s}: Не подтверждает")
        else:  # SHORT
            if current_price_tf < ema_20_tf < ema_50_tf:
                aligned_count += 1
                print(f"   ✅ {tf_name:4s}: Подтверждает SHORT")
            else:
                print(f"   ❌ {tf_name:4s}: Не подтверждает")
    
    print()
    print(f"   Подтверждено таймфреймов: {aligned_count}/{len(config.TIMEFRAMES)}")
    print(f"   Минимум требуется: {config.MIN_TIMEFRAME_ALIGNMENT}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 6. ФИНАЛЬНОЕ РЕШЕНИЕ
    # ═══════════════════════════════════════════════════════════════
    
    print("✅ 6. ФИНАЛЬНОЕ РЕШЕНИЕ")
    print("-"*80)
    
    # Проверка условий
    conditions = []
    
    # 1. Уверенность
    if final_confidence >= config.SIGNAL_THRESHOLDS['min_confidence']:
        conditions.append(f"✅ Уверенность {final_confidence:.1%} >= {config.SIGNAL_THRESHOLDS['min_confidence']:.0%}")
    else:
        conditions.append(f"❌ Уверенность {final_confidence:.1%} < {config.SIGNAL_THRESHOLDS['min_confidence']:.0%}")
    
    # 2. MTF выравнивание
    if aligned_count >= config.MIN_TIMEFRAME_ALIGNMENT:
        conditions.append(f"✅ MTF выравнивание {aligned_count} >= {config.MIN_TIMEFRAME_ALIGNMENT}")
    else:
        conditions.append(f"❌ MTF выравнивание {aligned_count} < {config.MIN_TIMEFRAME_ALIGNMENT}")
    
    # 3. Объём
    if volume_ratio >= config.SIGNAL_THRESHOLDS['min_volume_ratio']:
        conditions.append(f"✅ Объём {volume_ratio:.2f}x >= {config.SIGNAL_THRESHOLDS['min_volume_ratio']:.1f}x")
    else:
        conditions.append(f"❌ Объём {volume_ratio:.2f}x < {config.SIGNAL_THRESHOLDS['min_volume_ratio']:.1f}x")
    
    print("   Проверка условий:")
    for cond in conditions:
        print(f"      {cond}")
    
    print()
    
    # Финальное решение
    all_passed = all("✅" in c for c in conditions)
    
    if all_passed:
        print(f"   🎯 СИГНАЛ ПРИНЯТ!")
        print(f"   📊 Направление: {final_direction}")
        print(f"   💪 Уверенность: {final_confidence:.1%}")
        print(f"   💰 Вход по цене: ${current_price:.6f}")
    else:
        print(f"   ❌ СИГНАЛ ОТКЛОНЁН")
        print(f"   Причина: Не все условия выполнены")
    
    print()
    print("="*80)


def show_market_scanning():
    """Показывает процесс сканирования рынка"""
    
    print("="*80)
    print("🔍 ПРОЦЕСС СКАНИРОВАНИЯ РЫНКА")
    print("="*80)
    print()
    
    # Инициализация
    client = HTTP(
        testnet=config.USE_TESTNET,
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET,
    )
    
    scanner = MarketScanner(client, logger)
    
    # ═══════════════════════════════════════════════════════════════
    # 1. ДИНАМИЧЕСКОЕ СКАНИРОВАНИЕ
    # ═══════════════════════════════════════════════════════════════
    
    print("📊 1. ДИНАМИЧЕСКОЕ СКАНИРОВАНИЕ ЛИКВИДНЫХ МОНЕТ")
    print("-"*80)
    
    watchlist = scanner.get_liquid_symbols(
        min_volume_24h=config.MIN_VOLUME_24H_USD,
        limit=config.MAX_SYMBOLS_TO_SCAN
    )
    
    print(f"   Найдено монет: {len(watchlist)}")
    print(f"   Минимальный объём: ${config.MIN_VOLUME_24H_USD:,}")
    print()
    
    print("   Топ-10 монет по ликвидности:")
    for i, symbol in enumerate(watchlist[:10], 1):
        print(f"      {i:2d}. {symbol}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 2. АНАЛИЗ ПЕРВЫХ 3 МОНЕТ
    # ═══════════════════════════════════════════════════════════════
    
    print("🔍 2. АНАЛИЗ ПЕРВЫХ 3 МОНЕТ")
    print("-"*80)
    print()
    
    for symbol in watchlist[:3]:
        print(f"📊 Анализ {symbol}...")
        analyze_symbol_detailed(symbol)
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Анализ конкретной монеты
        symbol = sys.argv[1].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        analyze_symbol_detailed(symbol)
    else:
        # Полное сканирование
        show_market_scanning()
