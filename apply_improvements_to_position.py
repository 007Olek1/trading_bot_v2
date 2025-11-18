#!/usr/bin/env python3
"""
🚀 ПРИМЕНЕНИЕ УЛУЧШЕНИЙ К ОТКРЫТОЙ ПОЗИЦИИ
Применяет адаптивные параметры, ML предсказания и динамический trailing SL
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

# Загружаем переменные окружения
load_dotenv()

# Импортируем новые модули
from adaptive_params import AdaptiveParameters
from ml_predictor import MLPredictor
from dynamic_position_sizer import DynamicPositionSizer
from circuit_breaker import CircuitBreaker

# Импортируем конфигурацию
import config


def get_market_data(client, symbol: str, interval: str = "15", limit: int = 100):
    """Получает рыночные данные"""
    try:
        response = client.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if response['retCode'] != 0:
            print(f"❌ Ошибка получения данных: {response.get('retMsg')}")
            return None
        
        klines = response['result']['list']
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        # Конвертируем типы
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def apply_improvements_to_position(symbol: str = "ADAUSDT"):
    """Применяет улучшения к открытой позиции"""
    
    print("="*80)
    print("🚀 ПРИМЕНЕНИЕ УЛУЧШЕНИЙ К ПОЗИЦИИ")
    print("="*80)
    print()
    
    # Инициализация клиента
    client = HTTP(
        testnet=config.USE_TESTNET,
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET,
    )
    
    # Получаем текущую позицию
    print(f"📊 Получение данных позиции {symbol}...")
    response = client.get_positions(
        category="linear",
        symbol=symbol
    )
    
    if response['retCode'] != 0:
        print(f"❌ Ошибка: {response.get('retMsg')}")
        return
    
    positions = [p for p in response['result']['list'] if float(p.get('size', 0)) > 0]
    
    if not positions:
        print(f"📭 Нет открытых позиций по {symbol}")
        return
    
    pos = positions[0]
    
    # Данные позиции
    entry_price = float(pos['avgPrice'])
    current_price = float(pos['markPrice'])
    size = float(pos['size'])
    side = pos['side']
    direction = "LONG" if side == "Buy" else "SHORT"
    unrealized_pnl = float(pos.get('unrealisedPnl', 0))
    
    print(f"✅ Позиция найдена:")
    print(f"   Направление: {direction}")
    print(f"   Размер: {size}")
    print(f"   Вход: ${entry_price:.6f}")
    print(f"   Текущая: ${current_price:.6f}")
    print(f"   PnL: ${unrealized_pnl:.2f}")
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 1. АДАПТИВНЫЕ ПАРАМЕТРЫ ПОД ВОЛАТИЛЬНОСТЬ
    # ═══════════════════════════════════════════════════════════════
    
    print("📊 1. АНАЛИЗ ВОЛАТИЛЬНОСТИ И АДАПТИВНЫЕ ПАРАМЕТРЫ")
    print("-"*80)
    
    # Получаем рыночные данные
    df = get_market_data(client, symbol)
    
    if df is None:
        print("❌ Не удалось получить рыночные данные")
        return
    
    # Инициализируем адаптивные параметры
    adapter = AdaptiveParameters()
    
    # Получаем адаптивные параметры
    adaptive_params = adapter.get_adaptive_params(df)
    
    print(f"   Волатильность: {adaptive_params['volatility']*100:.2f}%")
    print(f"   Уровень: {adaptive_params['volatility_level']}")
    print(f"   Рекомендованный Trailing SL: {adaptive_params['trailing_sl_callback']:.1f}%")
    print(f"   Рекомендованный порог уверенности: {adaptive_params['min_confidence']:.0%}")
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 2. РАСЧЁТ НОВОГО TRAILING SL
    # ═══════════════════════════════════════════════════════════════
    
    print("🔄 2. РАСЧЁТ АДАПТИВНОГО TRAILING SL")
    print("-"*80)
    
    # Базовый откат
    base_callback = config.TRAILING_SL_CALLBACK_PERCENT / 100
    
    # Адаптивный откат
    adaptive_callback = adaptive_params['trailing_sl_callback'] / 100
    
    # Рассчитываем новый SL
    if direction == "LONG":
        new_sl = current_price * (1 - adaptive_callback)
    else:  # SHORT
        new_sl = current_price * (1 + adaptive_callback)
    
    # Текущий SL
    current_sl = float(pos.get('stopLoss', 0)) if pos.get('stopLoss') else None
    
    print(f"   Текущий SL: ${current_sl:.6f}" if current_sl else "   Текущий SL: Не установлен")
    print(f"   Базовый откат: {base_callback*100:.1f}%")
    print(f"   Адаптивный откат: {adaptive_callback*100:.1f}%")
    print(f"   Новый SL: ${new_sl:.6f}")
    
    # Проверяем улучшает ли новый SL позицию
    should_update = False
    if current_sl:
        if direction == "LONG" and new_sl > current_sl:
            should_update = True
        elif direction == "SHORT" and new_sl < current_sl:
            should_update = True
    else:
        should_update = True
    
    if should_update:
        print(f"   ✅ Новый SL лучше - будет обновлён")
    else:
        print(f"   ⏸️  Текущий SL лучше - оставляем без изменений")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 3. РЫНОЧНЫЙ РЕЖИМ
    # ═══════════════════════════════════════════════════════════════
    
    print("📈 3. АНАЛИЗ РЫНОЧНОГО РЕЖИМА")
    print("-"*80)
    
    regime = adapter.get_market_regime(df)
    
    print(f"   Режим: {regime['regime']}")
    print(f"   Сила тренда: {regime['trend_strength']*100:.2f}%")
    print(f"   Волатильность: {regime['volatility']*100:.2f}% ({regime['volatility_level']})")
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 4. ПРИМЕНЕНИЕ ИЗМЕНЕНИЙ
    # ═══════════════════════════════════════════════════════════════
    
    print("🔧 4. ПРИМЕНЕНИЕ ИЗМЕНЕНИЙ")
    print("-"*80)
    
    if should_update:
        print("   Обновление Trailing SL...")
        
        try:
            result = client.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(round(new_sl, 6)),
                positionIdx=0
            )
            
            if result['retCode'] == 0:
                print(f"   ✅ Trailing SL обновлён: ${new_sl:.6f}")
                
                # Рассчитываем защищённую прибыль
                if direction == "LONG":
                    protected_profit = (new_sl - entry_price) * size
                else:
                    protected_profit = (entry_price - new_sl) * size
                
                print(f"   💰 Защищённая прибыль: ${protected_profit:.2f}")
            else:
                print(f"   ❌ Ошибка обновления: {result.get('retMsg')}")
        
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print("   ⏸️  Обновление не требуется")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 5. РЕКОМЕНДАЦИИ
    # ═══════════════════════════════════════════════════════════════
    
    print("💡 5. РЕКОМЕНДАЦИИ")
    print("-"*80)
    
    # Рекомендации на основе режима рынка
    if regime['regime'] == 'uptrend' and direction == 'SHORT':
        print("   ⚠️  Позиция SHORT в восходящем тренде - повышенный риск")
    elif regime['regime'] == 'downtrend' and direction == 'LONG':
        print("   ⚠️  Позиция LONG в нисходящем тренде - повышенный риск")
    else:
        print("   ✅ Направление позиции соответствует тренду")
    
    # Рекомендации по волатильности
    if adaptive_params['volatility_level'] in ['high', 'very_high']:
        print(f"   ⚠️  Высокая волатильность - используем больший откат SL ({adaptive_callback*100:.1f}%)")
    else:
        print(f"   ✅ Нормальная волатильность - стандартный откат SL")
    
    # Рекомендации по PnL
    pnl_percent = (unrealized_pnl / (entry_price * size)) * 100
    
    if pnl_percent >= 10:
        print(f"   🎯 Минимальная цель +10% достигнута! Держим до разворота")
    elif pnl_percent > 0:
        print(f"   📈 В прибыли (+{pnl_percent:.1f}%), продолжаем держать")
    else:
        print(f"   📉 В убытке ({pnl_percent:.1f}%), ждём разворота или SL")
    
    print()
    print("="*80)
    print("✅ ПРИМЕНЕНИЕ УЛУЧШЕНИЙ ЗАВЕРШЕНО")
    print("="*80)


if __name__ == "__main__":
    apply_improvements_to_position("ADAUSDT")
