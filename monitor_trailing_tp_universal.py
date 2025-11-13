#!/usr/bin/env python3
"""
Универсальный монитор трейлинг TP/SL для ВСЕХ позиций
- Мониторит все открытые позиции
- Применяет трейлинг TP: старт +1% с шагом 0.5% до +5%
- Применяет SL: -$1 максимум, переводит в BE после +$1 прибыли
- Отправляет уведомления в Telegram при закрытии позиций
"""
import os
import time
import math
import requests
from pybit.unified_trading import HTTP

# Загружаем переменные окружения из объединенного файла
env_file = '/opt/bot/.env'
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

# Параметры трейлинг TP
# Для гарантированного +$1 на позицию $25 нужно минимум +4%
# Логика: старт +1% ($0.25), трейлинг 0.5% → +4% ($1.00) → +5% ($1.25)
BASE_TARGET = 1.0  # Стартовая цель +1% ($0.25 на $25)
GUARANTEED_TARGET = 4.0  # Гарантированный минимум +4% ($1.00 на $25)
STEP = 0.5  # Шаг трейлинга 0.5%
MAX_TARGET = 5.0  # Максимальная цель +5% ($1.25 на $25)

# Параметры SL
NOTIONAL = 25.0  # Размер позиции в USDT
RISK_USD = 1.0  # Максимальный риск -$1
BE_BUFFER = 0.001  # Буфер для BE (0.1%)
SLIPPAGE_BUFFER = 0.15  # Буфер на проскальзывание 0.15% (для предотвращения превышения лимита -$1)

# Адаптивные интервалы проверки
INTERVAL_MIN = 30  # Минимальный интервал (секунды) - для новых/волатильных позиций
INTERVAL_NORMAL = 60  # Нормальный интервал (секунды) - для обычных позиций
INTERVAL_LONG = 180  # Долгосрочный интервал (секунды) - для спокойных позиций > 1 часа
INTERVAL_MAX = 300  # Максимальный интервал (секунды) - для очень спокойных > 4 часов

# Пороги для адаптации
THRESHOLD_15MIN = 900  # 15 минут (секунды)
THRESHOLD_45MIN = 2700  # 45 минут
THRESHOLD_1H = 3600  # 1 час
THRESHOLD_4H = 14400  # 4 часа

# API ключи
api_key = os.getenv('BYBIT_API_KEY') or os.getenv('API_KEY')
api_secret = os.getenv('BYBIT_API_SECRET') or os.getenv('API_SECRET')

if not api_key or not api_secret:
    print("❌ ОШИБКА: BYBIT_API_KEY или BYBIT_API_SECRET не установлены!")
    exit(1)

S = HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=5000, timeout=15)

# Telegram
TG_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN') or os.getenv('TG_BOT_TOKEN')
TG_CHAT = os.getenv('TELEGRAM_CHAT_ID') or os.getenv('TG_CHAT_ID')

def tgsend(text: str):
    """Отправка сообщения в Telegram"""
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"⚠️ Ошибка отправки Telegram: {e}")

def profit_pct(entry: float, mark: float, side: str) -> float:
    """Рассчитывает процент прибыли"""
    if entry <= 0 or mark <= 0:
        return 0.0
    if side == 'Sell':
        return (entry - mark) / entry * 100.0
    else:
        return (mark - entry) / entry * 100.0

def get_adaptive_interval(created_time_ms: int, current_prof_pct: float, distance_to_tp: float) -> int:
    """
    Адаптивный интервал проверки в зависимости от:
    - Времени удержания позиции
    - Текущей волатильности (расстояние до TP)
    - Процента прибыли
    """
    if created_time_ms <= 0:
        return INTERVAL_MIN  # Если время неизвестно, используем минимальный интервал
    
    now = time.time()
    position_age = now - (created_time_ms / 1000.0)  # Возраст позиции в секундах
    
    # Новые позиции (первые 15 минут) - часто
    if position_age < THRESHOLD_15MIN:
        return INTERVAL_MIN  # 30 сек
    
    # Позиции 15-45 минут - средняя частота
    elif position_age < THRESHOLD_45MIN:
        # Если близко к TP или высокая прибыль - проверяем чаще
        if current_prof_pct > 2.0 or distance_to_tp < 1.0:
            return INTERVAL_MIN  # 30 сек
        return INTERVAL_NORMAL  # 60 сек
    
    # Позиции 45 минут - 1 час - реже
    elif position_age < THRESHOLD_1H:
        # Если близко к критическим уровням - проверяем чаще
        if current_prof_pct > 3.0 or distance_to_tp < 0.5:
            return INTERVAL_NORMAL  # 60 сек
        return INTERVAL_NORMAL  # 60 сек
    
    # Позиции 1-4 часа - еще реже
    elif position_age < THRESHOLD_4H:
        # Только если близко к TP или BE
        if current_prof_pct > 3.5 or distance_to_tp < 0.3:
            return INTERVAL_NORMAL  # 60 сек
        return INTERVAL_LONG  # 3 минуты
    
    # Позиции > 4 часов - минимальная частота
    else:
        # Только если очень близко к критическим уровням
        if current_prof_pct > 4.0 or distance_to_tp < 0.2:
            return INTERVAL_LONG  # 3 минуты
        return INTERVAL_MAX  # 5 минут

def main():
    """Основной цикл мониторинга"""
    last_targets = {}  # Храним последний установленный TP для каждого символа
    prev_sizes = {}  # Храним предыдущие размеры позиций для определения закрытия
    position_times = {}  # Время создания позиций: {symbol: timestamp_ms}
    
    print("🚀 Универсальный монитор трейлинг TP/SL запущен")
    print(f"📊 Адаптивный мониторинг: {INTERVAL_MIN}-{INTERVAL_MAX} секунд")
    print(f"🎯 TP: старт +{BASE_TARGET}% с шагом {STEP}% до +{MAX_TARGET}%")
    print(f"🛑 SL: максимум -${RISK_USD} на сделку → Trailing")
    
    while True:
        current_interval = INTERVAL_MIN  # Начальный интервал
        try:
            # Получаем все открытые позиции
            r = S.get_positions(category='linear', settleCoin='USDT', limit=200)
            positions_list = r.get('result', {}).get('list', []) or []
            
            current_open = {}  # Текущие открытые позиции: {symbol: size}
            
            # Обрабатываем каждую позицию
            for pos in positions_list:
                try:
                    size = float(pos.get('size') or 0.0)
                    symbol = pos.get('symbol')
                    
                    if not symbol or size <= 0:
                        continue
                    
                    current_open[symbol] = size
                    
                    side = pos.get('side')
                    entry = float(pos.get('avgPrice') or 0)
                    mark = float(pos.get('markPrice') or 0)
                    upnl = float(pos.get('unrealisedPnl') or 0)
                    
                    if entry <= 0 or mark <= 0:
                        continue
                    
                    # Сохраняем время создания позиции (если еще не сохранили)
                    if symbol not in position_times:
                        created_time = pos.get('createdTime')
                        if created_time:
                            position_times[symbol] = int(created_time)
                        else:
                            position_times[symbol] = int(time.time() * 1000)  # Текущее время как fallback
                    
                    # Рассчитываем процент прибыли
                    prof = profit_pct(entry, mark, side)
                    
                    # Рассчитываем расстояние до текущего TP (если установлен)
                    current_tp_str = pos.get('takeProfit')
                    distance_to_tp = 999.0  # Большое значение по умолчанию
                    if current_tp_str:
                        try:
                            current_tp = float(current_tp_str)
                            if side == 'Sell':
                                distance_to_tp = abs((current_tp - mark) / mark * 100.0)
                            else:
                                distance_to_tp = abs((mark - current_tp) / mark * 100.0)
                        except:
                            pass
                    
                    # Рассчитываем SL с учетом проскальзывания
                    # Уменьшаем риск на буфер проскальзывания, чтобы фактический убыток не превысил -$1
                    risk_pct_base = RISK_USD / NOTIONAL  # -4%
                    risk_pct = risk_pct_base - (SLIPPAGE_BUFFER / 100.0)  # -3.85% (с буфером)
                    sl = entry * (1 + risk_pct) if side == 'Sell' else entry * (1 - risk_pct)
                    
                    # Если прибыль >= $1, переводим SL в BE
                    if upnl >= 1.0:
                        sl = entry * (1 + BE_BUFFER) if side == 'Sell' else entry * (1 - BE_BUFFER)
                    
                    # Рассчитываем TP с трейлингом
                    # Логика: старт +1% ($0.25), трейлинг 0.5% → +4% ($1.00) → +5% ($1.25)
                    # Гарантированный минимум +$1 достигается при +4%
                    if prof < BASE_TARGET:
                        target = BASE_TARGET  # Старт +1%
                    elif prof < GUARANTEED_TARGET:
                        # Трейлинг от +1% до +4% (гарантия +$1)
                        steps = math.floor((prof - BASE_TARGET) / STEP)
                        target = min(BASE_TARGET + steps * STEP, GUARANTEED_TARGET)
                    else:
                        # Трейлинг от +4% до +5%
                        steps = math.floor((prof - GUARANTEED_TARGET) / STEP)
                        target = min(GUARANTEED_TARGET + steps * STEP, MAX_TARGET)
                    
                    # Рассчитываем адаптивный интервал для этой позиции
                    pos_interval = get_adaptive_interval(
                        position_times.get(symbol, 0),
                        prof,
                        distance_to_tp
                    )
                    # Используем минимальный интервал из всех позиций
                    current_interval = min(current_interval, pos_interval)
                    
                    # Проверяем, нужно ли обновить TP
                    prev_target = last_targets.get(symbol)
                    should_update = (
                        prev_target is None or  # Первый раз
                        target > prev_target or  # TP увеличился
                        upnl >= 1.0  # Прибыль >= $1, обновляем BE
                    )
                    
                    if should_update:
                        # Рассчитываем TP цену
                        if side == 'Sell':
                            tp = entry * (1 - target / 100.0)
                        else:
                            tp = entry * (1 + target / 100.0)
                        
                        try:
                            S.set_trading_stop(
                                category='linear',
                                symbol=symbol,
                                takeProfit=f"{tp:.6f}",
                                stopLoss=f"{sl:.6f}",
                                tpslMode='Full',
                                positionIdx=0
                            )
                            
                            status = "BE" if upnl >= 1.0 else f"TP{target:.1f}%"
                            print(f"✅ {symbol} {side}: {status} | TP={tp:.6f} SL={sl:.6f} | PnL=${upnl:.2f} ({prof:.2f}%)")
                            last_targets[symbol] = target
                            
                        except Exception as e:
                            print(f"❌ Ошибка установки TP/SL для {symbol}: {e}")
                    
                except Exception as e:
                    print(f"⚠️ Ошибка обработки позиции: {e}")
            
            # Проверяем закрытые позиции и отправляем уведомления
            for symbol, prev_size in list(prev_sizes.items()):
                if prev_size > 0 and current_open.get(symbol, 0.0) <= 0.0:
                    # Позиция закрылась
                    try:
                        cp = S.get_closed_pnl(category='linear', symbol=symbol, limit=1)
                        closed_list = (cp.get('result', {}) or {}).get('list', [])
                        
                        if closed_list:
                            row = closed_list[0]
                            pnl = float(row.get('closedPnl') or 0)
                            entry_price = float(row.get('avgEntryPrice') or 0)
                            exit_price = float(row.get('avgExitPrice') or 0)
                            ex_side = row.get('side') or 'Buy'
                            
                            direction = 'SHORT' if ex_side == 'Buy' else 'LONG'
                            pnl_emoji = "✅" if pnl >= 0 else "❌"
                            
                            msg = (
                                f"{pnl_emoji} *ПОЗИЦИЯ ЗАКРЫТА*\n\n"
                                f"🔖 {symbol} {direction}\n"
                                f"💵 Entry: ${entry_price:.6f}\n"
                                f"💰 Exit: ${exit_price:.6f}\n"
                                f"📊 Closed PnL: ${pnl:.2f}\n\n"
                                f"📌 Источник: биржа (TP/SL)"
                            )
                            tgsend(msg)
                            print(f"📨 Уведомление о закрытии {symbol} отправлено | PnL=${pnl:.2f}")
                            
                    except Exception as e:
                        print(f"⚠️ Ошибка получения данных о закрытии {symbol}: {e}")
            
            # Обновляем предыдущие размеры
            prev_sizes = current_open.copy()
            
            # Очищаем данные для закрытых позиций
            for symbol in list(position_times.keys()):
                if symbol not in current_open:
                    del position_times[symbol]
            
        except Exception as e:
            print(f"❌ Ошибка основного цикла: {e}")
        
        # Используем адаптивный интервал
        sleep_time = current_interval if current_interval > 0 else INTERVAL_NORMAL
        if len(current_open) > 0:
            print(f"⏱️ Следующая проверка через {sleep_time} сек (адаптивный интервал)")
        time.sleep(sleep_time)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Монитор остановлен пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

