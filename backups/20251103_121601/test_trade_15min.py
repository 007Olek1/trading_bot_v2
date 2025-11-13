#!/usr/bin/env python3
"""
Тестовая сделка на 15 минут - анализ рынка и открытие позиции
"""
import os
import sys
import asyncio
import ccxt
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WARSAW_TZ = pytz.timezone('Europe/Warsaw')

# Загружаем переменные окружения
env_file = Path("/opt/bot/.env")
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv()

async def analyze_market_and_find_signal(exchange):
    """Анализ рынка и поиск лучшего сигнала"""
    try:
        # Импортируем класс бота для анализа
        sys.path.insert(0, '/opt/bot')
        from super_bot_v4_mtf import SuperBotV4MTF
        
        bot = SuperBotV4MTF()
        await bot.initialize()
        
        logger.info("="*70)
        logger.info("📊 АНАЛИЗ РЫНКА ДЛЯ ТЕСТОВОЙ СДЕЛКИ")
        logger.info("="*70)
        
        # Анализ рыночных условий
        market_data = await bot.analyze_market_trend_v4()
        market_condition = market_data.get('trend', 'neutral').upper()
        btc_change = market_data.get('btc_change', 0)
        
        logger.info(f"\n📈 Рыночные условия: {market_condition}")
        logger.info(f"📊 Изменение BTC за 24ч: {btc_change:.2f}%")
        
        # Умный выбор символов
        symbols = await bot.smart_symbol_selection_v4(market_data)
        logger.info(f"\n🎯 Выбрано символов для анализа: {len(symbols)}")
        logger.info(f"   Топ-10: {', '.join(symbols[:10])}")
        
        # Анализируем первые 20 символов для быстрого поиска сигнала
        candidates = []
        analyzed = 0
        max_analyze = min(20, len(symbols))
        
        logger.info(f"\n🔍 Анализируем топ-{max_analyze} символов...")
        
        for symbol in symbols[:max_analyze]:
            try:
                analyzed += 1
                signal = await bot.analyze_symbol_v4(symbol)
                
                if signal:
                    # Проверяем все фильтры
                    mtf_data = await bot._fetch_multi_timeframe_data(symbol)
                    current_45m = mtf_data.get('45m', {})
                    current_1h = mtf_data.get('1h', {})
                    current_4h = mtf_data.get('4h', {})
                    current_15m = mtf_data.get('15m', {})
                    current_30m = mtf_data.get('30m', {})
                    
                    # Проверяем MTF подтверждение
                    mtf_ok = False
                    if signal.direction == 'buy':
                        mtf_ok = (current_45m.get('ema_9', 0) > current_45m.get('ema_21', 0) and
                                 current_1h.get('ema_9', 0) > current_1h.get('ema_21', 0) and
                                 current_4h.get('ema_9', 0) > current_4h.get('ema_21', 0))
                    else:
                        mtf_ok = (current_45m.get('ema_9', 0) < current_45m.get('ema_21', 0) and
                                 current_1h.get('ema_9', 0) < current_1h.get('ema_21', 0) and
                                 current_4h.get('ema_9', 0) < current_4h.get('ema_21', 0))
                    
                    # Проверяем импульс
                    impulse_ok = False
                    if signal.direction == 'buy':
                        impulse_ok = (current_15m.get('ema_9', 0) > current_15m.get('ema_21', 0) and
                                     current_30m.get('ema_9', 0) > current_30m.get('ema_21', 0))
                    else:
                        impulse_ok = (current_15m.get('ema_9', 0) < current_15m.get('ema_21', 0) and
                                     current_30m.get('ema_9', 0) < current_30m.get('ema_21', 0))
                    
                    # Проверяем волатильность
                    atr_pct = (current_45m.get('atr', 0) / current_45m.get('price', 1)) * 100
                    vol_ratio = current_45m.get('volume_ratio', 0)
                    
                    potential_ok = (atr_pct >= 1.2 and vol_ratio >= 1.2)
                    
                    if mtf_ok and impulse_ok and potential_ok:
                        candidates.append({
                            'symbol': symbol,
                            'signal': signal,
                            'confidence': signal.confidence,
                            'mtf_ok': mtf_ok,
                            'impulse_ok': impulse_ok,
                            'potential_ok': potential_ok
                        })
                        logger.info(f"✅ {symbol}: {signal.direction.upper()} | Уверенность: {signal.confidence:.0f}% | Все проверки OK")
                    
            except Exception as e:
                logger.debug(f"⚠️ Ошибка анализа {symbol}: {e}")
                continue
        
        logger.info(f"\n📊 Проанализировано: {analyzed}/{max_analyze} символов")
        
        if not candidates:
            logger.warning("\n⚠️ Подходящих сигналов не найдено. Все проверки пройдены, но сигналов нет.")
            return None
        
        # Выбираем лучший сигнал по уверенности
        best = max(candidates, key=lambda x: x['confidence'])
        
        logger.info("\n" + "="*70)
        logger.info("🎯 НАЙДЕН ЛУЧШИЙ СИГНАЛ ДЛЯ ТЕСТОВОЙ СДЕЛКИ")
        logger.info("="*70)
        logger.info(f"Символ: {best['symbol']}")
        logger.info(f"Направление: {best['signal'].direction.upper()}")
        logger.info(f"Уверенность: {best['confidence']:.0f}%")
        logger.info(f"Цена входа: ${best['signal'].entry_price:.6f}")
        logger.info(f"MTF подтверждение: {'✅' if best['mtf_ok'] else '❌'}")
        logger.info(f"Импульс 15m/30m: {'✅' if best['impulse_ok'] else '❌'}")
        logger.info(f"Волатильность: {'✅' if best['potential_ok'] else '❌'}")
        logger.info(f"Причины: {', '.join(best['signal'].reasons)}")
        
        return best
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа рынка: {e}", exc_info=True)
        return None

async def open_test_position(exchange, signal_data):
    """Открытие тестовой позиции"""
    try:
        signal = signal_data['signal']
        symbol = signal_data['symbol']
        
        logger.info("\n" + "="*70)
        logger.info("🚀 ОТКРЫТИЕ ТЕСТОВОЙ ПОЗИЦИИ")
        logger.info("="*70)
        
        # Параметры позиции (тестовые - меньше обычного)
        POSITION_SIZE = 2.0  # $2 маржи (меньше для теста)
        LEVERAGE = 5
        position_notional = POSITION_SIZE * LEVERAGE  # $10
        
        # Получаем информацию о символе
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Рассчитываем количество
        qty = position_notional / current_price
        qty = round(qty, 8)  # Округляем
        
        # Направление
        side = 'buy' if signal.direction == 'buy' else 'sell'
        
        logger.info(f"Символ: {symbol}")
        logger.info(f"Направление: {side.upper()}")
        logger.info(f"Цена: ${current_price:.6f}")
        logger.info(f"Количество: {qty}")
        logger.info(f"Размер позиции: ${position_notional:.2f} (маржа: ${POSITION_SIZE})")
        
        # Открываем позицию
        order = await exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=qty,
            params={'category': 'linear'}
        )
        
        logger.info(f"✅ Позиция открыта!")
        logger.info(f"   Order ID: {order.get('id')}")
        logger.info(f"   Entry Price: ${order.get('price', current_price):.6f}")
        
        # Устанавливаем TP +1.15% и SL -$1
        entry_price = float(order.get('price', current_price))
        
        if side == 'buy':
            tp_price = entry_price * 1.0115  # +1.15%
            sl_price = entry_price - (1.0 / (qty * LEVERAGE))  # -$1 на позицию $10
        else:
            tp_price = entry_price * 0.9885  # -1.15% для SHORT
            sl_price = entry_price + (1.0 / (qty * LEVERAGE))
        
        # Устанавливаем TP/SL
        try:
            from pybit.unified_trading import HTTP
            session = HTTP(
                api_key=os.getenv('BYBIT_API_KEY'),
                api_secret=os.getenv('BYBIT_API_SECRET'),
                testnet=False
            )
            
            bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
            
            # TP
            session.set_trading_stop(
                category='linear',
                symbol=bybit_symbol,
                takeProfit=str(tp_price),
                tpTriggerBy='LastPrice'
            )
            
            # SL
            session.set_trading_stop(
                category='linear',
                symbol=bybit_symbol,
                stopLoss=str(sl_price),
                slTriggerBy='LastPrice'
            )
            
            logger.info(f"✅ TP/SL установлены:")
            logger.info(f"   TP: ${tp_price:.6f} (+1.15%)")
            logger.info(f"   SL: ${sl_price:.6f} (-$1)")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось установить TP/SL автоматически: {e}")
        
        return {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'qty': qty,
            'order_id': order.get('id'),
            'opened_at': datetime.now(WARSAW_TZ)
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка открытия позиции: {e}", exc_info=True)
        return None

async def wait_and_close_position(exchange, position_info, wait_minutes=15):
    """Ожидание и закрытие позиции через указанное время"""
    try:
        logger.info("\n" + "="*70)
        logger.info(f"⏱️ ОЖИДАНИЕ {wait_minutes} МИНУТ ДО АВТОМАТИЧЕСКОГО ЗАКРЫТИЯ")
        logger.info("="*70)
        
        symbol = position_info['symbol']
        side = position_info['side']
        opened_at = position_info['opened_at']
        
        # Ждем указанное время
        wait_seconds = wait_minutes * 60
        elapsed = 0
        
        while elapsed < wait_seconds:
            await asyncio.sleep(30)  # Проверяем каждые 30 секунд
            elapsed += 30
            
            remaining = wait_seconds - elapsed
            minutes_left = remaining // 60
            seconds_left = remaining % 60
            
            if elapsed % 60 == 0:  # Каждую минуту
                logger.info(f"⏱️ Осталось: {minutes_left}м {seconds_left}с")
                
                # Проверяем текущий PnL
                try:
                    positions = await exchange.fetch_positions([symbol], params={'category': 'linear'})
                    for pos in positions:
                        if (pos.get('contracts', 0) or pos.get('size', 0)) > 0:
                            pnl = pos.get('unrealisedPnl', 0)
                            pnl_pct = pos.get('percentage', 0)
                            logger.info(f"   Текущий PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                            break
                except:
                    pass
        
        # Закрываем позицию
        logger.info("\n" + "="*70)
        logger.info("🔚 ЗАКРЫТИЕ ТЕСТОВОЙ ПОЗИЦИИ (15 минут истекли)")
        logger.info("="*70)
        
        # Получаем текущий размер позиции
        positions = await exchange.fetch_positions([symbol], params={'category': 'linear'})
        for pos in positions:
            size = pos.get('contracts', 0) or pos.get('size', 0)
            if size > 0:
                close_side = 'sell' if side == 'buy' else 'buy'
                
                order = await exchange.create_market_order(
                    symbol=symbol,
                    side=close_side,
                    amount=size,
                    params={'category': 'linear', 'reduceOnly': True}
                )
                
                exit_price = float(order.get('price', 0))
                entry_price = position_info['entry_price']
                
                if side == 'buy':
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
                position_notional = position_info['qty'] * entry_price
                pnl_usd = pnl_pct / 100 * position_notional
                
                logger.info(f"✅ Позиция закрыта!")
                logger.info(f"   Entry: ${entry_price:.6f}")
                logger.info(f"   Exit: ${exit_price:.6f}")
                logger.info(f"   PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
                logger.info(f"   Время удержания: {wait_minutes} минут")
                
                return {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd
                }
        
        logger.warning("⚠️ Позиция уже закрыта (TP/SL сработали)")
        return None
        
    except Exception as e:
        logger.error(f"❌ Ошибка закрытия позиции: {e}", exc_info=True)
        return None

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
        
        # Анализ рынка и поиск сигнала
        signal_data = await analyze_market_and_find_signal(exchange)
        
        if not signal_data:
            logger.warning("\n❌ Подходящих сигналов не найдено. Тестовая сделка отменена.")
            return
        
        # Подтверждение
        logger.info("\n" + "="*70)
        logger.info("❓ ОТКРЫТЬ ТЕСТОВУЮ ПОЗИЦИЮ?")
        logger.info("="*70)
        logger.info("Позиция будет автоматически закрыта через 15 минут")
        logger.info("Размер: $2 маржи ($10 позиция)")
        
        # Открываем позицию
        position_info = await open_test_position(exchange, signal_data)
        
        if not position_info:
            logger.error("❌ Не удалось открыть позицию")
            return
        
        # Ждем и закрываем
        result = await wait_and_close_position(exchange, position_info, wait_minutes=15)
        
        if result:
            logger.info("\n" + "="*70)
            logger.info("✅ ТЕСТОВАЯ СДЕЛКА ЗАВЕРШЕНА")
            logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())




