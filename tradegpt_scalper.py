#!/usr/bin/env python3
"""
TradeGPT Scalper для Bybit Futures
Минимальная прибыль: +$0.50
Автоматический Trailing TP
Фиксированный риск: -$0.15
Упрощенная логика без MTF
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import ccxt
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорт модулей
from bybit_api import BybitAPI
from disco57_simple import Disco57Simple
from telegram_notifier import TelegramNotifier

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

POSITION_SIZE = 1.0  # $1 USDT margin
LEVERAGE = 20  # x20
EFFECTIVE_EXPOSURE = POSITION_SIZE * LEVERAGE  # $20 USDT
MAX_POSITIONS = 3
MIN_PROFIT_USD = 0.50  # Минимальная прибыль
MAX_LOSS_USD = 0.15  # Максимальный убыток на сделку
DAILY_MAX_LOSS_USD = 5.0  # Лимит убытка в день
TRAILING_ACTIVATION_USD = 0.35  # Активация трейлинга при +$0.35
TRAILING_DISTANCE_PCT = 0.0015  # 0.15% отступ трейлинга
SCAN_INTERVAL_SEC = 300  # 5 минут между сканированиями
BYBIT_FEE_PCT = 0.00075  # Комиссия Bybit ~0.075%
SYMBOL_COOLDOWN_SEC = 60  # 1 минута блокировки символа при ошибке SL/TP
POSITION_CLOSE_CHECK_INTERVAL = 2  # Интервал проверки закрытия позиции (секунды)
POSITION_CLOSE_MAX_WAIT = 30  # Максимальное время ожидания закрытия (секунды)
TRAILING_UPDATE_MIN_INTERVAL = 10  # Минимальный интервал обновления трейлинга (секунды)

# Список монет для торговли (145 монет)
TRADING_SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 
    'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'SOL/USDT:USDT', 'DOT/USDT:USDT',
    'MATIC/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT',
    'ATOM/USDT:USDT', 'LTC/USDT:USDT', 'ETC/USDT:USDT', 'XLM/USDT:USDT',
    'ALGO/USDT:USDT', 'FIL/USDT:USDT', 'TRX/USDT:USDT', 'EOS/USDT:USDT',
    'AAVE/USDT:USDT', 'GRT/USDT:USDT', 'THETA/USDT:USDT', 'VET/USDT:USDT',
    'ICP/USDT:USDT', 'FTM/USDT:USDT', 'HBAR/USDT:USDT', 'NEAR/USDT:USDT',
    'SAND/USDT:USDT', 'MANA/USDT:USDT', 'AXS/USDT:USDT', 'GALA/USDT:USDT',
    'APE/USDT:USDT', 'CHZ/USDT:USDT', 'ENJ/USDT:USDT', 'BAT/USDT:USDT',
    'ZEC/USDT:USDT', 'DASH/USDT:USDT', 'COMP/USDT:USDT', 'MKR/USDT:USDT',
    'SNX/USDT:USDT', 'YFI/USDT:USDT', 'SUSHI/USDT:USDT', '1INCH/USDT:USDT',
    'CRV/USDT:USDT', 'BAL/USDT:USDT', 'REN/USDT:USDT', 'KSM/USDT:USDT',
    'QTUM/USDT:USDT', 'ZIL/USDT:USDT', 'ICX/USDT:USDT', 'ONT/USDT:USDT',
    'ZRX/USDT:USDT', 'OMG/USDT:USDT', 'ANT/USDT:USDT', 'LRC/USDT:USDT',
    'STORJ/USDT:USDT', 'CVC/USDT:USDT', 'KNC/USDT:USDT', 'REP/USDT:USDT',
    'BNT/USDT:USDT', 'RLC/USDT:USDT', 'NMR/USDT:USDT', 'OCEAN/USDT:USDT',
    'BAND/USDT:USDT', 'RSR/USDT:USDT', 'KAVA/USDT:USDT', 'IOTX/USDT:USDT',
    'COTI/USDT:USDT', 'ANKR/USDT:USDT', 'CHR/USDT:USDT', 'STMX/USDT:USDT',
    'HOT/USDT:USDT', 'DENT/USDT:USDT', 'WIN/USDT:USDT', 'FUN/USDT:USDT',
    'CELR/USDT:USDT', 'MTL/USDT:USDT', 'OGN/USDT:USDT', 'NKN/USDT:USDT',
    'SC/USDT:USDT', 'DGB/USDT:USDT', 'SXP/USDT:USDT', 'IRIS/USDT:USDT',
    'BLZ/USDT:USDT', 'ARPA/USDT:USDT', 'CTSI/USDT:USDT', 'TROY/USDT:USDT',
    'VITE/USDT:USDT', 'FTT/USDT:USDT', 'EUR/USDT:USDT', 'ONG/USDT:USDT',
    'DUSK/USDT:USDT', 'PERL/USDT:USDT', 'TOMO/USDT:USDT', 'CTXC/USDT:USDT',
    'LEND/USDT:USDT', 'DOCK/USDT:USDT', 'POLY/USDT:USDT', 'DATA/USDT:USDT',
    'MFT/USDT:USDT', 'BEAM/USDT:USDT', 'XTZ/USDT:USDT', 'RVN/USDT:USDT',
    'HC/USDT:USDT', 'ONE/USDT:USDT', 'FET/USDT:USDT', 'TFUEL/USDT:USDT',
    'ATOM/USDT:USDT', 'ERD/USDT:USDT', 'ARDR/USDT:USDT', 'NULS/USDT:USDT',
    'WAN/USDT:USDT', 'WRX/USDT:USDT', 'LTO/USDT:USDT', 'MBL/USDT:USDT',
    'CELO/USDT:USDT', 'HIVE/USDT:USDT', 'STPT/USDT:USDT', 'SOL/USDT:USDT',
    'CKB/USDT:USDT', 'PAXG/USDT:USDT', 'UNFI/USDT:USDT', 'ROSE/USDT:USDT',
    'AVA/USDT:USDT', 'XEM/USDT:USDT', 'SKL/USDT:USDT', 'SUSD/USDT:USDT',
    'SRM/USDT:USDT', 'EGLD/USDT:USDT', 'DIA/USDT:USDT', 'RUNE/USDT:USDT',
    'WNXM/USDT:USDT', 'TRB/USDT:USDT', 'BZRX/USDT:USDT', 'WBTC/USDT:USDT',
    'SXP/USDT:USDT', 'YFII/USDT:USDT', 'INJ/USDT:USDT', 'AUDIO/USDT:USDT',
    'CTK/USDT:USDT', 'AKRO/USDT:USDT', 'KP3R/USDT:USDT', 'AXS/USDT:USDT',
    'HARD/USDT:USDT'
]


# ============================================================================
# КЛАССЫ ДАННЫХ
# ============================================================================

@dataclass
class Position:
    """Активная позиция"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    sl_price: float
    tp_price: float
    entry_time: float
    trailing_active: bool = False
    max_profit: float = 0.0
    current_pnl: float = 0.0
    bybit_order_id: Optional[str] = None
    last_trailing_update: float = 0.0  # Время последнего обновления трейлинга


# ============================================================================
# ОСНОВНОЙ КЛАСС БОТА
# ============================================================================

class TradeGPTScalper:
    """TradeGPT Scalper - быстрый скальпинг с минимальным профитом +$0.50"""
    
    def __init__(self):
        self.api = BybitAPI()
        self.disco57 = Disco57Simple()
        self.telegram = TelegramNotifier()
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.symbol_cooldowns: Dict[str, float] = {}  # Временная блокировка символов при ошибках
        
        logger.info("TradeGPT Scalper инициализирован")
        logger.info(f"Позиция: ${POSITION_SIZE} x{LEVERAGE} = ${EFFECTIVE_EXPOSURE}")
        logger.info(f"Мин. профит: +${MIN_PROFIT_USD} | Макс. убыток: -${MAX_LOSS_USD}")
    
    async def start(self):
        """Запуск бота"""
        logger.info("=" * 60)
        logger.info("TradeGPT Scalper запущен")
        logger.info("=" * 60)
        
        # Загружаем активные позиции с биржи
        await self.load_active_positions()
        
        # Основной цикл
        while True:
            try:
                await self.main_loop()
                await asyncio.sleep(SCAN_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("Остановка бота...")
                break
            except Exception as e:
                logger.error(f"Ошибка в главном цикле: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def load_active_positions(self):
        """Загрузить активные позиции с биржи при старте"""
        try:
            positions = await self.api.fetch_positions()
            for pos in positions:
                if pos['contracts'] > 0:
                    symbol = pos['symbol']
                    side = 'long' if pos['side'] == 'Buy' else 'short'
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=float(pos['entryPrice']),
                        quantity=float(pos['contracts']),
                        sl_price=float(pos.get('stopLoss', 0)),
                        tp_price=float(pos.get('takeProfit', 0)),
                        entry_time=time.time(),
                        trailing_active=False,
                        max_profit=0.0,
                        current_pnl=float(pos.get('unrealisedPnl', 0))
                    )
                    logger.info(f"Загружена позиция: {symbol} {side.upper()} @ {pos['entryPrice']}")
        except Exception as e:
            logger.error(f"Ошибка загрузки позиций: {e}")
    
    async def main_loop(self):
        """Основной цикл сканирования и торговли"""
        # Сброс дневной статистики
        self.reset_daily_stats()
        
        # Проверка лимита убытка
        if self.daily_pnl <= -DAILY_MAX_LOSS_USD:
            logger.warning(f"Достигнут дневной лимит убытка: ${self.daily_pnl:.2f}")
            await asyncio.sleep(3600)  # Пауза на 1 час
            return
        
        # Обновление существующих позиций
        await self.update_positions()
        
        # Если есть свободные слоты - ищем новые входы
        if len(self.positions) < MAX_POSITIONS:
            await self.scan_for_entries()
    
    def reset_daily_stats(self):
        """Сброс дневной статистики в полночь"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            logger.info(f"Дневная статистика: PnL=${self.daily_pnl:.2f}, Сделок={self.daily_trades}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            logger.info("Дневная статистика сброшена")
    
    async def update_positions(self):
        """Обновление существующих позиций"""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            
            try:
                # Получаем текущую цену
                ticker = await self.api.fetch_ticker(symbol)
                current_price = float(ticker['last'])
                
                # Рассчитываем PnL
                if pos.side == 'long':
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - current_price) / pos.entry_price
                
                pnl_usd = pnl_pct * EFFECTIVE_EXPOSURE
                pos.current_pnl = pnl_usd
                
                # Проверка активации трейлинга
                if not pos.trailing_active and pnl_usd >= TRAILING_ACTIVATION_USD:
                    pos.trailing_active = True
                    pos.max_profit = pnl_usd
                    logger.info(f"{symbol}: Трейлинг активирован при +${pnl_usd:.2f}")
                    await self.telegram.send_message(
                        f"🔄 TRAILING АКТИВИРОВАН\n{symbol} {pos.side.upper()}\n"
                        f"Прибыль: +${pnl_usd:.2f}"
                    )
                    # Отменяем фиксированный TP через API
                    await self.api.set_take_profit(symbol, pos.side, 0.0)
                    logger.info(f"{symbol}: Фиксированный TP отменен через API")
                
                # Обновление трейлинга
                if pos.trailing_active:
                    current_time = time.time()
                    if current_time - pos.last_trailing_update >= TRAILING_UPDATE_MIN_INTERVAL:
                        if pnl_usd > pos.max_profit:
                            pos.max_profit = pnl_usd
                        
                        # Рассчитываем новый SL
                        if pos.side == 'long':
                            new_sl = current_price - (current_price * TRAILING_DISTANCE_PCT)
                            # Проверка, что новый SL не выше текущей цены и не ниже предыдущего SL
                            if new_sl > current_price:
                                logger.warning(f"{symbol}: Новый SL {new_sl:.6f} выше цены {current_price:.6f}, корректируем")
                                new_sl = current_price * 0.999
                            if new_sl < pos.sl_price:
                                logger.warning(f"{symbol}: Новый SL {new_sl:.6f} ниже предыдущего {pos.sl_price:.6f}, корректируем")
                                new_sl = pos.sl_price
                        else:  # short
                            new_sl = current_price + (current_price * TRAILING_DISTANCE_PCT)
                            # Проверка, что новый SL не ниже текущей цены и не выше предыдущего SL
                            if new_sl < current_price:
                                logger.warning(f"{symbol}: Новый SL {new_sl:.6f} ниже цены {current_price:.6f}, корректируем")
                                new_sl = current_price * 1.001
                            if new_sl > pos.sl_price:
                                logger.warning(f"{symbol}: Новый SL {new_sl:.6f} выше предыдущего {pos.sl_price:.6f}, корректируем")
                                new_sl = pos.sl_price
                        
                        # Обновляем SL через API
                        if new_sl != pos.sl_price:
                            sl_updated = await self.api.set_stop_loss(symbol, pos.side, new_sl)
                            if sl_updated:
                                pos.sl_price = new_sl
                                pos.last_trailing_update = current_time
                                logger.info(f"{symbol}: SL обновлен для трейлинга на {new_sl:.6f}")
                            else:
                                logger.error(f"{symbol}: Не удалось обновить SL для трейлинга")
                    
                    # Проверка отката от максимума
                    drawdown = pos.max_profit - pnl_usd
                    trailing_trigger = pos.max_profit * TRAILING_DISTANCE_PCT
                    
                    if drawdown >= trailing_trigger:
                        logger.info(f"{symbol}: Трейлинг сработал. Макс: ${pos.max_profit:.2f}, Текущий: ${pnl_usd:.2f}")
                        await self.close_position(symbol, current_price, "TRAILING")
                        continue
                
                # Проверка TP
                if not pos.trailing_active:
                    if pos.side == 'long' and current_price >= pos.tp_price:
                        await self.close_position(symbol, current_price, "TP")
                    elif pos.side == 'short' and current_price <= pos.tp_price:
                        await self.close_position(symbol, current_price, "TP")
                
                # Проверка SL
                if pos.side == 'long' and current_price <= pos.sl_price:
                    await self.close_position(symbol, current_price, "SL")
                elif pos.side == 'short' and current_price >= pos.sl_price:
                    await self.close_position(symbol, current_price, "SL")
                
            except Exception as e:
                logger.error(f"Ошибка обновления позиции {symbol}: {e}")
    
    async def scan_for_entries(self):
        """Сканирование монет для поиска входов"""
        logger.info(f"Сканирование {len(TRADING_SYMBOLS)} монет...")
        current_time = time.time()
        
        for symbol in TRADING_SYMBOLS:
            # Пропускаем если уже в позиции
            if symbol in self.positions:
                continue
            
            # Пропускаем если символ на кулдауне из-за ошибки SL/TP
            if symbol in self.symbol_cooldowns and current_time < self.symbol_cooldowns[symbol]:
                logger.debug(f"{symbol} на кулдауне до {self.symbol_cooldowns[symbol]}")
                continue
            
            try:
                # Получаем данные
                ticker = await self.api.fetch_ticker(symbol)
                candles = await self.api.fetch_ohlcv(symbol, '5m', limit=20)
                
                if not candles or len(candles) < 10:
                    continue
                
                # Анализ сигнала
                signal = await self.analyze_entry(symbol, ticker, candles)
                
                if signal:
                    await self.open_position(signal)
                    
                    # Пауза после открытия позиции
                    await asyncio.sleep(2)
                    
                    # Если достигли лимита позиций - выходим
                    if len(self.positions) >= MAX_POSITIONS:
                        break
                        
            except Exception as e:
                logger.debug(f"Ошибка анализа {symbol}: {e}")
                continue
    
    async def analyze_entry(self, symbol: str, ticker: Dict, candles: List) -> Optional[Dict]:
        """Анализ возможности входа"""
        try:
            price = float(ticker['last'])
            
            # Рассчитываем индикаторы
            closes = [float(c[4]) for c in candles]
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            volumes = [float(c[5]) for c in candles]
            
            # EMA 9 и 21
            ema_9 = self.calculate_ema(closes, 9)
            ema_21 = self.calculate_ema(closes, 21)
            
            # Объем
            avg_volume = sum(volumes[-10:]) / 10
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Momentum
            momentum = closes[-1] - closes[-14] if len(closes) >= 14 else 0
            
            # Волатильность (ATR)
            atr = self.calculate_atr(highs, lows, closes, 14)
            
            # Определение направления
            direction = None
            
            # LONG условия
            if ema_9 > ema_21 and volume_ratio > 1.0 and momentum > 0:
                direction = 'long'
            
            # SHORT условия
            elif ema_9 < ema_21 and volume_ratio > 1.0 and momentum < 0:
                direction = 'short'
            
            if not direction:
                return None
            
            # Проверка через Disco57
            disco_decision = self.disco57.predict(
                price=price,
                volume_ratio=volume_ratio,
                momentum=momentum,
                volatility=atr / price if price > 0 else 0
            )
            
            if disco_decision == 'BLOCK':
                logger.debug(f"{symbol}: Disco57 заблокировал вход")
                return None
            
            # Расчет TP и SL
            tp_price, sl_price = self.calculate_tp_sl(price, direction, atr)
            
            # Проверка минимальной прибыли
            if direction == 'long':
                tp_pct = (tp_price - price) / price
            else:
                tp_pct = (price - tp_price) / price
            
            tp_usd = tp_pct * EFFECTIVE_EXPOSURE
            
            if tp_usd < MIN_PROFIT_USD:
                logger.debug(f"{symbol}: TP ${tp_usd:.2f} < ${MIN_PROFIT_USD}")
                return None
            
            # Проверка SL
            if direction == 'long':
                sl_pct = (price - sl_price) / price
            else:
                sl_pct = (sl_price - price) / price
            
            sl_usd = sl_pct * EFFECTIVE_EXPOSURE
            
            if sl_usd > MAX_LOSS_USD:
                logger.debug(f"{symbol}: SL ${sl_usd:.2f} > ${MAX_LOSS_USD}")
                return None
            
            # Проверка спреда и комиссии
            if 'bid' in ticker and 'ask' in ticker:
                spread = (float(ticker['ask']) - float(ticker['bid'])) / price
                total_cost_pct = spread + BYBIT_FEE_PCT * 2  # Учитываем вход и выход
                if tp_pct <= total_cost_pct:
                    logger.debug(f"{symbol}: TP {tp_pct:.4f} <= спред + комиссия {total_cost_pct:.4f}")
                    return None
                if spread > 0.001:  # 0.1% макс спред
                    logger.debug(f"{symbol}: Спред слишком большой {spread:.4f}")
                    return None
            
            logger.info(f"✅ СИГНАЛ: {symbol} {direction.upper()} @ ${price:.6f}")
            logger.info(f"   TP: ${tp_price:.6f} (+${tp_usd:.2f}) | SL: ${sl_price:.6f} (-${sl_usd:.2f})")
            
            return {
                'symbol': symbol,
                'side': direction,
                'price': price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_usd': tp_usd,
                'sl_usd': sl_usd
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None
    
    async def open_position(self, signal: Dict):
        """Открытие позиции"""
        symbol = signal['symbol']
        side = signal['side']
        price = signal['price']
        
        try:
            # Рассчитываем количество
            quantity = EFFECTIVE_EXPOSURE / price
            
            # Открываем позицию через API
            order = await self.api.create_order(
                symbol=symbol,
                side='buy' if side == 'long' else 'sell',
                amount=quantity,
                price=None,  # Market order
                leverage=LEVERAGE
            )
            
            if not order:
                logger.error(f"Не удалось открыть позицию {symbol}")
                return
            
            # Устанавливаем SL и TP
            sl_set = await self.api.set_stop_loss(symbol, side, signal['sl_price'])
            tp_set = await self.api.set_take_profit(symbol, side, signal['tp_price'])
            
            if not sl_set or not tp_set:
                logger.error(f"Не удалось установить SL/TP для {symbol}. Закрываем позицию.")
                await self.api.close_position(symbol)
                # Добавляем кулдаун для символа
                self.symbol_cooldowns[symbol] = time.time() + SYMBOL_COOLDOWN_SEC
                logger.warning(f"{symbol} добавлен в кулдаун на {SYMBOL_COOLDOWN_SEC} секунд из-за ошибки SL/TP")
                return
            
            # Сохраняем позицию
            pos = Position(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                sl_price=signal['sl_price'],
                tp_price=signal['tp_price'],
                entry_time=time.time(),
                bybit_order_id=order.get('id')
            )
            
            self.positions[symbol] = pos
            self.daily_trades += 1
            
            logger.info(f"✅ ОТКРЫТА ПОЗИЦИЯ: {symbol} {side.upper()}")
            
            # Уведомление в Telegram
            await self.telegram.send_message(
                f"🟢 OPEN\n"
                f"{symbol} {side.upper()}\n"
                f"Вход: ${price:.6f}\n"
                f"SL: -${signal['sl_usd']:.2f}\n"
                f"TP: +${signal['tp_usd']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка открытия позиции {symbol}: {e}")
    
    async def close_position(self, symbol: str, exit_price: float, reason: str):
        """Закрытие позиции"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        try:
            # Закрываем через API
            await self.api.close_position(symbol)
            
            # Ожидаем подтверждения закрытия через API
            start_time = time.time()
            while time.time() - start_time < POSITION_CLOSE_MAX_WAIT:
                positions = await self.api.fetch_positions()
                if not any(p['symbol'] == symbol and float(p.get('contracts', 0)) > 0 for p in positions):
                    logger.info(f"Подтверждено закрытие позиции {symbol} через API")
                    break
                logger.debug(f"Ожидание закрытия позиции {symbol}...")
                await asyncio.sleep(POSITION_CLOSE_CHECK_INTERVAL)
            else:
                logger.warning(f"Время ожидания закрытия {symbol} истекло, возможна фантомная позиция")
            
            # Рассчитываем PnL
            if pos.side == 'long':
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            
            pnl_usd = pnl_pct * EFFECTIVE_EXPOSURE
            self.daily_pnl += pnl_usd
            
            logger.info(f"❌ ЗАКРЫТА ПОЗИЦИЯ: {symbol} | {reason} | PnL: ${pnl_usd:.2f}")
            
            # Уведомление в Telegram
            status_emoji = "✅" if pnl_usd > 0 else "❌"
            await self.telegram.send_message(
                f"{status_emoji} CLOSED ({reason})\n"
                f"{symbol} {pos.side.upper()}\n"
                f"Вход: ${pos.entry_price:.6f}\n"
                f"Выход: ${exit_price:.6f}\n"
                f"PnL: ${pnl_usd:+.2f}\n"
                f"Дневной PnL: ${self.daily_pnl:+.2f}"
            )
            
            # Удаляем из списка
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
    
    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================================
    
    def calculate_ema(self, data: List[float], period: int) -> float:
        """Расчет EMA"""
        if len(data) < period:
            return sum(data) / len(data) if data else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def calculate_atr(self, highs: List[float], lows: List[float], 
                      closes: List[float], period: int = 14) -> float:
        """Расчет ATR"""
        if len(highs) < period + 1:
            return 0
        
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0
        
        atr = sum(trs[:period]) / period
        multiplier = 1 / period
        
        for tr in trs[period:]:
            atr = (tr - atr) * multiplier + atr
        
        return atr


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

async def main():
    """Главная функция"""
    bot = TradeGPTScalper()
    await bot.start()


if __name__ == '__main__':
    asyncio.run(main())
