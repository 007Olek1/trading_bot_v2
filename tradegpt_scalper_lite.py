#!/usr/bin/env python3
"""
TradeGPT Scalper V5 - С Disco57 (DiscoRL) обучением
Работает как в LONG, так и в SHORT
Обучается на каждой свече, адаптируется и становится точнее
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv

from bybit_api import BybitAPI
from telegram_notifier import TelegramNotifier

# Disco57 - адаптивное обучение
try:
    from disco57_learner import Disco57Learner
    DISCO57_AVAILABLE = True
except ImportError:
    DISCO57_AVAILABLE = False
    print("⚠️ Disco57 недоступен")

# База данных истории сделок
try:
    from trade_history_db import TradeHistoryDB
    TRADE_DB_AVAILABLE = True
except ImportError:
    TRADE_DB_AVAILABLE = False
    print("⚠️ TradeHistoryDB недоступен")

# Telegram команды
try:
    from telegram_commands import TelegramCommandsHandler
    TELEGRAM_COMMANDS_AVAILABLE = True
except ImportError:
    TELEGRAM_COMMANDS_AVAILABLE = False
    print("⚠️ Telegram Commands недоступен")

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

# ============================================================================
# КОНСТАНТЫ - ТРЕНДОВАЯ СТРАТЕГИЯ
# ============================================================================

POSITION_SIZE = float(os.getenv('POSITION_SIZE', 1.0))
LEVERAGE = int(os.getenv('LEVERAGE', 25))  # Увеличили до 25x как у конкурентов
EFFECTIVE_EXPOSURE = POSITION_SIZE * LEVERAGE  # $25
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 3))  # 3 позиции одновременно

# ТРЕНДОВАЯ СТРАТЕГИЯ - держим позицию пока тренд идет
# SL в процентах от входа, а не в долларах
SL_PERCENT_STRONG = 0.012  # -1.2% при сильном сигнале
SL_PERCENT_MEDIUM = 0.008  # -0.8% при среднем сигнале
MAX_LOSS_USD = SL_PERCENT_STRONG * EFFECTIVE_EXPOSURE  # Для расчетов по умолчанию

# ============================================================================
# ЗАЩИТА ПРИБЫЛИ - Break-Even + Partial TP + Trailing
# ============================================================================
# Уровень 1: Сужение SL при +0.5%
BREAKEVEN_LEVEL_1_PCT = 0.005  # +0.5% прибыли
BREAKEVEN_SL_1_PCT = 0.005     # SL сужается до -0.5%

# Уровень 2: Безубыток + первый partial TP при +1.2%
BREAKEVEN_LEVEL_2_PCT = 0.012   # +1.2% прибыли
BREAKEVEN_SL_2_PCT = 0.0        # SL переводится в безубыток
PARTIAL_TP_LEVEL_1_PCT = 0.012
PARTIAL_TP_LEVEL_1_FRACTION = 0.20

# Уровень 3: Доп. частичное закрытие и защита прибыли при +1.8%
PARTIAL_TP_LEVEL_2_PCT = 0.018
PARTIAL_TP_LEVEL_2_FRACTION = 0.40
PROFIT_LOCK_LEVEL_PCT = 0.018
PROFIT_LOCK_SL_PCT = 0.005      # фиксируем минимум +0.5%

# Уровень 4: Trailing активируется при +2%
TRAILING_ACTIVATION_PCT = 0.02  # +2% прибыли для активации trailing
TRAILING_DISTANCE_PCT = float(os.getenv('TRAILING_DISTANCE_PCT', '0.005'))  # 0.5% trailing distance

# Уровень 5: Жесткий trailing при +3%
TRAILING_TIGHT_LEVEL_PCT = 0.03  # +3% прибыли
TRAILING_TIGHT_DISTANCE_PCT = float(os.getenv('TRAILING_TIGHT_DISTANCE_PCT', '0.003'))  # 0.3% trailing distance

# Дневной лимит
DAILY_MAX_LOSS_USD = float(os.getenv('DAILY_MAX_LOSS_USD', 5.0))

# Интервалы
SCAN_INTERVAL_SEC = int(os.getenv('SCAN_INTERVAL_SEC', 60))  # Чаще сканируем для трендов
POSITION_CHECK_INTERVAL = 10  # Проверяем позиции каждые 10 сек
BYBIT_FEE_PCT = 0.00075
SYMBOL_COOLDOWN_SEC = 300  # 5 минут кулдаун после закрытия
SYMBOL_ENTRY_COOLDOWN_SEC = int(os.getenv('SYMBOL_ENTRY_COOLDOWN_SEC', 1800))
SECTOR_ENTRY_COOLDOWN_SEC = int(os.getenv('SECTOR_ENTRY_COOLDOWN_SEC', 900))
MEME_SECTOR_COOLDOWN_SEC = int(os.getenv('MEME_SECTOR_COOLDOWN_SEC', 1800))
POSITION_CLOSE_CHECK_INTERVAL = 2
POSITION_CLOSE_MAX_WAIT = 30

# ФИЛЬТРЫ ДЛЯ ТРЕНДОВ
MIN_VOLUME_24H_USD = 10_000_000
MAX_SPREAD_PCT = 0.001
MIN_SIGNAL_STRENGTH = 3
MEME_MIN_SIGNAL_STRENGTH = int(os.getenv('MEME_MIN_SIGNAL_STRENGTH', MIN_SIGNAL_STRENGTH + 1))
DISCO57_MIN_CONFIDENCE = 0.7
MEME_MIN_DISCO_CONFIDENCE = float(os.getenv('MEME_MIN_DISCO_CONFIDENCE', 0.8))
MIN_ATR_PCT = 0.004  # Минимальная волатильность 0.4%
MIN_RANGE_PCT = 0.006  # Диапазон движения за последние свечи минимум 0.6%

MAINTENANCE_INTERVAL_SEC = int(os.getenv('MAINTENANCE_INTERVAL_SEC', 1800))
DAILY_REPORT_HOUR = int(os.getenv('DAILY_REPORT_HOUR', 9))
BACKFILL_LOOKBACK_HOURS = int(os.getenv('BACKFILL_LOOKBACK_HOURS', 24))

# Минимальный тренд для входа
MIN_TREND_STRENGTH_PCT = 0.005  # Минимум 0.5% движение для подтверждения тренда

# Ограничения по секторам и риск-менеджменту
MEME_SYMBOLS = {
    'DOGE', 'SHIB', 'PEPE', 'BONK', 'FLOKI', '1000PEPE', '1000BONK', '1000FLOKI',
    '1000TURBO', '1000000MOG', 'WIF', 'MEME', 'BOME', 'NOT', 'MOG'
}
SYMBOL_SECTOR_MAP = {symbol: 'MEME' for symbol in MEME_SYMBOLS}
MAX_SECTOR_POSITIONS = {
    'MEME': 1,
}
LOSS_STREAK_THRESHOLD = 2
LOSS_STREAK_SIZE_MULTIPLIER = 0.5

# УМНЫЙ СЕЛЕКТОР: 145 МОНЕТ
# Загружаем из файла или используем базовый список
try:
    from SYMBOLS_145 import TRADING_SYMBOLS_145
    TRADING_SYMBOLS = TRADING_SYMBOLS_145
    print(f"✅ Загружено {len(TRADING_SYMBOLS)} символов из SYMBOLS_145")
except ImportError:
    # Базовый список ТОП-50 монет
    TRADING_SYMBOLS = [
        # ТОП-20 по капитализации
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
        'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT',
        'DOT/USDT:USDT', 'NEAR/USDT:USDT', 'LTC/USDT:USDT', 'BCH/USDT:USDT',
        'UNI/USDT:USDT', 'APT/USDT:USDT', 'OP/USDT:USDT', 'ARB/USDT:USDT',
        'SUI/USDT:USDT', 'INJ/USDT:USDT', 'TIA/USDT:USDT', 'SEI/USDT:USDT',
        # Дополнительные ликвидные
        'ATOM/USDT:USDT', 'FIL/USDT:USDT', 'IMX/USDT:USDT', 'RUNE/USDT:USDT',
        'GRT/USDT:USDT', 'AAVE/USDT:USDT', 'MKR/USDT:USDT', 'SNX/USDT:USDT',
        'CRV/USDT:USDT', 'LDO/USDT:USDT', 'ENS/USDT:USDT', 'DYDX/USDT:USDT',
        'GMX/USDT:USDT', 'BLUR/USDT:USDT', 'WLD/USDT:USDT', 'JUP/USDT:USDT',
        'PYTH/USDT:USDT', 'STRK/USDT:USDT', 'MANTA/USDT:USDT', 'DYM/USDT:USDT',
        'ORDI/USDT:USDT', 'WIF/USDT:USDT', '1000PEPE/USDT:USDT', '1000BONK/USDT:USDT',
        '1000FLOKI/USDT:USDT', 'MEME/USDT:USDT', 'BOME/USDT:USDT', 'NOT/USDT:USDT',
        'TON/USDT:USDT', 'ENA/USDT:USDT',
    ]
    print(f"⚠️ Используется базовый список: {len(TRADING_SYMBOLS)} символов")

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class Position:
    """Активная позиция"""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    sl_price: float
    tp_price: float
    entry_time: float
    trailing_active: bool = False
    max_profit: float = 0.0
    current_pnl: float = 0.0
    bybit_order_id: Optional[str] = None
    last_trailing_update: float = 0.0
    breakeven_level_1_hit: bool = False
    breakeven_level_2_hit: bool = False
    partial_tp_level_1_done: bool = False
    partial_tp_level_2_done: bool = False
    profit_lock_applied: bool = False
    original_quantity: float = 0.0
    tight_trailing: bool = False
    entry_rsi: float = 50.0
    sl_pct: float = SL_PERCENT_STRONG

# ============================================================================
# ОСНОВНОЙ КЛАСС БОТА (LITE VERSION)
# ============================================================================

class TradeGPTScalperLite:
    """TradeGPT Scalper V5 с Disco57 (DiscoRL) обучением"""
    
    def __init__(self):
        self.api = BybitAPI()
        self.telegram = TelegramNotifier()
        
        # Disco57 - адаптивное обучение
        self.disco57 = None
        if DISCO57_AVAILABLE:
            try:
                self.disco57 = Disco57Learner()
                logger.info(f"✅ Disco57 активен | Win Rate: {self.disco57.get_win_rate():.1f}%")
            except Exception as e:
                logger.warning(f"Disco57 не инициализирован: {e}")
        
        # База данных истории сделок (72 часа с авторотацией)
        self.trade_db = None
        if TRADE_DB_AVAILABLE:
            try:
                self.trade_db = TradeHistoryDB()
                logger.info("✅ TradeHistoryDB активна (72ч ротация)")
            except Exception as e:
                logger.warning(f"TradeHistoryDB не инициализирована: {e}")
        
        # Telegram команды
        self.telegram_commands = None
        self.trading_enabled = True  # Флаг для /stop и /resume
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.symbol_cooldowns: Dict[str, float] = {}
        self.symbol_last_entry: Dict[str, float] = {}
        self.sector_last_entry: Dict[str, float] = {}
        self.loss_streak = 0
        
        # Кэш признаков для обучения
        self.trade_features_cache: Dict[str, any] = {}
        self.last_daily_summary_date = None
        self._maintenance_task = None
        
        logger.info(f"TradeGPT Trend Trader V5 инициализирован")
        logger.info(f"• Умный селектор: {len(TRADING_SYMBOLS)} монет")
        logger.info(f"• TradeGPT логика: ✅")
        logger.info(f"• Disco57 (DiscoRL): {'✅' if self.disco57 else '❌'}")
        logger.info(f"Позиция: ${POSITION_SIZE} x{LEVERAGE} = ${EFFECTIVE_EXPOSURE}")
        logger.info(
            f"SL: strong -{SL_PERCENT_STRONG*100:.1f}% | medium -{SL_PERCENT_MEDIUM*100:.1f}% | "
            f"Trailing: +{TRAILING_ACTIVATION_PCT*100:.1f}%"
        )
    
    async def start(self):
        """Запуск бота"""
        logger.info("=" * 60)
        logger.info("TradeGPT Scalper Lite запущен")
        logger.info("=" * 60)
        
        # Инициализация Telegram команд
        if TELEGRAM_COMMANDS_AVAILABLE:
            try:
                telegram_token = os.getenv('TELEGRAM_TOKEN')
                telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
                self.telegram_commands = TelegramCommandsHandler(
                    bot_instance=self,
                    telegram_token=telegram_token,
                    chat_id=telegram_chat_id
                )
                await self.telegram_commands.setup_commands()
                logger.info("✅ Telegram команды активированы")
            except Exception as e:
                logger.warning(f"Telegram команды не инициализированы: {e}")
        
        await self.telegram.send_startup_message()
        await self.load_active_positions()
        try:
            await self.backfill_recent_trades()
        except Exception as e:
            logger.warning(f"Первичный бекфилл не выполнен: {e}")
        self._maintenance_task = asyncio.create_task(self.maintenance_loop())
        
        try:
            while True:
                # Проверка флага торговли от Telegram команд
                if self.telegram_commands and not self.telegram_commands.is_trading_enabled():
                    await asyncio.sleep(SCAN_INTERVAL_SEC)
                    continue
                
                await self.main_loop()
                await asyncio.sleep(SCAN_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Бот остановлен пользователем")
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}", exc_info=True)
            await self.telegram.send_error(f"Критическая ошибка: {e}")
        finally:
            if self._maintenance_task:
                self._maintenance_task.cancel()
                try:
                    await self._maintenance_task
                except asyncio.CancelledError:
                    pass
            # Остановка Telegram команд
            if self.telegram_commands:
                await self.telegram_commands.shutdown()
            await self.api.close()
    
    async def load_active_positions(self):
        """Загрузка активных позиций при старте с SL/TP напрямую через Bybit API"""
        try:
            # Прямой запрос к Bybit API (не через ccxt)
            positions_data = await self.api.get_positions_with_sl_tp()
            
            for pos in positions_data:
                size = float(pos.get("size", 0))
                if size > 0:
                    bybit_symbol = pos.get("symbol")  # ALLOUSDT
                    # Конвертируем в ccxt формат
                    symbol = bybit_symbol.replace("USDT", "/USDT:USDT")
                    
                    raw_side = pos.get("side", "")
                    if raw_side == "Buy":
                        side = "long"
                    elif raw_side == "Sell":
                        side = "short"
                    else:
                        continue
                    
                    entry_price = float(pos.get("avgPrice", 0))
                    sl_str = pos.get("stopLoss", "")
                    tp_str = pos.get("takeProfit", "")
                    sl_price = float(sl_str) if sl_str and sl_str != "" else 0
                    tp_price = float(tp_str) if tp_str and tp_str != "" else 0
                    
                    # Определяем уровни защиты на основе текущего SL
                    breakeven_1 = False
                    breakeven_2 = False
                    sl_pct_value = SL_PERCENT_STRONG
                    if sl_price > 0 and entry_price > 0:
                        if side == "long":
                            sl_pct = (entry_price - sl_price) / entry_price
                            sl_pct_value = max(0.002, sl_pct)
                        else:
                            sl_pct = (sl_price - entry_price) / entry_price
                            sl_pct_value = max(0.002, sl_pct)
                        
                        # Если SL уже в плюсе - уровень 2 достигнут
                        if sl_pct >= 0:
                            breakeven_2 = True
                            breakeven_1 = True
                        # Если SL сужен до -0.75% - уровень 1 достигнут
                        elif sl_pct >= -0.0075:
                            breakeven_1 = True
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        quantity=size,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        entry_time=time.time(),
                        breakeven_level_1_hit=breakeven_1,
                        breakeven_level_2_hit=breakeven_2,
                        original_quantity=size,
                        sl_pct=sl_pct_value
                    )
                    self._record_entry_timestamp(symbol)
                    
                    sl_info = f"${sl_price:.6f}" if sl_price > 0 else "НЕТ"
                    logger.info(f"Загружена позиция: {symbol} {side.upper()} @ {entry_price} | SL: {sl_info}")
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки позиций: {e}")
    
    async def main_loop(self):
        """Основной цикл бота"""
        await self.reset_daily_stats()
        
        if self.daily_pnl <= -DAILY_MAX_LOSS_USD:
            logger.warning(f"Достигнут дневной лимит убытка: ${self.daily_pnl:.2f}")
            await self.telegram.send_daily_limit_reached(self.daily_pnl)
            await asyncio.sleep(3600)
            return
        
        await self.update_positions()
        
        if len(self.positions) < MAX_POSITIONS:
            await self.scan_for_entries()
    
    async def reset_daily_stats(self):
        """Сброс дневной статистики"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            logger.info(f"Сброс дневной статистики. Вчера: PnL ${self.daily_pnl:.2f}, Сделок: {self.daily_trades}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
    
    async def scan_for_entries(self):
        """Сканирование монет для поиска входов"""
        if len(self.positions) >= MAX_POSITIONS:
            return False
        
        logger.info(f"Сканирование {len(TRADING_SYMBOLS)} монет...")
        current_time = time.time()
        candidates: List[Dict] = []
        
        for symbol in TRADING_SYMBOLS:
            if symbol in self.positions:
                continue
            now = time.time()
            normalized = self._normalize_symbol(symbol)
            last_entry_ts = self.symbol_last_entry.get(normalized)
            if last_entry_ts and now - last_entry_ts < SYMBOL_ENTRY_COOLDOWN_SEC:
                logger.debug(
                    f"{symbol}: на кулдауне после предыдущего входа ещё {int(SYMBOL_ENTRY_COOLDOWN_SEC - (now - last_entry_ts))}с"
                )
                continue

            sector = self._get_sector(symbol)
            if sector:
                sector_cooldown = MEME_SECTOR_COOLDOWN_SEC if sector == 'MEME' else SECTOR_ENTRY_COOLDOWN_SEC
                last_sector_ts = self.sector_last_entry.get(sector)
                if last_sector_ts and now - last_sector_ts < sector_cooldown:
                    logger.debug(
                        f"{symbol}: сектор {sector} на кулдауне ещё {int(sector_cooldown - (now - last_sector_ts))}с"
                    )
                    continue
                max_sector_positions = MAX_SECTOR_POSITIONS.get(sector, 1)
                if self._sector_position_count(sector) >= max_sector_positions:
                    logger.debug(f"{symbol}: сектор {sector} уже занят ({max_sector_positions})")
                    continue

            if symbol in self.symbol_cooldowns and current_time < self.symbol_cooldowns[symbol]:
                continue
            
            try:
                ticker = await self.api.fetch_ticker(symbol)
                candles = await self.api.fetch_ohlcv(symbol, '5m', limit=20)
                
                if not candles or len(candles) < 10:
                    continue
                
                signal = await self.analyze_entry(symbol, ticker, candles)
                
                if signal:
                    signal['scan_timestamp'] = now
                    signal['sector'] = sector
                    candidates.append(signal)
                        
            except Exception as e:
                logger.debug(f"Ошибка анализа {symbol}: {e}")
                continue
        
        if not candidates:
            return False
        
        available_slots = MAX_POSITIONS - len(self.positions)
        if available_slots <= 0:
            return False
        
        candidates.sort(
            key=lambda s: (
                s.get('signal_strength', 0),
                s.get('disco_confidence', 0.0)
            ),
            reverse=True
        )
        
        opened = False
        for signal in candidates:
            if available_slots <= 0:
                break
            
            symbol = signal['symbol']
            if symbol in self.positions:
                continue
            
            now = time.time()
            normalized = self._normalize_symbol(symbol)
            last_entry_ts = self.symbol_last_entry.get(normalized)
            if last_entry_ts and now - last_entry_ts < SYMBOL_ENTRY_COOLDOWN_SEC:
                continue
            
            sector = signal.get('sector') or self._get_sector(symbol)
            if sector:
                sector_cooldown = MEME_SECTOR_COOLDOWN_SEC if sector == 'MEME' else SECTOR_ENTRY_COOLDOWN_SEC
                last_sector_ts = self.sector_last_entry.get(sector)
                if last_sector_ts and now - last_sector_ts < sector_cooldown:
                    continue
                max_sector_positions = MAX_SECTOR_POSITIONS.get(sector, 1)
                if self._sector_position_count(sector) >= max_sector_positions:
                    continue
            
            await self.open_position(signal)
            available_slots -= 1
            opened = True
            await asyncio.sleep(2)
        
        return opened
    
    async def analyze_entry(self, symbol: str, ticker: Dict, candles: List) -> Optional[Dict]:
        """Анализ возможности входа с ЖЕСТКИМИ ФИЛЬТРАМИ"""
        try:
            price = float(ticker['last'])
            sector = self._get_sector(symbol)
            is_meme = sector == 'MEME'
            
            # ========== ПРОВЕРКА СПРЕДА ==========
            if 'bid' in ticker and 'ask' in ticker:
                bid = float(ticker['bid'])
                ask = float(ticker['ask'])
                spread = (ask - bid) / price
                
                # ЖЕСТКИЙ ФИЛЬТР: спред должен быть < 0.05%
                if spread > MAX_SPREAD_PCT:
                    logger.debug(f"{symbol}: Спред слишком большой: {spread*100:.3f}%")
                    return None
            else:
                # Нет данных о спреде - пропускаем
                return None
            
            closes = [float(c[4]) for c in candles]
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            volumes = [float(c[5]) for c in candles]
            
            # ========== ПРОВЕРКА ОБЪЕМА ==========
            total_volume_usd = sum(volumes[-20:]) * price
            if total_volume_usd < MIN_VOLUME_24H_USD / 24:  # Примерно за час
                logger.debug(f"{symbol}: Низкий объем: ${total_volume_usd:,.0f}")
                return None
            
            # ========== РАСЧЕТ ИНДИКАТОРОВ ==========
            ema_9 = self.calculate_ema(closes, 9)
            ema_21 = self.calculate_ema(closes, 21)
            ema_50 = self.calculate_ema(closes, 18)  # Для тренда
            
            avg_volume = sum(volumes[-10:]) / 10
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            momentum = closes[-1] - closes[-14] if len(closes) >= 14 else 0
            momentum_pct = momentum / closes[-14] if len(closes) >= 14 and closes[-14] > 0 else 0
            
            atr = self.calculate_atr(highs, lows, closes, 14)
            atr_pct = atr / price if price > 0 else 0
            price_range = max(highs[-12:]) - min(lows[-12:]) if len(highs) >= 12 else 0
            range_pct = (price_range / price) if price > 0 else 0
            if atr_pct < MIN_ATR_PCT or range_pct < MIN_RANGE_PCT:
                logger.debug(
                    f"{symbol}: Волатильность/диапазон слишком низкие (ATR {atr_pct*100:.2f}%, Range {range_pct*100:.2f}%)"
                )
                return None
            
            # RSI расчет
            rsi = self.calculate_rsi(closes, 14)
            
            # ========== ПОДСЧЕТ СИЛЫ СИГНАЛА ==========
            signal_strength = 0
            direction = None
            
            # Проверка для LONG
            if ema_9 > ema_21:
                signal_strength += 1  # EMA бычий
                if price > ema_50:
                    signal_strength += 1  # Цена выше тренда
                if momentum_pct > 0.003:  # Импульс > 0.3%
                    signal_strength += 1
                if volume_ratio > 0.9:  # Объем >= 90% от среднего
                    signal_strength += 1
                if 30 < rsi < 70:  # RSI не в экстремуме
                    signal_strength += 1
                
                if signal_strength >= MIN_SIGNAL_STRENGTH:
                    direction = 'long'
            
            # Проверка для SHORT
            elif ema_9 < ema_21:
                signal_strength += 1  # EMA медвежий
                if price < ema_50:
                    signal_strength += 1  # Цена ниже тренда
                if momentum_pct < -0.003:  # Импульс < -0.3%
                    signal_strength += 1
                if volume_ratio > 0.9:  # Объем >= 90% от среднего
                    signal_strength += 1
                if 30 < rsi < 70:
                    signal_strength += 1
                
                if signal_strength >= MIN_SIGNAL_STRENGTH:
                    direction = 'short'
            
            # Нет сигнала или слабый сигнал
            if not direction:
                return None

            if is_meme and signal_strength < MEME_MIN_SIGNAL_STRENGTH:
                logger.debug(
                    f"{symbol}: MEME сигнал ({signal_strength}) < требуемых {MEME_MIN_SIGNAL_STRENGTH}"
                )
                return None
            
            # ========== РАСЧЕТ TP/SL ==========
            sl_pct = SL_PERCENT_STRONG if signal_strength >= MIN_SIGNAL_STRENGTH + 1 else SL_PERCENT_MEDIUM
            tp_price, sl_price = self.calculate_tp_sl(price, direction, sl_pct)
            
            # ТРЕНДОВАЯ СТРАТЕГИЯ: SL в процентах, без фиксированного TP
            sl_usd = sl_pct * EFFECTIVE_EXPOSURE
            
            # ========== ФИНАЛЬНЫЕ ПРОВЕРКИ ДЛЯ ТРЕНДА ==========
            
            # Проверяем что спред не съест прибыль
            total_cost_pct = spread + BYBIT_FEE_PCT * 2
            if total_cost_pct > 0.002:  # Если издержки > 0.2% - пропускаем
                logger.debug(f"{symbol}: Издержки слишком высокие: {total_cost_pct*100:.2f}%")
                return None
            
            # ========== DISCO57 ФИЛЬТР ==========
            disco_confidence = 0.0
            disco_allow = True
            
            if self.disco57:
                try:
                    features = self.disco57.extract_features(candles, ticker)
                    orderbook = await self.api.fetch_order_book(symbol, limit=5)
                    bids = orderbook.get('bids', []) or []
                    asks = orderbook.get('asks', []) or []
                    bid_vol = sum(size for _, size in bids)
                    ask_vol = sum(size for _, size in asks)
                    total_vol = bid_vol + ask_vol
                    if total_vol > 0:
                        imbalance = bid_vol / total_vol
                        delta = (bid_vol - ask_vol) / total_vol
                        delta = (delta + 1) / 2  # Приводим к 0-1
                    else:
                        imbalance = 0.5
                        delta = 0.5
                    features.book_imbalance = imbalance
                    features.book_delta = delta
                    disco_allow, disco_confidence = self.disco57.predict(features, direction)
                    
                    if not disco_allow:
                        logger.debug(f"{symbol}: Disco57 BLOCK (confidence: {disco_confidence:.2f})")
                        return None
                    if disco_confidence < DISCO57_MIN_CONFIDENCE:
                        logger.debug(
                            f"{symbol}: Disco57 confidence {disco_confidence:.2f} < {DISCO57_MIN_CONFIDENCE:.2f}, пропускаем"
                        )
                        return None
                    if is_meme and disco_confidence < MEME_MIN_DISCO_CONFIDENCE:
                        logger.debug(
                            f"{symbol}: MEME требует Disco57 >= {MEME_MIN_DISCO_CONFIDENCE:.2f} (получено {disco_confidence:.2f})"
                        )
                        return None
                    
                    # Сохраняем признаки для обучения после закрытия
                    self.trade_features_cache[symbol] = {
                        'features': features,
                        'direction': direction,
                        'entry_time': time.time()
                    }
                except Exception as e:
                    logger.warning(f"Disco57 ошибка для {symbol}: {e}")
            
            disco_str = f" | Disco57: {disco_confidence:.0%}" if self.disco57 else ""
            logger.info(f"✅ ТРЕНД: {symbol} {direction.upper()} @ ${price:.4f} | Сила: {signal_strength}/5{disco_str}")
            
            return {
                'symbol': symbol,
                'side': direction,
                'price': price,
                'tp_price': 0,  # Нет фиксированного TP - trailing!
                'sl_price': sl_price,
                'sl_usd': sl_usd,
                'sl_pct': sl_pct,
                'signal_strength': signal_strength,
                'disco_confidence': disco_confidence,
                'entry_rsi': rsi
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None
    
    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Расчет RSI"""
        if len(closes) < period + 1:
            return 50.0  # Нейтральное значение
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_ema(self, data: List[float], period: int) -> float:
        """Расчет EMA"""
        if len(data) < period:
            return sum(data) / len(data) if data else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Расчет ATR"""
        if len(closes) < period + 1:
            return 0.0
        
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0
        
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = ((period - 1) * atr + tr) / period
        
        return atr
    
    def calculate_tp_sl(self, price: float, direction: str, sl_pct: float) -> tuple:
        """
        ТРЕНДОВАЯ СТРАТЕГИЯ:
        - SL фиксированный в процентах (динамический)
        - TP = 0 (нет фиксированного TP, используем trailing)
        """
        if direction == 'long':
            sl_price = price * (1 - sl_pct)
            tp_price = 0  # Нет фиксированного TP - trailing!
        else:
            sl_price = price * (1 + sl_pct)
            tp_price = 0  # Нет фиксированного TP - trailing!
        
        return tp_price, sl_price
    
    def _normalize_symbol(self, symbol: str) -> str:
        base = symbol.split('/')[0]
        base = base.replace(':USDT', '').replace('USDT', '')
        return base.upper()

    def _record_entry_timestamp(self, symbol: str, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        normalized = self._normalize_symbol(symbol)
        self.symbol_last_entry[normalized] = timestamp
        sector = self._get_sector(symbol)
        if sector:
            self.sector_last_entry[sector] = timestamp

    def _get_sector(self, symbol: str) -> Optional[str]:
        base = self._normalize_symbol(symbol)
        return SYMBOL_SECTOR_MAP.get(base)
    
    def _sector_position_count(self, sector: str) -> int:
        count = 0
        for pos in self.positions.values():
            if self._get_sector(pos.symbol) == sector:
                count += 1
        return count
    
    async def open_position(self, signal: Dict):
        """Открытие позиции с точным размером"""
        symbol = signal['symbol']
        side = signal['side']
        price = signal['price']
        
        try:
            # Расчет количества контрактов с запасом 5% для компенсации округления
            # EFFECTIVE_EXPOSURE = $20, но Bybit округляет вниз
            size_multiplier = LOSS_STREAK_SIZE_MULTIPLIER if self.loss_streak >= LOSS_STREAK_THRESHOLD else 1.0
            target_exposure = EFFECTIVE_EXPOSURE * size_multiplier * 1.05  # запас 5%
            if size_multiplier < 1.0:
                logger.info(
                    f"{symbol}: уменьшенный размер позиции (стрик убытков {self.loss_streak}). "
                    f"Экспозиция x{size_multiplier:.2f}"
                )
            quantity = target_exposure / price
            
            # Логируем расчет
            actual_exposure = quantity * price
            logger.info(f"{symbol}: Расчет позиции: {quantity:.6f} контрактов = ${actual_exposure:.2f}")
            
            order = await self.api.create_order(
                symbol=symbol,
                side='buy' if side == 'long' else 'sell',
                amount=quantity,
                price=None,
                leverage=LEVERAGE
            )
            
            if not order:
                logger.error(f"Не удалось открыть позицию {symbol}")
                return
            
            # Проверяем реальный размер позиции
            filled_qty = order.get('filled') or order.get('amount') or quantity
            if filled_qty:
                filled_qty = float(filled_qty)
            else:
                filled_qty = quantity
            real_exposure = filled_qty * price
            logger.info(f"{symbol}: Реальная позиция: {filled_qty:.6f} = ${real_exposure:.2f}")
            
            # ТРЕНДОВАЯ СТРАТЕГИЯ: только SL, без фиксированного TP
            sl_set = await self.api.set_stop_loss(symbol, side, signal['sl_price'])
            
            if not sl_set:
                logger.error(f"Не удалось установить SL для {symbol}. Закрываем позицию.")
                await self.api.close_position(symbol)
                self.symbol_cooldowns[symbol] = time.time() + SYMBOL_COOLDOWN_SEC
                return
            
            sl_pct = signal.get('sl_pct', SL_PERCENT_STRONG)
            sl_usd = sl_pct * EFFECTIVE_EXPOSURE
            
            pos = Position(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                sl_price=signal['sl_price'],
                tp_price=0,  # Нет фиксированного TP - trailing!
                entry_time=time.time(),
                bybit_order_id=order.get('id'),
                sl_pct=sl_pct,
                entry_rsi=signal.get('entry_rsi', 50.0)
            )
            pos.original_quantity = quantity
            
            self.positions[symbol] = pos
            self.daily_trades += 1
            self._record_entry_timestamp(symbol)
            
            # Записать в базу данных
            if self.trade_db:
                self.trade_db.add_trade_open(
                    symbol=symbol,
                    side=side,
                    entry_price=price,
                    quantity=quantity,
                    signal_strength=signal.get('signal_strength', 0),
                    disco_confidence=signal.get('disco_confidence', 0)
                )
            
            logger.info(
                f"✅ ОТКРЫТА ПОЗИЦИЯ: {symbol} {side.upper()} | SL: -{sl_pct*100:.2f}% | "
                f"Trailing: +{TRAILING_ACTIVATION_PCT*100:.1f}%"
            )
            
            await self.telegram.send_trade_opened(
                symbol=symbol,
                side=side,
                entry_price=price,
                sl_usd=sl_usd,
                tp_usd=0,  # Нет фиксированного TP
                sl_price=signal['sl_price'],
                signal_strength=signal.get('signal_strength', 0),
                disco_confidence=signal.get('disco_confidence', 0)
            )
            
        except Exception as e:
            logger.error(f"Ошибка открытия позиции {symbol}: {e}")
    
    async def close_position(self, symbol: str, exit_price: float, reason: str):
        """Закрытие позиции"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        try:
            await self.api.close_position(symbol)
            
            start_time = time.time()
            while time.time() - start_time < POSITION_CLOSE_MAX_WAIT:
                positions = await self.api.fetch_positions()
                if not any(p['symbol'] == symbol and float(p.get('contracts', 0)) > 0 for p in positions):
                    logger.info(f"Подтверждено закрытие позиции {symbol} через API")
                    break
                await asyncio.sleep(POSITION_CLOSE_CHECK_INTERVAL)
            
            if pos.side == 'long':
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            
            pnl_usd = pnl_pct * EFFECTIVE_EXPOSURE
            self.daily_pnl += pnl_usd
            if pnl_usd < 0:
                self.loss_streak = min(self.loss_streak + 1, 5)
            else:
                self.loss_streak = 0
            
            # ========== DISCO57 ОБУЧЕНИЕ ==========
            if self.disco57 and symbol in self.trade_features_cache:
                try:
                    cached = self.trade_features_cache[symbol]
                    features = cached['features']
                    direction = cached['direction']
                    entry_ts = cached.get('entry_time', pos.entry_time)
                    duration_sec = max(1.0, time.time() - entry_ts)
                    if pnl_usd < 0:
                        stop_speed = 1 - min(duration_sec / 120, 1.0)
                    else:
                        stop_speed = min(duration_sec / 600, 1.0) * 0.2
                    features.stop_speed = max(0.0, min(stop_speed, 1.0))
                    self.disco57.learn(
                        features=features,
                        direction=direction,
                        pnl=pnl_usd
                    )
                    del self.trade_features_cache[symbol]
                    logger.info(f"🤖 Disco57 обучен на {symbol} | Win Rate: {self.disco57.get_win_rate():.1f}%")
                except Exception as e:
                    logger.warning(f"Disco57 ошибка обучения: {e}")
            
            # Записать в базу данных
            if self.trade_db:
                self.trade_db.close_trade(
                    symbol=symbol,
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    reason=reason,
                    trailing_activated=pos.trailing_active
                )
            
            logger.info(f"❌ ЗАКРЫТА ПОЗИЦИЯ: {symbol} | {reason} | PnL: ${pnl_usd:.2f}")
            
            await self.telegram.send_trade_closed(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                reason=reason,
                daily_pnl=self.daily_pnl
            )
            
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
    
    async def update_positions(self):
        """
        ЗАЩИТА ПРИБЫЛИ - Break-Even + Partial TP + Trailing
        
        Уровни защиты:
        +0.5%  → SL сужается до -0.5%
        +1.2% → Безубыток + закрыть 20%
        +1.8% → Profit lock (+0.5%) + доп. partial 40%
        +2.0% → Trailing активен (шаг 0.75%)
        +3.0% → Жесткий trailing (шаг <1%)
        """
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            
            try:
                ticker = await self.api.fetch_ticker(symbol)
                current_price = float(ticker['last'])
                
                # Расчет текущего PnL в %
                if pos.side == 'long':
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - current_price) / pos.entry_price
                
                pnl_usd = pnl_pct * EFFECTIVE_EXPOSURE
                pos.current_pnl = pnl_usd
                
                # ========== УРОВЕНЬ 1: +0.5% - Сужение SL ==========
                if not pos.breakeven_level_1_hit and pnl_pct >= BREAKEVEN_LEVEL_1_PCT:
                    pos.breakeven_level_1_hit = True
                    
                    # Новый SL = -0.5% от входа
                    if pos.side == 'long':
                        new_sl = pos.entry_price * (1 - BREAKEVEN_SL_1_PCT)
                    else:
                        new_sl = pos.entry_price * (1 + BREAKEVEN_SL_1_PCT)
                    
                    sl_updated = await self.api.set_stop_loss(symbol, pos.side, new_sl)
                    if sl_updated:
                        old_sl = pos.sl_price
                        pos.sl_price = new_sl
                        logger.info(f"🛡️ {symbol}: SL сужен при +0.5% | ${old_sl:.4f} → ${new_sl:.4f}")
                        await self.telegram.send_message(
                            f"🛡️ <b>{symbol}</b> +0.5%\nSL сужен до -0.5%"
                        )
                
                # ========== УРОВЕНЬ 2: +1.2% - Безубыток + Partial TP (20%) ==========
                if not pos.breakeven_level_2_hit and pnl_pct >= BREAKEVEN_LEVEL_2_PCT:
                    pos.breakeven_level_2_hit = True
                    
                    # Перевод SL в безубыток
                    new_sl = pos.entry_price
                    if pos.side == 'short':
                        new_sl = pos.entry_price
                    
                    sl_updated = await self.api.set_stop_loss(symbol, pos.side, new_sl)
                    if sl_updated:
                        pos.sl_price = new_sl
                        logger.info(f"🔒 {symbol}: SL переведен в безубыток при +1.2% | SL = ${new_sl:.4f}")
                    
                    if not pos.partial_tp_level_1_done and pos.quantity > 0:
                        partial_qty = min(pos.original_quantity * PARTIAL_TP_LEVEL_1_FRACTION, pos.quantity)
                        if partial_qty > 0:
                            try:
                                close_side = 'sell' if pos.side == 'long' else 'buy'
                                await self.api.create_order(
                                    symbol=symbol,
                                    side=close_side,
                                    amount=partial_qty,
                                    reduce_only=True
                                )
                                pos.partial_tp_level_1_done = True
                                pos.quantity = max(0.0, pos.quantity - partial_qty)
                                partial_pnl = pnl_pct * EFFECTIVE_EXPOSURE * PARTIAL_TP_LEVEL_1_FRACTION
                                self.daily_pnl += partial_pnl
                                logger.info(f"💰 {symbol}: PARTIAL TP 20% | +${partial_pnl:.2f}")
                                await self.telegram.send_message(
                                    f"💰 <b>{symbol}</b> PARTIAL TP #1\n"
                                    f"Закрыто 20% при +1.2%\n"
                                    f"Прибыль: +${partial_pnl:.2f}\n"
                                    f"Остаток защищен безубытком"
                                )
                            except Exception as e:
                                logger.error(f"Ошибка partial TP #1 {symbol}: {e}")
                
                # ========== УРОВЕНЬ 3: +1.8% - Profit lock + Partial TP (40%) ==========
                if pnl_pct >= PARTIAL_TP_LEVEL_2_PCT:
                    if not pos.profit_lock_applied:
                        if pos.side == 'long':
                            new_sl = pos.entry_price * (1 + PROFIT_LOCK_SL_PCT)
                        else:
                            new_sl = pos.entry_price * (1 - PROFIT_LOCK_SL_PCT)
                        sl_updated = await self.api.set_stop_loss(symbol, pos.side, new_sl)
                        if sl_updated:
                            pos.sl_price = new_sl
                            pos.profit_lock_applied = True
                            logger.info(f"🔐 {symbol}: Profit lock активирован (+0.5%)")
                            await self.telegram.send_message(
                                f"🔐 <b>{symbol}</b> +1.8%\nSL фиксирует +0.5% прибыли"
                            )
                    if not pos.partial_tp_level_2_done and pos.quantity > 0:
                        partial_qty = min(pos.original_quantity * PARTIAL_TP_LEVEL_2_FRACTION, pos.quantity)
                        if partial_qty > 0:
                            try:
                                close_side = 'sell' if pos.side == 'long' else 'buy'
                                await self.api.create_order(
                                    symbol=symbol,
                                    side=close_side,
                                    amount=partial_qty,
                                    reduce_only=True
                                )
                                pos.partial_tp_level_2_done = True
                                pos.quantity = max(0.0, pos.quantity - partial_qty)
                                partial_pnl = pnl_pct * EFFECTIVE_EXPOSURE * PARTIAL_TP_LEVEL_2_FRACTION
                                self.daily_pnl += partial_pnl
                                logger.info(f"💰 {symbol}: PARTIAL TP 40% | +${partial_pnl:.2f}")
                                await self.telegram.send_message(
                                    f"💰 <b>{symbol}</b> PARTIAL TP #2\n"
                                    f"Закрыто 40% при +1.8%\n"
                                    f"Прибыль: +${partial_pnl:.2f}\n"
                                    f"Остаток с profit lock"
                                )
                            except Exception as e:
                                logger.error(f"Ошибка partial TP #2 {symbol}: {e}")
                
                # ========== УРОВЕНЬ 4: +2% - Trailing активен ==========
                if not pos.trailing_active and pnl_pct >= TRAILING_ACTIVATION_PCT:
                    pos.trailing_active = True
                    pos.max_profit = pnl_usd
                    logger.info(f"🎯 {symbol}: TRAILING активирован при +{pnl_pct*100:.1f}%")
                    await self.telegram.send_trailing_activated(symbol, pos.side, pnl_usd)
                
                # ========== УРОВЕНЬ 5: +3% - Жесткий trailing ==========
                if not pos.tight_trailing and pnl_pct >= TRAILING_TIGHT_LEVEL_PCT:
                    pos.tight_trailing = True
                    logger.info(f"🔥 {symbol}: TIGHT TRAILING при +{pnl_pct*100:.1f}% (шаг <1%)")
                    await self.telegram.send_message(
                        f"🔥 <b>{symbol}</b> +3%!\nTight trailing активен (0.5%)"
                    )
                
                # ========== ОБНОВЛЕНИЕ TRAILING STOP ==========
                if pos.trailing_active:
                    if pnl_usd > pos.max_profit:
                        pos.max_profit = pnl_usd
                        
                        # Выбираем дистанцию trailing
                        trail_dist = TRAILING_TIGHT_DISTANCE_PCT if pos.tight_trailing else TRAILING_DISTANCE_PCT
                        
                        if pos.side == 'long':
                            new_sl = current_price * (1 - trail_dist)
                            if new_sl > pos.sl_price:
                                sl_updated = await self.api.set_stop_loss(symbol, pos.side, new_sl)
                                if sl_updated:
                                    old_sl = pos.sl_price
                                    pos.sl_price = new_sl
                                    logger.info(f"📈 {symbol}: Trailing SL ${old_sl:.4f} → ${new_sl:.4f}")
                        else:
                            new_sl = current_price * (1 + trail_dist)
                            if new_sl < pos.sl_price:
                                sl_updated = await self.api.set_stop_loss(symbol, pos.side, new_sl)
                                if sl_updated:
                                    old_sl = pos.sl_price
                                    pos.sl_price = new_sl
                                    logger.info(f"📉 {symbol}: Trailing SL ${old_sl:.4f} → ${new_sl:.4f}")
                
                # ========== ПРОВЕРКА SL ==========
                if pos.side == 'long' and current_price <= pos.sl_price:
                    reason = "TRAILING" if pos.trailing_active else ("BREAKEVEN" if pos.breakeven_level_2_hit else "SL")
                    await self.close_position(symbol, current_price, reason)
                elif pos.side == 'short' and current_price >= pos.sl_price:
                    reason = "TRAILING" if pos.trailing_active else ("BREAKEVEN" if pos.breakeven_level_2_hit else "SL")
                    await self.close_position(symbol, current_price, reason)
                
            except Exception as e:
                logger.error(f"Ошибка обновления позиции {symbol}: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    bot = TradeGPTScalperLite()
    asyncio.run(bot.start())
