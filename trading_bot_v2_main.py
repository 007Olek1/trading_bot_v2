#!/usr/bin/env python3
"""
🤖 ТОРГОВЫЙ БОТ V2.0 - УЛЬТРА-БЕЗОПАСНАЯ ВЕРСИЯ
✅ Stop Loss ордера НА БИРЖЕ
✅ Жесткие лимиты рисков
✅ Аварийные стопы
✅ Тестовый режим (3 сделки по $1)
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import pytz
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from logging.handlers import RotatingFileHandler

# Наши модули
from bot_v2_config import Config
from bot_v2_safety import risk_manager, emergency_stop, position_guard
from bot_v2_exchange import exchange_manager
from bot_v2_signals import signal_analyzer
from bot_v2_ai_agent import trading_bot_agent, health_monitor
from bot_v2_auto_healing import auto_healing
from bot_v2_volatility_analyzer import enhanced_symbol_selector

# Создаем папку для логов если не существует
os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)

# Настройка логирования с РОТАЦИЕЙ
log_handler = RotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=10 * 1024 * 1024,  # 10 MB на файл
    backupCount=5,               # Хранить 5 файлов (50 MB всего)
    encoding='utf-8'
)
log_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))

logging.basicConfig(
    level=logging.INFO,  # INFO вместо DEBUG - меньше логов!
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[
        log_handler,
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingBotV2:
    """Торговый бот V2.0 - безопасность превыше всего"""
    
    def __init__(self):
        self.running = False
        self.paused = False
        self.open_positions: List[Dict[str, Any]] = []
        self.bot_errors_count = 0
        self.last_heartbeat = datetime.now()
        # Символы, по которым прямо сейчас идет открытие позиции (анти-дубликаты)
        self.pending_symbols = set()
        
        # Статистика сигналов
        self.signals_stats = {
            'total_analyzed': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'weak_signals': 0,
            'no_signals': 0
        }
        
        # Cooldown для предотвращения повторных входов
        # Формат: {symbol: datetime последней сделки}
        self.symbol_cooldown = {}
        self.cooldown_hours = 6  # Таймаут 6 часов между сделками по одной монете
        
        # КРИТИЧНО: Также запоминаем направление последней сделки
        # Формат: {symbol: side ("buy" или "sell")}
        self.symbol_last_side = {}
        
        logger.info("=" * 60)
        logger.info("🤖 ТОРГОВЫЙ БОТ V2.0 ИНИЦИАЛИЗИРОВАН")
        logger.info("=" * 60)
        
        # Проверка конфигурации
        errors = Config.validate_config()
        if errors:
            for error in errors:
                logger.error(error)
            raise ValueError("Ошибки конфигурации!")
        
        logger.info(f"💰 Тестовый режим: {Config.TEST_MODE}")
        if Config.TEST_MODE:
            logger.info(f"🧪 Размер позиции: ${Config.TEST_POSITION_SIZE_USD}")
            logger.info(f"🧪 Максимум сделок: {Config.TEST_MAX_TRADES}")
        else:
            logger.info(f"💰 Размер позиции: ${Config.POSITION_SIZE_USD}")
        logger.info(f"🛡️ Макс убыток/сделка: {Config.MAX_LOSS_PER_TRADE_PERCENT}%")
        logger.info(f"💵 Макс дневной убыток: ${Config.MAX_DAILY_LOSS_USD}")
    
    @staticmethod
    def format_price_change_pct(current_price: float, target_price: float, side: str) -> str:
        """
        Форматирует изменение цены в проценты с правильным знаком
        
        Args:
            current_price: Текущая цена (Entry)
            target_price: Целевая цена (SL или TP)
            side: "buy" или "sell"
        
        Returns:
            Строка вида "+5.0%" или "-3.0%"
        """
        price_change_pct = ((target_price - current_price) / current_price) * 100
        
        # Для LONG (BUY):
        # - SL ниже Entry → отрицательный процент → "-X%"
        # - TP выше Entry → положительный процент → "+X%"
        
        # Для SHORT (SELL):
        # - SL выше Entry → положительный процент → "+X%" (защита от роста!)
        # - TP ниже Entry → отрицательный процент → "-X%" (прибыль от падения!)
        
        sign = "+" if price_change_pct > 0 else ""
        return f"{sign}{price_change_pct:.1f}%"
    
    def is_symbol_on_cooldown(self, symbol: str) -> tuple[bool, float]:
        """
        Проверяет находится ли монета на cooldown
        
        Args:
            symbol: Символ монеты
        
        Returns:
            (is_cooldown, hours_remaining)
        """
        if symbol not in self.symbol_cooldown:
            return False, 0.0
        
        last_trade_time = self.symbol_cooldown[symbol]
        time_passed = datetime.now() - last_trade_time
        hours_passed = time_passed.total_seconds() / 3600
        
        if hours_passed >= self.cooldown_hours:
            # Cooldown истёк, удаляем из словаря
            del self.symbol_cooldown[symbol]
            return False, 0.0
        
        hours_remaining = self.cooldown_hours - hours_passed
        return True, hours_remaining
    
    def add_symbol_to_cooldown(self, symbol: str, side: str):
        """Добавляет монету в cooldown после открытия позиции"""
        self.symbol_cooldown[symbol] = datetime.now()
        self.symbol_last_side[symbol] = side.lower()
        logger.info(f"⏰ {symbol} {side.upper()} добавлена в cooldown на {self.cooldown_hours} часов")
    
    async def start(self):
        """Запуск бота"""
        try:
            logger.info("🚀 Запуск бота...")
            
            # 1. Подключение к бирже
            logger.info("🏦 Подключение к Bybit...")
            connected = await exchange_manager.connect()
            if not connected:
                raise Exception("Не удалось подключиться к Bybit!")
            
            # 2. Проверка баланса
            balance = await exchange_manager.get_balance()
            if balance is None or balance < 10:
                raise Exception(f"Недостаточный баланс: ${balance}")
            
            logger.info(f"💰 Баланс: ${balance:.2f} USDT")
            
            # 3. Синхронизация позиций с биржи
            await self.sync_positions_from_exchange()
            
            # 4. Запуск Telegram
            logger.info("📱 Запуск Telegram бота...")
            self.telegram_app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
            
            # Команды
            self.telegram_app.add_handler(CommandHandler("start", self.cmd_start))
            self.telegram_app.add_handler(CommandHandler("status", self.cmd_status))
            self.telegram_app.add_handler(CommandHandler("positions", self.cmd_positions))
            self.telegram_app.add_handler(CommandHandler("history", self.cmd_history))
            self.telegram_app.add_handler(CommandHandler("close_all", self.cmd_close_all))
            self.telegram_app.add_handler(CommandHandler("stop", self.cmd_stop))
            self.telegram_app.add_handler(CommandHandler("pause", self.cmd_pause))
            self.telegram_app.add_handler(CommandHandler("resume", self.cmd_resume))
            
            # Запуск Telegram в фоне (без polling - только уведомления)
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            
            # 5. Планировщик задач
            scheduler = AsyncIOScheduler()
            
            # Основной торговый цикл - СИНХРОНИЗАЦИЯ С ЗАКРЫТИЕМ 5-МИН СВЕЧЕЙ!
            # Запускаем каждые 5 минут в :00, :05, :10, :15 и т.д.
            scheduler.add_job(
                self.trading_loop,
                'cron',
                minute='*/5',  # Каждые 5 минут
                second=5,      # +5 сек после закрытия свечи
                timezone='UTC'
            )
            
            # Проверка здоровья
            scheduler.add_job(
                self.health_check,
                'interval',
                seconds=Config.HEALTH_CHECK_INTERVAL_SECONDS
            )
            
            # Heartbeat каждый час (СРАЗУ + каждый час)
            scheduler.add_job(
                self.send_heartbeat,
                'interval',
                hours=1,
                next_run_time=datetime.now() + timedelta(seconds=30)  # Первый через 30 сек
            )
            
            scheduler.start()
            
            self.running = True
            
            # Уведомление о запуске
            await self.send_telegram(
                f"🚀 БОТ V2.0 ЗАПУЩЕН!\n\n"
                f"💰 Баланс: ${balance:.2f}\n"
                f"💎 Режим: {'🧪 ТЕСТОВЫЙ' if Config.TEST_MODE else '✅ РЕАЛЬНАЯ ТОРГОВЛЯ'}\n"
                f"💵 Размер позиции: ${Config.get_position_size()}\n"
                f"🛡️ Макс убыток: -{Config.MAX_LOSS_PER_TRADE_PERCENT}%\n"
                f"📊 Макс позиций: {Config.MAX_POSITIONS}\n"
                f"⏰ Интервал анализа: {Config.TRADING_INTERVAL_SECONDS // 60} мин\n"
                f"🎯 TP: Адаптивные (на основе сигнала и волатильности)"
            )
            
            logger.info("✅ Бот запущен успешно!")
            logger.info("=" * 60)
            
            # Держим бота работающим
            while self.running:
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка запуска: {e}")
            await self.shutdown()
            raise
    
    async def trading_loop(self):
        """Основной торговый цикл"""
        try:
            if not self.running or self.paused:
                return
            
            logger.info("=" * 60)
            logger.info("🔄 ТОРГОВЫЙ ЦИКЛ")
            logger.info("=" * 60)
            
            # 1. Проверка расписания
            if not self.is_trading_allowed():
                logger.info("⏸️ Торговля не разрешена по расписанию")
                return
            
            # 2. Синхронизация позиций
            await self.sync_positions_from_exchange()
            
            # 3. Проверка аварийных условий
            should_stop, reason = await emergency_stop.check_emergency_conditions(
                risk_manager,
                self.open_positions,
                self.bot_errors_count
            )
            
            if should_stop:
                logger.critical(f"🚨 EMERGENCY STOP: {reason}")
                await self.emergency_shutdown(reason)
                return
            
            # 4. Получаем ВОЛАТИЛЬНЫЕ символы с анализом трендов - ВСЕГДА анализируем рынок!
            logger.info("🚀 Получение волатильных монет с анализом трендов...")
            
            # Проверяем кэш
            if enhanced_symbol_selector.is_cache_valid():
                symbols = enhanced_symbol_selector.get_cached_symbols()
                logger.info(f"📊 Использую кэшированные символы: {len(symbols)}")
            else:
                # Получаем свежий анализ волатильности
                volatile_symbols_data = await enhanced_symbol_selector.get_volatile_symbols(
                    exchange_manager, top_n=100
                )
                symbols = [data['symbol'] for data in volatile_symbols_data]
                
                if not symbols:
                    logger.warning("⚠️ Не удалось получить волатильные символы, используем базовые")
                    symbols = await exchange_manager.get_top_volume_symbols(top_n=50)
            
            if not symbols:
                logger.warning("⚠️ Не удалось получить символы")
                return
            
            logger.info(f"🔍 Анализ {len(symbols)} волатильных символов...")
            
            # Обновляем время анализа в Health Monitor
            health_monitor.record_successful_analysis()
            
            # 5. Проверка лимита позиций
            if len(self.open_positions) >= Config.MAX_POSITIONS:
                logger.info(f"📊 Лимит позиций ({Config.MAX_POSITIONS}/{Config.MAX_POSITIONS}) достигнут - мониторинг продолжается")
                logger.info("✅ Торговый цикл завершен (позиции заполнены)")
                logger.info("=" * 60)
                return
            
            # 6. Получаем баланс
            balance = await exchange_manager.get_balance()
            if balance is None:
                logger.error("❌ Не удалось получить баланс")
                return
            
            # 7. Проверка возможности открыть сделку
            can_trade, reason = risk_manager.can_open_trade(balance)
            if not can_trade:
                logger.warning(f"⚠️ Нельзя открыть сделку: {reason}")
                
                # Специальное уведомление при достижении тестового лимита
                if Config.TEST_MODE and "Лимит тестовых сделок" in reason:
                    await self.send_telegram(
                        f"🧪 ТЕСТОВЫЙ РЕЖИМ ЗАВЕРШЕН!\n\n"
                        f"✅ Выполнено {Config.TEST_MAX_TRADES} тестовых сделок\n"
                        f"📊 Результаты:\n"
                        f"   • Дневной P&L: ${-risk_manager.daily_loss:.2f}\n"
                        f"   • Серия убытков: {risk_manager.consecutive_losses}\n\n"
                        f"⏸️ Бот остановлен для анализа результатов\n"
                        f"📝 Проверьте логи и примите решение о дальнейших действиях"
                    )
                    await self.pause_bot()
                return
            
            # 8. Анализируем символы
            for symbol in symbols:
                if len(self.open_positions) >= Config.MAX_POSITIONS:
                    break
                
                # Пропускаем, если по символу уже идет процесс открытия (защита от дубликатов)
                if symbol in self.pending_symbols:
                    logger.debug(f"⏳ Пропуск {symbol}: открытие уже в процессе")
                    continue

                # Пропускаем если уже есть позиция
                if any(p['symbol'] == symbol for p in self.open_positions):
                    continue
                
                # ПРОВЕРКА COOLDOWN - предотвращаем повторные входы
                is_cooldown, hours_remaining = self.is_symbol_on_cooldown(symbol)
                if is_cooldown:
                    logger.debug(f"⏰ {symbol} на cooldown (осталось {hours_remaining:.1f}ч)")
                    continue
                
                # Анализ
                signal_result = await self.analyze_symbol(symbol)
                
                if signal_result and signal_result.get('signal'):
                    # Пытаемся открыть сделку
                    self.pending_symbols.add(symbol)
                    try:
                        position = await self.open_position(
                            symbol=symbol,
                            side=signal_result['signal'],
                            signal_data=signal_result
                        )
                    finally:
                        self.pending_symbols.discard(symbol)
                    
                    if position:
                        logger.info(f"✅ Позиция открыта: {symbol}")
                        break  # Открыли одну - хватит
            
            logger.info("✅ Торговый цикл завершен")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Ошибка в торговом цикле: {e}")
            self.bot_errors_count += 1
    
    def calculate_adaptive_tp_levels(
        self, 
        signal_data: Dict[str, Any],
        current_price: float
    ) -> List[float]:
        """
        Расчет АДАПТИВНЫХ уровней Take Profit на основе:
        - Уверенности сигнала
        - Волатильности (ATR)
        - Силы тренда
        
        Returns:
            List[float]: 5 процентных уровней для TP
        """
        confidence = signal_data.get('confidence', 85)
        indicators = signal_data.get('indicators', {})
        
        # Базовые уровни (консервативные)
        base_levels = [0.006, 0.013, 0.019, 0.026, 0.032]  # 0.6% - 3.2%
        
        # Множитель на основе уверенности
        if confidence >= 95:
            confidence_mult = 1.5  # Очень сильный сигнал - агрессивные цели
        elif confidence >= 90:
            confidence_mult = 1.3  # Сильный сигнал
        elif confidence >= 85:
            confidence_mult = 1.1  # Умеренный сигнал
        else:
            confidence_mult = 1.0  # Слабый (но прошел фильтр)
        
        # Множитель на основе волатильности (ATR)
        atr = indicators.get('atr', 0)
        if atr > 0:
            # Высокая волатильность - увеличиваем цели
            atr_percent = (atr / current_price) * 100
            if atr_percent > 3:  # Высокая волатильность
                volatility_mult = 1.4
            elif atr_percent > 2:  # Средняя
                volatility_mult = 1.2
            else:  # Низкая
                volatility_mult = 1.0
        else:
            volatility_mult = 1.0
        
        # Итоговый множитель
        total_mult = confidence_mult * volatility_mult
        
        # Ограничиваем множитель (макс 2x от базовых уровней)
        total_mult = min(2.0, total_mult)
        
        # Применяем множитель к базовым уровням
        adaptive_levels = [level * total_mult for level in base_levels]
        
        # Ограничиваем максимальные цели (не более 6%)
        adaptive_levels = [min(0.06, level) for level in adaptive_levels]
        
        logger.info(
            f"🎯 Адаптивные TP: уверенность={confidence}%, "
            f"ATR={(atr/current_price*100):.2f}%, "
            f"множитель={total_mult:.2f}x"
        )
        
        return adaptive_levels
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Анализ символа"""
        try:
            # Получаем свечи
            ohlcv = await exchange_manager.fetch_ohlcv(symbol, timeframe="5m", limit=100)
            if not ohlcv:
                return None
            
            # Конвертируем в DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Анализ сигнала
            signal_result = signal_analyzer.analyze(df)
            
            # Записываем успешный анализ
            health_monitor.record_successful_analysis()
            
            # Статистика
            self.signals_stats['total_analyzed'] += 1
            
            # 🔍 DEBUG: Логируем ВСЕ сигналы (даже слабые)
            if signal_result.get('signal'):
                if signal_result['signal'] == 'buy':
                    self.signals_stats['buy_signals'] += 1
                else:
                    self.signals_stats['sell_signals'] += 1
                    
                logger.info(
                    f"📊 {symbol}: {signal_result['signal'].upper()} "
                    f"({signal_result['confidence']:.0f}%) - {signal_result['reason']}"
                )
            elif signal_result.get('confidence', 0) > 0:
                self.signals_stats['weak_signals'] += 1
                # Логируем слабые сигналы (не прошедшие фильтр)
                logger.info(
                    f"🔍 {symbol}: Слабый сигнал - {signal_result.get('reason', 'нет данных')} "
                    f"(уверенность: {signal_result.get('confidence', 0):.0f}%)"
                )
            else:
                self.signals_stats['no_signals'] += 1
            
            return signal_result if signal_result['signal'] else None
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа {symbol}: {e}")
            health_monitor.record_error("analysis", str(e))
            return None
    
    async def open_position(
        self,
        symbol: str,
        side: str,
        signal_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        ОТКРЫТИЕ ПОЗИЦИИ С ЗАЩИТОЙ
        Критически важно: SL ордер ОБЯЗАТЕЛЕН!
        """
        try:
            logger.info(f"🚀 Открытие позиции: {symbol} {side.upper()}")
            
            # 0. ПРОВЕРКА AI АГЕНТА
            balance = await exchange_manager.get_balance()
            agent_allow, agent_reason = trading_bot_agent.should_allow_new_trade(
                signal_confidence=signal_data.get('confidence', 0) / 100,
                balance=balance
            )
            
            if not agent_allow:
                logger.warning(f"🤖 Агент заблокировал сделку: {agent_reason}")
                health_monitor.record_error("agent_block", agent_reason)
                return None
            
            logger.info(f"🤖 Агент РАЗРЕШИЛ сделку: {agent_reason}")
            
            # 1. Получаем текущую цену
            ohlcv = await exchange_manager.fetch_ohlcv(symbol, limit=1)
            if not ohlcv:
                logger.error("❌ Не удалось получить цену")
                return None
            
            current_price = float(ohlcv[-1][4])  # close price

            # 1.1 Доп. защита: перед входом убеждаемся, что позиции по символу НЕТ на бирже
            try:
                live_positions = await exchange_manager.fetch_positions()
                if any(p.get('symbol') == symbol and float(p.get('contracts', 0) or 0) > 0 for p in live_positions):
                    logger.warning(f"🛑 Пропуск открытия {symbol}: позиция уже существует на бирже")
                    return None
            except Exception as _:
                # Не блокируем из-за временной ошибки проверки, просто продолжаем
                pass
            
            # 2. Устанавливаем леверидж
            await exchange_manager.set_leverage(symbol, Config.LEVERAGE)
            
            # 3. Рассчитываем размер позиции
            balance = await exchange_manager.get_balance()
            position_size_usd = risk_manager.calculate_position_size(balance)
            amount = (position_size_usd * Config.LEVERAGE) / current_price
            
            # 4. Рассчитываем SL и TP
            stop_loss, take_profit = risk_manager.calculate_sl_tp_prices(current_price, side)
            sl_pct = self.format_price_change_pct(current_price, stop_loss, side)
            tp_pct = self.format_price_change_pct(current_price, take_profit, side)
            logger.info(f"🎯 SL/TP: вход=${current_price:.4f}, SL=${stop_loss:.4f} ({sl_pct}), TP=${take_profit:.4f} ({tp_pct})")
            
            # 5. Открываем позицию на бирже
            logger.info(f"💰 Создаю market ордер: {amount:.6f} @ ${current_price:.4f}")
            market_order = await exchange_manager.create_market_order(symbol, side, amount)
            
            if not market_order:
                logger.error("❌ Не удалось создать market ордер")
                return None
            
            # КРИТИЧНО: Получаем ФАКТИЧЕСКИЙ размер позиции с биржи (может быть округлен!)
            import asyncio
            await asyncio.sleep(0.5)  # Даем бирже время обработать ордер
            positions = await exchange_manager.fetch_positions()
            actual_amount = amount  # По умолчанию используем расчетный
            
            for pos in positions:
                if pos['symbol'] == symbol and float(pos.get('contracts', 0)) > 0:
                    actual_amount = float(pos.get('contracts', 0))
                    logger.info(f"📊 Фактический размер позиции: {actual_amount} (расчетный был {amount:.6f})")
                    break
            
            # 6. КРИТИЧНО: Создаем Stop Loss ордер НА БИРЖЕ
            logger.info("🛡️ Создаю Stop Loss ордер на бирже...")
            close_side = "sell" if side == "buy" else "buy"
            
            sl_order = await exchange_manager.create_stop_market_order(
                symbol=symbol,
                side=close_side,
                amount=actual_amount,  # Используем фактический размер!
                stop_price=stop_loss
            )
            
            # ПРОВЕРКА: SL ордер создан?
            if not sl_order or not sl_order.get('id'):
                logger.critical(f"🚨 КРИТИЧНО: SL ордер НЕ СОЗДАН для {symbol}!")
                
                # ЗАКРЫВАЕМ ПОЗИЦИЮ НЕМЕДЛЕННО!
                logger.warning("⚠️ Закрываю позицию без SL...")
                await exchange_manager.create_market_order(symbol, close_side, amount)
                
                await self.send_telegram(
                    f"🚨 КРИТИЧЕСКАЯ ОШИБКА!\n\n"
                    f"Не удалось создать SL ордер для {symbol}\n"
                    f"Позиция немедленно закрыта!\n"
                    f"Проверьте настройки биржи!"
                )
                
                return None
            
            logger.info(f"✅ SL ордер создан: {sl_order['id']}")
            
            # 7. Создаем МНОГОУРОВНЕВЫЙ Take Profit (с учетом минимального размера биржи)
            logger.info("🎯 Создаю многоуровневый Take Profit...")
            
            # Получаем минимальный размер ордера для этой монеты
            try:
                market = await exchange_manager.exchange.load_markets()
                market_info = market.get(symbol, {})
                min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0.01)
                logger.debug(f"📏 Минимальный размер для {symbol}: {min_amount}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось получить минимальный размер, использую 0.01: {e}")
                min_amount = 0.01
            
            # Адаптивные уровни TP на основе сигнала и волатильности
            tp_levels = self.calculate_adaptive_tp_levels(signal_data, current_price)
            
            # АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ КОЛИЧЕСТВА TP УРОВНЕЙ
            # Определяем сколько уровней можем создать исходя из минимального размера
            max_tp_levels = 5  # Желаемое количество
            
            # Пробуем создать максимум уровней, но если размер < минимума, уменьшаем количество
            num_tp_levels = max_tp_levels
            for num_levels in [5, 3, 2, 1]:
                tp_amount_per_level = actual_amount / num_levels
                if tp_amount_per_level >= min_amount:
                    num_tp_levels = num_levels
                    break
            else:
                # Если даже вся позиция < минимума
                num_tp_levels = 1
                tp_amount_per_level = actual_amount
            
            logger.info(
                f"🎯 TP стратегия: {num_tp_levels} уровней по {tp_amount_per_level:.6f} каждый "
                f"(мин={min_amount:.6f})"
            )
            logger.info(
                f"🎯 TP уровни: {[f'{l*100:.1f}%' for l in tp_levels]} "
                f"(ROI при 5X: {[f'{l*100*Config.LEVERAGE:.1f}%' for l in tp_levels]})"
            )
            
            # Если можем создать только 1 TP - берём средний уровень
            if num_tp_levels == 1:
                logger.warning(
                    f"⚠️ Размер уровня {tp_amount_per_level:.6f} < минимум {min_amount:.6f}. "
                    f"Создаю 1 TP ордер на всю позицию"
                )
                # Берем средний уровень (1.9% = 9.5% ROI при 5X)
                tp_level = 0.019
                if side == "buy":
                    tp_price = current_price * (1 + tp_level)
                else:
                    tp_price = current_price * (1 - tp_level)
                
                try:
                    tp_order = await exchange_manager.create_limit_order(
                        symbol=symbol,
                        side=close_side,
                        amount=actual_amount,  # ВСЯ ФАКТИЧЕСКАЯ позиция
                        price=round(tp_price, 4)
                    )
                    
                    if tp_order and tp_order.get('id'):
                        tp_orders = [tp_order]
                        logger.info(f"✅ TP создан @ ${tp_price:.4f} (+{tp_level*100:.1f}% = +{tp_level*100*Config.LEVERAGE:.1f}% ROI)")
                    else:
                        logger.warning(f"⚠️ Не удалось создать TP")
                        tp_orders = []
                except Exception as e:
                    logger.error(f"❌ Ошибка создания TP: {e}")
                    tp_orders = []
            else:
                # Создаем N уровней TP (автоматически определено выше)
                tp_orders = []
                
                # Выбираем нужное количество уровней из массива tp_levels
                if num_tp_levels == 5:
                    selected_levels = tp_levels  # Все 5 уровней
                elif num_tp_levels == 3:
                    selected_levels = [tp_levels[0], tp_levels[2], tp_levels[4]]  # 1, 3, 5
                elif num_tp_levels == 2:
                    selected_levels = [tp_levels[1], tp_levels[4]]  # 2, 5
                else:
                    selected_levels = [tp_levels[2]]  # Средний уровень
                
                for i, level in enumerate(selected_levels, 1):
                    if side == "buy":
                        tp_price = current_price * (1 + level)
                    else:
                        tp_price = current_price * (1 - level)
                    
                    try:
                        tp_order = await exchange_manager.create_limit_order(
                            symbol=symbol,
                            side=close_side,
                            amount=tp_amount_per_level,
                            price=round(tp_price, 4)
                        )
                        
                        if tp_order and tp_order.get('id'):
                            tp_orders.append(tp_order)
                            logger.info(f"✅ TP{i} создан @ ${tp_price:.4f} ({self.format_price_change_pct(current_price, tp_price, side)})")
                        else:
                            logger.warning(f"⚠️ Не удалось создать TP{i}")
                    except Exception as e:
                        logger.error(f"❌ Ошибка создания TP{i}: {e}")
            
            # 8. Создаем запись о позиции
            position = {
                "symbol": symbol,
                "side": side,
                "entry_price": current_price,
                "amount": actual_amount,  # ФАКТИЧЕСКИЙ размер с биржи!
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "sl_order_id": sl_order['id'],
                "tp_order_id": tp_orders[0]['id'] if tp_orders else None,  # Первый TP
                "market_order_id": market_order['id'],
                "open_time": datetime.now(),
                "signal_confidence": signal_data['confidence'],
                "signal_reason": signal_data['reason']
            }
            
            self.open_positions.append(position)
            
            # 9. Уведомление с многоуровневым TP
            invested = position_size_usd  # Сколько вложено (без leverage)
            
            # Формируем текст с TP уровнями с ПРАВИЛЬНЫМИ знаками
            # Используем те же уровни что и в ордерах
            if num_tp_levels == 5:
                display_levels = tp_levels
            elif num_tp_levels == 3:
                display_levels = [tp_levels[0], tp_levels[2], tp_levels[4]]
            elif num_tp_levels == 2:
                display_levels = [tp_levels[1], tp_levels[4]]
            else:
                display_levels = [tp_levels[2]]
            
            targets_text = ""
            emojis = ["🥇", "🥈", "🥉", "💎", "🚀"]
            for i, level in enumerate(display_levels, 1):
                if side == "buy":
                    tp_price = current_price * (1 + level)
                else:
                    tp_price = current_price * (1 - level)
                
                # Правильный процент с знаком
                tp_pct_str = self.format_price_change_pct(current_price, tp_price, side)
                tp_pct = level * 100
                profit_usd = invested * (tp_pct / 100) * Config.LEVERAGE
                
                emoji = emojis[i-1] if i <= len(emojis) else "🎯"
                targets_text += f"   {emoji} ${tp_price:.4f} ({tp_pct_str} = +${profit_usd:.2f})\n"
            
            # SL с правильным знаком и убыток в $
            sl_pct_str = self.format_price_change_pct(current_price, stop_loss, side)
            sl_pct = abs((stop_loss - current_price) / current_price * 100)
            loss_usd = invested * (sl_pct / 100) * Config.LEVERAGE
            
            await self.send_telegram(
                f"🟢 ПОЗИЦИЯ ОТКРЫТА\n\n"
                f"💎 {symbol} | {side.upper()} | {Config.LEVERAGE}X\n"
                f"💰 Entry: ${current_price:.4f}\n"
                f"💵 Инвестировано: ${invested:.2f}\n\n"
                f"🎯 Targets:\n{targets_text}\n"
                f"🛡️ Stop Loss: ${stop_loss:.4f} ({sl_pct_str} = -${loss_usd:.2f})\n\n"
                f"🎲 Уверенность: {signal_data['confidence']:.0f}%\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
            
            logger.info(f"✅ Позиция {symbol} успешно открыта с защитой!")
            
            # Добавляем монету в cooldown с указанием направления
            self.add_symbol_to_cooldown(symbol, side)
            
            return position
            
        except Exception as e:
            logger.error(f"❌ Ошибка открытия позиции: {e}")
            self.bot_errors_count += 1
            return None
    
    async def sync_positions_from_exchange(self):
        """Синхронизация позиций с биржи"""
        try:
            exchange_positions = await exchange_manager.fetch_positions()
            
            # Обновляем наш список
            self.open_positions = []
            
            for ex_pos in exchange_positions:
                symbol = ex_pos['symbol']
                size = float(ex_pos.get('contracts', 0))
                
                if size > 0:
                    # КРИТИЧНО: Проверяем SL НА ПОЗИЦИИ, а не в ордерах!
                    # Bybit использует trading-stop API, который устанавливает SL на позицию
                    stop_loss_price = ex_pos.get('stopLoss') or ex_pos.get('info', {}).get('stopLoss')
                    
                    # Создаем фиктивный sl_order_id если SL установлен
                    if stop_loss_price and stop_loss_price != "" and stop_loss_price != "0":
                        try:
                            sl_value = float(stop_loss_price)
                            if sl_value > 0:
                                sl_order_id = f"SL_{symbol}_{int(sl_value * 10000)}"
                                logger.debug(f"✅ Найден SL на позиции: ${sl_value:.4f}")
                            else:
                                sl_order_id = None
                        except (ValueError, TypeError):
                            sl_order_id = None
                    else:
                        sl_order_id = None
                    
                    # Получаем TP ордера
                    orders = await exchange_manager.fetch_open_orders(symbol)
                    tp_order_id = None
                    
                    for order in orders:
                        # TP ордер: limit + reduceOnly
                        if order.get('type') == 'limit' and order.get('reduceOnly'):
                            if not tp_order_id:  # Берем только первый TP
                                tp_order_id = order['id']
                                logger.debug(f"✅ Найден TP ордер: {tp_order_id}")
                    
                    position = {
                        "symbol": symbol,
                        "side": ex_pos.get('side'),
                        "entry_price": float(ex_pos.get('entryPrice', 0)),
                        "amount": size,
                        "sl_order_id": sl_order_id,  # Фиктивный ID на основе stopLoss позиции
                        "tp_order_id": tp_order_id,
                        "open_time": datetime.now(),
                        "signal_confidence": 0,
                        "signal_reason": "Синхронизировано с биржи"
                    }
                    
                    self.open_positions.append(position)
            
            if self.open_positions:
                logger.info(f"📊 Синхронизировано {len(self.open_positions)} позиций")
            
        except Exception as e:
            logger.error(f"❌ Ошибка синхронизации позиций: {e}")
    
    async def health_check(self):
        """Проверка здоровья бота с Auto-Healing"""
        try:
            # 1. ПРОВЕРКА ЗДОРОВЬЯ БОТА (передаем количество позиций)
            is_healthy, health_status = health_monitor.is_healthy(
                open_positions_count=len(self.open_positions),
                max_positions=Config.MAX_POSITIONS
            )
            
            if not is_healthy:
                logger.warning(f"🏥 Health Monitor: {health_status}")
                
                # Попытка самоисправления
                healed, healing_action = await auto_healing.diagnose_and_heal(
                    exchange_manager,
                    health_monitor,
                    self.open_positions
                )
                
                if healed:
                    logger.info(f"🔧 Auto-Healing: {healing_action}")
                    await self.send_telegram(
                        f"🔧 *AUTO-HEALING*\n\n"
                        f"Проблема: {health_status}\n"
                        f"Действие: {healing_action}\n"
                        f"✅ Проблема исправлена автоматически!"
                    )
                else:
                    logger.error(f"❌ Auto-Healing не смог исправить: {healing_action}")
            
            # 2. Проверка подключения к бирже
            if not exchange_manager.connected:
                logger.error("❌ Потеряно подключение к бирже!")
                health_monitor.record_error("exchange_connection", "Потеряно подключение")
                
                # Auto-Healing попытается восстановить
                healed, _ = await auto_healing.heal_exchange_connection(exchange_manager)
                if not healed:
                    await exchange_manager.connect()
            
            # 3. TRAILING STOP LOSS - перемещаем SL за прибылью
            await self.update_trailing_stop_loss()
            
        except Exception as e:
            logger.error(f"❌ Ошибка health check: {e}")
            health_monitor.record_error("health_check", str(e))
    
    async def update_trailing_stop_loss(self):
        """Обновление Trailing Stop Loss для всех позиций"""
        try:
            if not self.open_positions:
                return
            
            # Получаем актуальные позиции с биржи
            live_positions = await exchange_manager.fetch_positions()
            
            for pos in live_positions:
                symbol = pos['symbol']
                side = pos['side']
                entry_price = pos['entryPrice']
                current_price = pos.get('markPrice', entry_price)
                
                # Получаем info нашей позиции
                our_pos = next((p for p in self.open_positions if p['symbol'] == symbol), None)
                if not our_pos:
                    continue
                
                # Рассчитываем прибыль в процентах
                if side.lower() in ['buy', 'long']:
                    profit_pct = ((current_price - entry_price) / entry_price) * 100 * Config.LEVERAGE
                else:
                    profit_pct = ((entry_price - current_price) / entry_price) * 100 * Config.LEVERAGE
                
                # Получаем текущий SL
                info = pos.get('info', {})
                current_sl = info.get('stopLoss')
                
                if not current_sl:
                    continue
                
                current_sl = float(current_sl)
                new_sl = None
                
                # ЛОГИКА TRAILING:
                # Для BUY: SL двигается ВВЕРХ при росте цены (вверх к текущей цене)
                # Для SELL: SL двигается ВНИЗ при падении цены (вниз к текущей цене), НО НИКОГДА не ниже Entry!
                
                if profit_pct >= 10:
                    if side.lower() in ['buy', 'long']:
                        new_sl = entry_price * 1.05  # +5% от Entry
                        trailing_level = "+5%"
                    else:  # SELL
                        # Для SELL при большой прибыли: SL = Entry + 2% (защита прибыли)
                        new_sl = entry_price * 1.02  # +2% от Entry
                        trailing_level = "+2%"
                    
                elif profit_pct >= 5:
                    if side.lower() in ['buy', 'long']:
                        new_sl = entry_price * 1.02   # +2% от Entry
                        trailing_level = "+2%"
                    else:  # SELL
                        # Для SELL: SL = Entry + 1%
                        new_sl = entry_price * 1.01   # +1% от Entry
                        trailing_level = "+1%"
                    
                elif profit_pct >= 2:
                    if side.lower() in ['buy', 'long']:
                        new_sl = entry_price  # Безубыток
                        trailing_level = "безубыток"
                    else:  # SELL
                        # Для SELL: SL = Entry + 0.2% (микро-защита, чтобы не закрыть в безубыток сразу)
                        new_sl = entry_price * 1.002  # +0.2% от Entry
                        trailing_level = "+0.2%"
                
                # Обновляем SL только если новый лучше текущего
                if new_sl:
                    # Для LONG: новый SL должен быть выше текущего
                    # Для SHORT: новый SL должен быть НИЖЕ текущего (но СТРОГО выше Entry!)
                    should_update = False
                    
                    if side.lower() in ['buy', 'long']:
                        should_update = new_sl > current_sl
                    else:  # SELL
                        # КРИТИЧНО: SL для SELL НИКОГДА не должен быть <= Entry!
                        min_sl_for_sell = entry_price * 1.001  # Минимум Entry + 0.1%
                        if new_sl < min_sl_for_sell:
                            new_sl = min_sl_for_sell
                            trailing_level = "+0.1% (мин)"
                        
                        should_update = new_sl < current_sl and new_sl > entry_price
                    
                    if should_update:
                        # Обновляем SL на бирже
                        close_side = "sell" if side.lower() in ['buy', 'long'] else "buy"
                        
                        sl_order = await exchange_manager.create_stop_market_order(
                            symbol=symbol,
                            side=close_side,
                            amount=pos['contracts'],
                            stop_price=new_sl
                        )
                        
                        if sl_order:
                            logger.info(f"🎯 Trailing SL обновлен для {symbol}: {current_sl:.4f} → {new_sl:.4f} ({trailing_level})")
                            
                            await self.send_telegram(
                                f"🎯 *TRAILING STOP*\n\n"
                                f"💎 {symbol}\n"
                                f"📈 Прибыль: +{profit_pct:.1f}%\n"
                                f"🛡️ Новый SL: ${new_sl:.4f} ({trailing_level})\n"
                                f"✅ Прибыль защищена!"
                            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка Trailing SL: {e}")
    
    async def send_heartbeat(self):
        """Отправка heartbeat сообщения каждый час"""
        try:
            balance = await exchange_manager.get_balance()
            warsaw_time = datetime.now(pytz.timezone('Europe/Warsaw'))
            
            # Статус
            status_emoji = "🟢" if self.running and not self.paused else "⏸️" if self.paused else "🔴"
            test_mode_text = "🧪 ТЕСТОВЫЙ РЕЖИМ" if Config.TEST_MODE else "💰 РАБОЧИЙ РЕЖИМ"
            
            # ДЕТАЛЬНАЯ информация о позициях с биржи
            positions_text = ""
            total_pnl = 0
            
            if self.open_positions:
                # Получаем актуальные позиции с биржи
                live_positions = await exchange_manager.fetch_positions()
                
                positions_text = f"\n💰 Баланс: ${balance:.2f} USDT\n"
                positions_text += f"📊 Открытых позиций: {len(live_positions)}\n"
                
                for pos in live_positions:
                    symbol = pos['symbol']
                    side = pos['side']
                    size = pos['contracts']
                    entry = pos['entryPrice']
                    pnl = pos.get('unrealizedPnl', 0)
                    leverage = pos.get('leverage', 5)
                    
                    # Получаем SL
                    info = pos.get('info', {})
                    stop_loss = info.get('stopLoss', 'НЕТ')
                    
                    # Эмодзи в зависимости от PnL
                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    
                    total_pnl += pnl
                    
                    positions_text += f"\n{pnl_emoji} {symbol} | {side.upper()} | {leverage}X\n"
                    positions_text += f"   Entry: {entry:.4f}\n"
                    positions_text += f"   Size: {size:.4f}\n"
                    positions_text += f"   PnL: ${pnl:.4f}\n"
                    positions_text += f"   SL: {stop_loss}\n"
                
                positions_text += f"\n💵 TOTAL PnL: ${total_pnl:.4f}\n"
            else:
                positions_text = "\n   ✅ Нет открытых позиций"
            
            # Получаем отчеты агентов
            agent_report = trading_bot_agent.get_performance_report()
            health_report = health_monitor.get_health_report(
                open_positions_count=len(self.open_positions),
                max_positions=Config.MAX_POSITIONS
            )
            
            # Статус здоровья
            health_emoji = "✅" if health_report['is_healthy'] else "⚠️"
            
            await self.send_telegram(
                f"💓 *HEARTBEAT - БОТ V2.0*\n\n"
                f"{status_emoji} *Статус:* {'Работает' if self.running and not self.paused else 'Пауза' if self.paused else 'Остановлен'}\n"
                f"⏰ *Время:* {warsaw_time.strftime('%H:%M:%S')} (Варшава)\n"
                f"{positions_text}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{test_mode_text}\n"
                f"📈 *P&L сегодня:* ${-risk_manager.daily_loss:.2f}\n"
                f"🔢 *Сделок:* {risk_manager.trades_today}/{Config.MAX_TRADES_PER_DAY}\n"
                f"🔄 *Серия убытков:* {risk_manager.consecutive_losses}\n\n"
                f"🤖 *AI АГЕНТ:*\n"
                f"   Win Rate: {agent_report['win_rate']:.0%}\n"
                f"   Profit Factor: {agent_report['profit_factor']:.2f}\n"
                f"   Всего сделок: {agent_report['total_trades']}\n\n"
                f"{health_emoji} *ЗДОРОВЬЕ:*\n"
                f"   Статус: {health_report['health_status']}\n"
                f"   Ошибок: {health_report['total_errors']}\n"
                f"   Healing попыток: {auto_healing.healing_attempts}"
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка heartbeat: {e}")
    
    def is_trading_allowed(self) -> bool:
        """Проверка расписания торговли"""
        now = datetime.now(pytz.timezone(Config.TIMEZONE))
        weekday = now.weekday()
        hour = now.hour
        
        # Суббота
        if weekday == 5 and Config.WEEKEND_REST:
            return False
        
        # Воскресенье
        if weekday == 6:
            if Config.SUNDAY_EVENING_TRADING:
                if Config.SUNDAY_TRADING_START_HOUR <= hour < Config.SUNDAY_TRADING_END_HOUR:
                    return True
            return False
        
        return True
    
    async def save_trade_to_history(self, trade_data: Dict[str, Any]):
        """Сохранение сделки в историю"""
        try:
            import json
            history_file = "trade_history.json"
            
            # Читаем существующую историю
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                history = []
            
            # Добавляем новую сделку
            history.append(trade_data)
            
            # Сохраняем
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"💾 Сделка сохранена в историю: {trade_data['symbol']} (P&L: ${trade_data['pnl']:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения истории: {e}")
    
    async def send_telegram(self, message: str):
        """Отправка сообщения в Telegram"""
        try:
            if self.telegram_app and Config.TELEGRAM_ALERTS_ENABLED:
                await self.telegram_app.bot.send_message(
                    chat_id=Config.TELEGRAM_CHAT_ID,
                    text=message
                )
        except Exception as e:
            logger.error(f"❌ Ошибка отправки Telegram: {e}")
    
    async def emergency_shutdown(self, reason: str):
        """Аварийное отключение"""
        logger.critical(f"🚨🚨🚨 EMERGENCY SHUTDOWN: {reason}")
        
        try:
            # Закрыть все позиции
            for position in self.open_positions:
                # ПРАВИЛЬНАЯ ЛОГИКА: LONG/Buy закрывается SELL, SHORT/Sell закрывается BUY
                side_lower = position['side'].lower()
                if side_lower in ["buy", "long"]:
                    close_side = "sell"
                elif side_lower in ["sell", "short"]:
                    close_side = "buy"
                else:
                    logger.error(f"❌ Неизвестный side: {position['side']}")
                    continue
                    
                logger.critical(f"🚨 Закрываю {position['symbol']}: side={position['side']} -> close_side={close_side}")
                await exchange_manager.create_market_order(
                    position['symbol'],
                    close_side,
                    position['amount']
                )
            
            # Отменить все ордера
            await exchange_manager.cancel_all_orders()
            
            self.running = False
            emergency_stop.activate(reason)
            
            await self.send_telegram(
                f"🚨🚨🚨 EMERGENCY STOP!\n\n"
                f"Причина: {reason}\n\n"
                f"✅ Все позиции закрыты\n"
                f"✅ Все ордера отменены\n"
                f"✅ Бот остановлен"
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка emergency shutdown: {e}")
    
    async def close_position(self, position: Dict[str, Any], reason: str = "Manual") -> bool:
        """
        Закрытие позиции с записью в TradingBotAgent
        
        Args:
            position: Позиция для закрытия
            reason: Причина закрытия (TP/SL/Manual)
        
        Returns:
            bool: True если успешно закрыто
        """
        try:
            symbol = position['symbol']
            side = position['side']
            amount = position['amount']
            entry_price = position['entry_price']
            
            # 1. Отменяем SL/TP ордера
            if position.get('sl_order_id'):
                try:
                    await exchange_manager.cancel_order(symbol, position['sl_order_id'])
                    logger.info(f"✅ SL ордер {position['sl_order_id']} отменен")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось отменить SL: {e}")
            
            if position.get('tp_order_id'):
                try:
                    await exchange_manager.cancel_order(symbol, position['tp_order_id'])
                    logger.info(f"✅ TP ордер {position['tp_order_id']} отменен")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось отменить TP: {e}")
            
            # 2. Закрываем позицию по рынку
            side_lower = side.lower()
            if side_lower in ["buy", "long"]:
                close_side = "sell"
            elif side_lower in ["sell", "short"]:
                close_side = "buy"
            else:
                logger.error(f"❌ Неизвестный side: {side}")
                return False
                
            logger.info(f"📤 Закрываю позицию: side={side} -> close_side={close_side}")
            close_order = await exchange_manager.create_market_order(
                symbol,
                close_side,
                amount
            )
            
            if not close_order:
                logger.error(f"❌ Не удалось создать ордер закрытия для {symbol}")
                return False
            
            # 3. Получаем цену закрытия
            exit_price = float(close_order.get('price', entry_price))
            
            # 4. Рассчитываем прибыль
            if side == "buy":
                pnl = (exit_price - entry_price) * amount * Config.LEVERAGE
                pnl_pct = ((exit_price - entry_price) / entry_price * 100) * Config.LEVERAGE
            else:
                pnl = (entry_price - exit_price) * amount * Config.LEVERAGE
                pnl_pct = ((entry_price - exit_price) / entry_price * 100) * Config.LEVERAGE
            
            # 5. ЗАПИСЫВАЕМ РЕЗУЛЬТАТ В AI АГЕНТА
            trading_bot_agent.record_trade(
                profit=pnl,
                win=(pnl > 0),
                confidence=position.get('signal_confidence', 0)
            )
            
            # 6. Обновляем risk_manager
            if pnl < 0:
                risk_manager.record_loss(abs(pnl))
            else:
                risk_manager.record_win(pnl)
            
            # 7. СОХРАНЯЕМ В ИСТОРИЮ СДЕЛОК
            await self.save_trade_to_history({
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_percent': pnl_pct,
                'reason': reason,
                'confidence': position.get('signal_confidence', 0),
                'leverage': Config.LEVERAGE,
                'open_time': position.get('open_time', datetime.now().isoformat()),
                'close_time': datetime.now().isoformat()
            })
            
            # 8. Удаляем из списка открытых
            self.open_positions = [p for p in self.open_positions if p['symbol'] != symbol]
            
            # 9. Уведомление
            emoji = "🟢" if pnl > 0 else "🔴"
            await self.send_telegram(
                f"{emoji} ПОЗИЦИЯ ЗАКРЫТА\n\n"
                f"💎 {symbol}\n"
                f"📍 Причина: {reason}\n"
                f"💰 Вход: ${entry_price:.4f}\n"
                f"💵 Выход: ${exit_price:.4f}\n"
                f"📊 P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
            
            logger.info(f"✅ Позиция {symbol} закрыта: {pnl:+.2f} ({pnl_pct:+.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка закрытия позиции: {e}")
            health_monitor.record_error("close_position", str(e))
            return False
    
    async def pause_bot(self):
        """Пауза бота"""
        self.paused = True
        logger.info("⏸️ Бот поставлен на паузу")
    
    async def shutdown(self):
        """Корректное отключение"""
        logger.info("🛑 Отключение бота...")
        
        self.running = False
        
        if self.telegram_app:
            await self.telegram_app.stop()
            await self.telegram_app.shutdown()
        
        await exchange_manager.disconnect()
        
        logger.info("✅ Бот отключен")
    
    # === TELEGRAM КОМАНДЫ ===
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        await update.message.reply_text(
            "🤖 *БОТ V2.0* - Управление\n\n"
            "*📊 Информация:*\n"
            "/status - краткий статус\n"
            "/positions - детали позиций\n"
            "/history - история сделок\n\n"
            "*🎮 Управление:*\n"
            "/pause - пауза бота\n"
            "/resume - возобновить\n"
            "/close\\_all - закрыть все позиции\n"
            "/stop - остановить бота",
            parse_mode='MarkdownV2'
        )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status - краткий статус"""
        balance = await exchange_manager.get_balance()
        
        status_emoji = "🟢" if self.running and not self.paused else "⏸️" if self.paused else "🔴"
        
        status_text = (
            f"📊 *СТАТУС БОТА V2.0*\n\n"
            f"{status_emoji} *Работает*\n"
            f"💰 Баланс: ${balance:.2f}\n"
            f"📊 Позиций: {len(self.open_positions)}/{Config.MAX_POSITIONS}\n"
            f"🧪 Тест: {'ДА' if Config.TEST_MODE else 'НЕТ'}\n"
            f"📈 P&L день: ${-risk_manager.daily_loss:.2f}\n"
            f"🔢 Сделок: {risk_manager.trades_today}\n"
            f"⚠️ Ошибок: {self.bot_errors_count}"
        )
        
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /positions - детали всех позиций"""
        try:
            balance = await exchange_manager.get_balance()
            live_positions = await exchange_manager.fetch_positions()
            
            if not live_positions:
                await update.message.reply_text("✅ Нет открытых позиций")
                return
            
            msg = f"💰 Баланс: ${balance:.2f} USDT\n"
            msg += f"📊 Открытых позиций: {len(live_positions)}\n\n"
            
            total_pnl = 0
            for pos in live_positions:
                symbol = pos['symbol']
                side = pos['side']
                size = pos['contracts']
                entry = pos['entryPrice']
                pnl = pos.get('unrealizedPnl', 0)
                leverage = pos.get('leverage', 5)
                
                info = pos.get('info', {})
                stop_loss = info.get('stopLoss', 'НЕТ')
                
                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                total_pnl += pnl
                
                msg += f"{pnl_emoji} {symbol} | {side.upper()} | {leverage}X\n"
                msg += f"   Entry: {entry:.4f}\n"
                msg += f"   Size: {size:.4f}\n"
                msg += f"   PnL: ${pnl:.4f}\n"
                msg += f"   SL: {stop_loss}\n\n"
            
            msg += f"💵 TOTAL PnL: ${total_pnl:.4f}"
            
            await update.message.reply_text(msg)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /history - история закрытых сделок"""
        try:
            import json
            
            try:
                with open("trade_history.json", 'r') as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                await update.message.reply_text("📊 История пуста - нет закрытых сделок")
                return
            
            if not history:
                await update.message.reply_text("📊 История пуста")
                return
            
            # Показываем последние 10 сделок
            recent = history[-10:]
            
            total_pnl = sum([t['pnl'] for t in history])
            winning = len([t for t in history if t['pnl'] > 0])
            
            msg = f"📊 *ИСТОРИЯ СДЕЛОК*\n\n"
            msg += f"Всего: {len(history)} | Win Rate: {winning/len(history)*100:.0f}%\n"
            msg += f"Total P&L: ${total_pnl:.2f}\n\n"
            msg += f"*Последние {len(recent)} сделок:*\n\n"
            
            for trade in reversed(recent):
                emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                msg += f"{emoji} {trade['symbol']} | {trade['reason']}\n"
                msg += f"   P&L: ${trade['pnl']:.2f} ({trade['pnl_percent']:.1f}%)\n"
                msg += f"   {trade['close_time'][:16]}\n\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /close_all - закрыть все позиции"""
        try:
            if not self.open_positions:
                await update.message.reply_text("✅ Нет открытых позиций")
                return
            
            count = len(self.open_positions)
            await update.message.reply_text(f"🔄 Закрываю {count} позиций...")
            
            closed = 0
            for pos in list(self.open_positions):
                success = await self.close_position(pos, reason="Manual (Telegram)")
                if success:
                    closed += 1
            
            await update.message.reply_text(
                f"✅ Закрыто: {closed}/{count} позиций"
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /pause"""
        await self.pause_bot()
        await update.message.reply_text("⏸️ Бот поставлен на паузу")
    
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /resume"""
        self.paused = False
        await update.message.reply_text("▶️ Бот возобновлен")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /stop"""
        await update.message.reply_text("🛑 Останавливаю бота...")
        await self.shutdown()


async def main():
    """Главная функция"""
    bot = TradingBotV2()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("⚠️ Получен сигнал остановки")
        await bot.shutdown()
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        await bot.shutdown()
        raise


if __name__ == "__main__":
    asyncio.run(main())

