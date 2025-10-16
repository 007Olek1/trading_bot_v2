#!/usr/bin/env python3
"""
🤖 ТОРГОВЫЙ БОТ V3.0 - ПРОФЕССИОНАЛЬНАЯ ТОРГОВАЯ СИСТЕМА
✅ Продвинутый анализ (10+ индикаторов, уровни, объемы)
✅ Множественные TP + Trailing Stop
✅ Анализ свечных паттернов
✅ Профессиональные отчеты в Telegram
✅ AI агенты + автоисправление
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

# Наши модули
from bot_v2_config import Config
from bot_v2_safety import risk_manager, emergency_stop, position_guard
from bot_v2_exchange import exchange_manager
from bot_v2_signals import signal_analyzer  # Старый (для совместимости)
from bot_v2_advanced_signals import advanced_signal_analyzer  # Новый продвинутый
from bot_v2_position_manager import position_manager
from bot_v2_candle_analyzer import candle_analyzer
from bot_v2_ai_agent import trading_bot_agent, health_monitor
from bot_v2_auto_healing import auto_healing
from bot_v2_super_ai_agent import super_ai_agent
# ML/LLM агенты V3.5
from bot_v3_ml_engine import ml_engine
from bot_v3_llm_agent import llm_agent
# Система самомониторинга V3.5
from bot_v3_self_monitor import self_monitor

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG if Config.LOG_LEVEL == "DEBUG" else logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
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
        # Символы, которые прямо сейчас находятся в процессе открытия (анти-дубликаты)
        self.pending_symbols = set()
        
        # Статистика сигналов
        self.signals_stats = {
            'total_analyzed': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'weak_signals': 0,
            'no_signals': 0
        }
        
        # Telegram
        self.telegram_app = None
        
        logger.info("=" * 60)
        logger.info("🤖 ТОРГОВЫЙ БОТ V3.5 AUTONOMOUS ML/LLM - САМООБУЧАЮЩАЯСЯ СИСТЕМА")
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
            self.telegram_app.add_handler(CommandHandler("stop", self.cmd_stop))
            self.telegram_app.add_handler(CommandHandler("pause", self.cmd_pause))
            self.telegram_app.add_handler(CommandHandler("resume", self.cmd_resume))
            
            # Запуск Telegram в фоне (без polling - только уведомления)
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            
            # 5. Планировщик задач
            scheduler = AsyncIOScheduler()
            
            # Основной торговый цикл - СИНХРОНИЗАЦИЯ С ЗАКРЫТИЕМ СВЕЧЕЙ!
            # Запускаем в :00, :15, :30, :45 каждого часа (когда 15-мин свечи закрываются)
            scheduler.add_job(
                self.trading_loop,
                'cron',
                minute='0,15,30,45',  # Точное время закрытия свечей
                second=5,  # +5 сек после закрытия для гарантии
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
            
            # 6. ЗАПУСК СИСТЕМЫ САМОМОНИТОРИНГА
            logger.info("👁️ Запуск системы самомониторинга...")
            asyncio.create_task(self_monitor.start())
            
            scheduler.start()
            
            self.running = True
            
            # 7. ПРОВЕРКА СУЩЕСТВУЮЩИХ ПОЗИЦИЙ ПРИ СТАРТЕ
            await self._check_existing_positions_on_startup()
            
            # Уведомление о запуске
            mode_emoji = "🧪" if Config.TEST_MODE else "💰"
            mode_text = "ТЕСТОВЫЙ РЕЖИМ" if Config.TEST_MODE else "РАБОЧИЙ РЕЖИМ"
            
            # Take Profit info
            tp_text = f"+25% + Trailing Stop"  # Фиксированный текст
            
            # Простое стартовое сообщение
            position_size = Config.get_position_size()
            leverage = Config.LEVERAGE
            
            # Компактное стартовое сообщение
            llm_status = "✅" if Config.USE_LLM_FILTER else "⏸️"
            await self.send_telegram(
                f"🚀 *БОТ V3.6 ULTRA SAFE*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 ${balance:.2f} | 💎 РЕАЛЬНАЯ ТОРГОВЛЯ\n"
                f"📊 {leverage}X | 💵 ${position_size}/сделка | 🛡️ SL -10%\n"
                f"🎯 Мин. уверенность: *{Config.MIN_CONFIDENCE_PERCENT}%*\n"
                f"📈 ТОП 100 монет | ⏰ Анализ каждые 15 мин\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🧠 ML | 🤖 LLM {llm_status} | 👁️ Auto-Healing\n"
                f"✅ Все системы активны!"
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
            # КРИТИЧНО: Проверяем Emergency Stop
            if emergency_stop.emergency_stopped:
                logger.critical("⛔ Торговля заблокирована: EMERGENCY STOP активен!")
                return
            
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
            
            # 3. Проверка аварийных условий (КРИТИЧНО: передаем РЕАЛЬНЫЕ позиции с биржи!)
            real_positions = await exchange_manager.fetch_positions()
            
            should_stop, reason = await emergency_stop.check_emergency_conditions(
                risk_manager,
                real_positions,  # НЕ self.open_positions!
                self.bot_errors_count
            )
            
            if should_stop:
                logger.critical(f"🚨 EMERGENCY STOP: {reason}")
                await self.emergency_shutdown(reason)
                return
            
            # 4. Проверка лимита позиций
            if len(self.open_positions) >= Config.MAX_POSITIONS:
                logger.info(f"📊 Лимит позиций ({Config.MAX_POSITIONS}) достигнут")
                return
            
            # 5. Получаем баланс
            balance = await exchange_manager.get_balance()
            if balance is None:
                logger.error("❌ Не удалось получить баланс")
                return
            
            # 6. Проверка возможности открыть сделку
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
            
            # 7. Используем ФИКСИРОВАННЫЙ список ТОП 100 популярных монет
            logger.info("📊 Используем ТОП 100 популярных монет...")
            
            # Уведомление о начале анализа
            current_time = datetime.now().strftime('%H:%M:%S')
            await self.send_telegram(
                f"🔄 *ТОРГОВЫЙ ЦИКЛ*\n\n"
                f"⏰ {current_time}\n"
                f"📊 ТОП 100 популярных монет\n"
                f"🔍 Начинаю анализ...\n"
                f"⏱️ ~3 минуты"
            )
            
            # Используем фиксированный список из конфига
            symbols = Config.TOP_100_SYMBOLS
            
            if not symbols:
                logger.warning("⚠️ Список символов пуст")
                await self.send_telegram(f"⚠️ Ошибка: список монет пуст")
                return
            
            logger.info(f"🔍 Анализ {len(symbols)} символов...")
            
            # 8. КРИТИЧНО: Собираем ВСЕ сильные сигналы (≥85%) для AI анализа
            all_signals = []
            strong_signals_found = 0
            
            for symbol in symbols:
                # Пропускаем если уже есть позиция
                if any(p['symbol'] == symbol for p in self.open_positions):
                    continue
                # Пропускаем, если открытие по символу уже в процессе (между итерациями/джобами)
                if symbol in self.pending_symbols:
                    logger.debug(f"⏳ Пропуск {symbol}: открытие уже инициировано")
                    continue
                
                # Получаем ПОЛНЫЙ анализ (включая сигналы 85%+)
                signal_result = await self.analyze_symbol_full(symbol)
                
                # Добавляем ТОЛЬКО сильные сигналы (≥85%)
                if signal_result and signal_result.get('signal'):
                    if signal_result.get('confidence', 0) >= Config.MIN_CONFIDENCE_PERCENT:
                        all_signals.append({**signal_result, 'symbol': symbol})
                        strong_signals_found += 1
                        logger.info(f"💎 СИЛЬНЫЙ СИГНАЛ: {symbol} {signal_result['signal'].upper()} {signal_result['confidence']:.0f}%")
            
            logger.info(f"🎯 Найдено сильных сигналов (≥{Config.MIN_CONFIDENCE_PERCENT}%): {strong_signals_found}")
            
            # Уведомление о результатах анализа
            current_time = datetime.now().strftime('%H:%M:%S')
            await self.send_telegram(
                f"🎯 *РЕЗУЛЬТАТЫ АНАЛИЗА*\n\n"
                f"⏰ {current_time}\n"
                f"📊 Проанализировано: {len(symbols)} монет\n"
                f"💎 Сильных сигналов (≥{Config.MIN_CONFIDENCE_PERCENT}%): {strong_signals_found}"
            )
            
            # 9. СУПЕР AI АГЕНТ выбирает ЛУЧШИЙ сигнал
            logger.info(f"🧠 AI АГЕНТ: Анализ {len(all_signals)} сигналов...")
            
            if all_signals:
                await self.send_telegram(
                    f"🧠 *AI АГЕНТ*\n\n"
                    f"Анализирую {len(all_signals)} сильных сигналов..."
                )
            
            # Анализ рыночных условий
            market_conditions = await super_ai_agent.analyze_market_conditions(all_signals)
            
            # AI выбор лучшего сигнала
            best_signal = await super_ai_agent.select_best_signal(
                signals=all_signals,
                current_positions=len(self.open_positions),
                balance=await exchange_manager.get_balance()
            )
            
            # Если нет сигналов - уведомляем
            if not best_signal:
                current_time = datetime.now().strftime('%H:%M:%S')
                await self.send_telegram(
                f"😴 *НЕТ КАЧЕСТВЕННЫХ СИГНАЛОВ*\n\n"
                f"⏰ {current_time}\n"
                f"📊 Проверено: {len(symbols)} монет\n"
                f"🎯 Сигналов ≥{Config.MIN_CONFIDENCE_PERCENT}%: {strong_signals_found}\n"
                    f"🧠 AI выбрал: 0\n\n"
                    f"⏳ Следующий анализ через 15 минут"
                )
            
            if best_signal and len(self.open_positions) < Config.MAX_POSITIONS:
                symbol = best_signal['symbol']
                
                # ФИНАЛЬНАЯ ПРОВЕРКА AI АГЕНТА (120% контроль!)
                logger.info(f"🧠 AI: Финальная проверка {symbol}...")
                ai_approved, ai_reason = await super_ai_agent.validate_trade_before_open(
                    symbol=symbol,
                    side=best_signal['signal'],
                    signal_data=best_signal,
                    balance=await exchange_manager.get_balance()
                )
                
                if not ai_approved:
                    logger.warning(f"🧠 AI ЗАБЛОКИРОВАЛ: {ai_reason}")
                else:
                    logger.info(f"🧠 AI ОДОБРИЛ: {ai_reason}")
                    
                    # Открываем позицию с анти-дубликат защитой
                    if symbol in self.pending_symbols:
                        logger.debug(f"⏳ Пропуск {symbol}: уже открывается")
                    else:
                        self.pending_symbols.add(symbol)
                        try:
                            position = await self.open_position(
                                symbol=symbol,
                                side=best_signal['signal'],
                                signal_data=best_signal
                            )
                        finally:
                            self.pending_symbols.discard(symbol)
                    
                    if position:
                        logger.info(f"✅ Позиция открыта: {symbol}")
                        super_ai_agent.decisions_made += 1
                        # Пауза 30 секунд
                        logger.info("⏸️ Пауза 30 сек перед следующим анализом...")
                        await asyncio.sleep(30)
            
            logger.info("✅ Торговый цикл завершен")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Ошибка в торговом цикле: {e}")
            self.bot_errors_count += 1
    
    async def analyze_symbol_full(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        ПОЛНЫЙ анализ символа (возвращает ВСЕ сигналы, включая ≥85%)
        Используется для сбора сильных сигналов для AI
        """
        try:
            # Получаем свечи (15-минутный таймфрейм!)
            ohlcv = await exchange_manager.fetch_ohlcv(symbol, timeframe="15m", limit=100)
            if not ohlcv:
                return None
            
            # Конвертируем в DataFrame (используем ВСЕ свечи, биржа дает закрытые данные)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            logger.debug(f"✅ {symbol}: Анализирую {len(df)} свечей...")
            
            # Анализ сигнала (продвинутый)
            signal_result = advanced_signal_analyzer.analyze(df)
            
            # Анализ свечей
            candle_result = candle_analyzer.analyze_candle_close(df)
            signal_result['candle_analysis'] = candle_result
            
            # Проверка: сильная свеча подтверждает сигнал
            if candle_result.get('strong') and signal_result.get('signal'):
                if candle_result.get('bullish') and signal_result['signal'] == 'buy':
                    signal_result['confidence'] = min(100, signal_result['confidence'] * 1.1)
                elif candle_result.get('bearish') and signal_result['signal'] == 'sell':
                    signal_result['confidence'] = min(100, signal_result['confidence'] * 1.1)
            
            # 🧠 ML ENGINE: Предсказание сигнала с помощью XGBoost + LSTM
            current_price = float(df['close'].iloc[-1])
            ml_result = await ml_engine.predict_signal(df, signal_result, current_price)
            signal_result['ml_prediction'] = ml_result
            
            # Если ML улучшил сигнал - используем его
            if ml_result.get('signal'):
                signal_result['signal'] = ml_result['signal']
                signal_result['confidence'] = ml_result['confidence']
                signal_result['ml_enhanced'] = True
                logger.debug(f"🧠 ML улучшил сигнал {symbol}: {ml_result['signal']} {ml_result['confidence']:.1f}%")
            
            # Записываем успешный анализ
            health_monitor.record_successful_analysis()
            
            # ВОЗВРАЩАЕМ ПОЛНЫЙ РЕЗУЛЬТАТ (даже если signal=None)
            return signal_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа {symbol}: {e}")
            health_monitor.record_error("analysis", str(e))
            return None
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Анализ символа С ОЖИДАНИЕМ ЗАКРЫТИЯ СВЕЧИ
        (Старая версия - оставлена для совместимости)
        """
        try:
            # Получаем свечи (15-минутный таймфрейм!)
            ohlcv = await exchange_manager.fetch_ohlcv(symbol, timeframe="15m", limit=100)
            if not ohlcv:
                return None
            
            # КРИТИЧНО: Проверяем что последняя свеча ЗАКРЫТА
            last_candle_time = ohlcv[-1][0]  # timestamp
            current_time = pd.Timestamp.now().timestamp() * 1000
            time_since_candle = current_time - last_candle_time
            
            # Свеча 15 минут = 900000 мс. Ждем когда пройдет минимум 15 мин
            CANDLE_INTERVAL_MS = 15 * 60 * 1000  # 900000 мс
            if time_since_candle < CANDLE_INTERVAL_MS:
                # Текущая свеча еще не закрыта - пропускаем
                logger.debug(f"⏳ {symbol}: Ожидание закрытия 15-мин свечи ({int(time_since_candle/1000)}с/{int(CANDLE_INTERVAL_MS/1000)}с)")
                return None
            
            logger.info(f"✅ {symbol}: Свеча закрыта, анализирую...")
            
            # Конвертируем в DataFrame (БЕЗ последней незакрытой свечи)
            df = pd.DataFrame(ohlcv[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Анализ сигнала (продвинутый)
            signal_result = advanced_signal_analyzer.analyze(df)
            
            # Анализ свечей
            candle_result = candle_analyzer.analyze_candle_close(df)
            signal_result['candle_analysis'] = candle_result
            
            # Проверка: сильная свеча подтверждает сигнал
            if candle_result.get('strong') and signal_result.get('signal'):
                if candle_result.get('bullish') and signal_result['signal'] == 'buy':
                    signal_result['confidence'] = min(100, signal_result['confidence'] * 1.1)
                elif candle_result.get('bearish') and signal_result['signal'] == 'sell':
                    signal_result['confidence'] = min(100, signal_result['confidence'] * 1.1)
            
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
            
            # 0.5 🤖 LLM АНАЛИЗ РЫНОЧНОГО КОНТЕКСТА (если включен)
            if Config.USE_LLM_FILTER:
                ml_prediction = signal_data.get('ml_prediction', {})
                market_conditions = {
                    'volatility': signal_data.get('volatility', 'MEDIUM'),
                    'trend': signal_data.get('trend', 'NEUTRAL'),
                    'sentiment': 'NEUTRAL'
                }
                
                llm_analysis = await llm_agent.analyze_market_context(
                    symbol=symbol,
                    current_price=signal_data.get('current_price', 0),
                    signal_result=signal_data,
                    ml_result=ml_prediction,
                    market_conditions=market_conditions
                )
                
                logger.info(f"🤖 LLM рекомендация: {llm_analysis.get('recommendation', 'UNKNOWN')} "
                           f"({llm_analysis.get('confidence', 0):.0f}%) - {llm_analysis.get('risk_level', 'UNKNOWN')} risk")
                
                # КРИТИЧНО: Проверяем рекомендацию LLM
                llm_recommendation = llm_analysis.get('recommendation', '').lower()
                llm_confidence = llm_analysis.get('confidence', 0)
                llm_risk = llm_analysis.get('risk_level', 'UNKNOWN')
                
                current_time = datetime.now().strftime('%H:%M:%S')
                
                # Если LLM говорит HOLD - НЕ ОТКРЫВАЕМ!
                if llm_recommendation == 'hold':
                    logger.warning(f"🤖 LLM БЛОКИРОВАЛ сделку: рекомендация HOLD")
                    
                    await self.send_telegram(
                        f"⚠️ *LLM БЛОКИРОВАЛ СДЕЛКУ*\n\n"
                        f"⏰ {current_time}\n"
                        f"💎 {symbol}\n"
                        f"📊 Сигнал: {side.upper()} {signal_data.get('confidence', 0):.0f}%\n"
                        f"🤖 LLM: HOLD ({llm_confidence:.0f}%)\n"
                        f"⚠️ Риск: {llm_risk}\n\n"
                        f"❌ Сделка отменена"
                    )
                    return None
                
                # Если LLM дает противоположную рекомендацию - НЕ ОТКРЫВАЕМ!
                if side == 'buy' and llm_recommendation == 'sell':
                    logger.warning(f"🤖 LLM БЛОКИРОВАЛ: сигнал BUY, но LLM рекомендует SELL")
                    
                    await self.send_telegram(
                        f"⚠️ *LLM БЛОКИРОВАЛ СДЕЛКУ*\n\n"
                        f"⏰ {current_time}\n"
                        f"💎 {symbol}\n"
                        f"📊 Сигнал: BUY {signal_data.get('confidence', 0):.0f}%\n"
                        f"🤖 LLM: SELL ({llm_confidence:.0f}%)\n"
                        f"⚠️ Противоречие!\n\n"
                        f"❌ Сделка отменена"
                    )
                    return None
                elif side == 'sell' and llm_recommendation == 'buy':
                    logger.warning(f"🤖 LLM БЛОКИРОВАЛ: сигнал SELL, но LLM рекомендует BUY")
                    
                    await self.send_telegram(
                        f"⚠️ *LLM БЛОКИРОВАЛ СДЕЛКУ*\n\n"
                        f"⏰ {current_time}\n"
                        f"💎 {symbol}\n"
                        f"📊 Сигнал: SELL {signal_data.get('confidence', 0):.0f}%\n"
                        f"🤖 LLM: BUY ({llm_confidence:.0f}%)\n"
                        f"⚠️ Противоречие!\n\n"
                        f"❌ Сделка отменена"
                    )
                    return None
                
                logger.info(f"✅ LLM ОДОБРИЛ сделку: {llm_recommendation.upper()} совпадает с сигналом")
                
                await self.send_telegram(
                    f"✅ *LLM ОДОБРИЛ СДЕЛКУ*\n\n"
                    f"⏰ {current_time}\n"
                    f"💎 {symbol}\n"
                    f"📊 Сигнал: {side.upper()} {signal_data.get('confidence', 0):.0f}%\n"
                    f"🤖 LLM: {llm_recommendation.upper()} ({llm_confidence:.0f}%)\n"
                    f"🛡️ Риск: {llm_risk}\n\n"
                    f"🚀 Открываю позицию..."
                )
            else:
                logger.info(f"📊 LLM фильтр отключен - открываем по AI сигналу")
            
            # 1. Получаем текущую цену
            ohlcv = await exchange_manager.fetch_ohlcv(symbol, limit=1)
            if not ohlcv:
                logger.error("❌ Не удалось получить цену")
                return None
            
            current_price = float(ohlcv[-1][4])  # close price

            # 1.1 Доп. защита: перед входом проверяем, нет ли уже реальной позиции по символу на бирже
            try:
                live_positions = await exchange_manager.fetch_positions()
                if any(p.get('symbol') == symbol and float(p.get('contracts', 0) or 0) > 0 for p in live_positions):
                    logger.warning(f"🛑 Пропуск открытия {symbol}: позиция уже существует на бирже")
                    return None
            except Exception:
                # Если проверка не удалась, не блокируем, но записываем debug
                logger.debug("⚠️ Не удалось проверить наличие позиции перед входом")
            
            # 2. Устанавливаем леверидж
            await exchange_manager.set_leverage(symbol, Config.LEVERAGE)
            
            # 3. Рассчитываем размер позиции
            balance = await exchange_manager.get_balance()
            position_size_usd = risk_manager.calculate_position_size(balance)
            amount = (position_size_usd * Config.LEVERAGE) / current_price
            
            # 4. Рассчитываем SL и TP
            stop_loss, take_profit = risk_manager.calculate_sl_tp_prices(current_price, side)
            
            # 5. Открываем позицию на бирже
            logger.info(f"💰 Создаю market ордер: {amount:.6f} @ ${current_price:.4f}")
            market_order = await exchange_manager.create_market_order(symbol, side, amount)
            
            if not market_order:
                logger.error("❌ Не удалось создать market ордер")
                return None
            
            # 6. КРИТИЧНО: Создаем Stop Loss ордер НА БИРЖЕ
            logger.info("🛡️ Создаю Stop Loss ордер на бирже...")
            close_side = "sell" if side == "buy" else "buy"
            
            sl_order = await exchange_manager.create_stop_market_order(
                symbol=symbol,
                side=close_side,
                amount=amount,
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
            
            # 7. Создаем МНОГОУРОВНЕВЫЕ Take Profit ордера
            # Targets: +0.6%, +1.3%, +1.9%, +2.6%, +3.2%
            tp_levels = [0.006, 0.013, 0.019, 0.026, 0.032]  # Проценты изменения цены
            tp_percentages = [0.20, 0.20, 0.20, 0.20, 0.20]  # По 20% позиции на каждый TP
            
            tp_orders = []
            tp_prices = []
            
            for i, tp_pct in enumerate(tp_levels):
                if side == "buy":
                    tp_price = current_price * (1 + tp_pct)
                else:
                    tp_price = current_price * (1 - tp_pct)
                
                tp_prices.append(tp_price)
                
                # Размер для этого TP
                tp_amount = amount * tp_percentages[i]
                
                # Создаем ордер
            tp_order = await exchange_manager.create_limit_order(
                symbol=symbol,
                side=close_side,
                    amount=tp_amount,
                    price=tp_price
                )
                
                if tp_order:
                    tp_orders.append(tp_order['id'])
                    logger.info(f"✅ TP{i+1} создан: ${tp_price:.4f} ({tp_percentages[i]*100:.0f}% позиции)")
                else:
                    tp_orders.append(None)
                    logger.warning(f"⚠️ TP{i+1} не создан")
            
            # 8. Создаем запись о позиции
            position = {
                "symbol": symbol,
                "side": side,
                "entry_price": current_price,
                "amount": amount,
                "stop_loss": stop_loss,
                "take_profit": take_profit,  # Сохраняем максимальный TP для совместимости
                "tp_prices": tp_prices,  # Все уровни TP
                "tp_orders": tp_orders,  # ID всех TP ордеров
                "sl_order_id": sl_order['id'],
                "tp_order_id": tp_orders[0] if tp_orders else None,  # Для совместимости
                "market_order_id": market_order['id'],
                "open_time": datetime.now(),
                "signal_confidence": signal_data['confidence'],
                "signal_reason": signal_data['reason']
            }
            
            self.open_positions.append(position)
            
            # 9. Уведомление с МНОГОУРОВНЕВЫМИ TARGETS
            sl_pct = abs((stop_loss - current_price) / current_price * 100)
            
            # Форматируем все targets
            targets_text = ""
            for i, tp_price in enumerate(tp_prices):
                tp_pct = abs((tp_price - current_price) / current_price * 100)
                emoji = ["🥇", "🥈", "🥉", "💎", "🚀"][i]
                targets_text += f"   {emoji} ${tp_price:.4f} (+{tp_pct:.1f}%)\n"
            
            # Компактное сообщение о позиции
            await self.send_telegram(
                f"🟢 *ПОЗИЦИЯ ОТКРЫТА*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"💎 *{symbol}* | {side.upper()} | {Config.LEVERAGE}X\n"
                f"💰 Entry: *${current_price:.4f}* | Размер: *${position_size_usd:.2f}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 *Targets:*\n{targets_text}"
                f"🛡️ SL: *${stop_loss:.4f}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎲 Уверенность: *{signal_data['confidence']:.0f}%*\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
            
            logger.info(f"✅ Позиция {symbol} успешно открыта с защитой!")
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
                    # Получаем ордера для этой позиции
                    orders = await exchange_manager.fetch_open_orders(symbol)
                    
                    sl_order_id = None
                    tp_order_id = None
                    
                    for order in orders:
                        if order.get('type') == 'STOP_MARKET':
                            sl_order_id = order['id']
                        elif order.get('type') == 'LIMIT':
                            tp_order_id = order['id']
                    
                    position = {
                        "symbol": symbol,
                        "side": ex_pos.get('side'),
                        "entry_price": float(ex_pos.get('entryPrice', 0)),
                        "amount": size,
                        "sl_order_id": sl_order_id,
                        "tp_order_id": tp_order_id,
                        "open_time": datetime.now(),  # Примерно
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
            # 1. ПРОВЕРКА ЗДОРОВЬЯ БОТА
            is_healthy = health_monitor.is_healthy()
            
            if not is_healthy:
                health_status = health_monitor.get_status()
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
            
            # 3. Проверка SL на ВСЕХ позициях (проверяем РЕАЛЬНОЕ состояние биржи!)
            exchange_positions = await exchange_manager.fetch_positions()
            
            for position in exchange_positions:
                symbol = position['symbol']
                
                # Проверяем наличие SL напрямую с биржи (НЕ из памяти!)
                stop_loss = position.get('stopLoss') or position.get('info', {}).get('stopLoss')
                
                # Если SL отсутствует или равен 0
                if not stop_loss or stop_loss == "" or stop_loss == "0":
                        logger.critical(
                        f"🚨 {symbol}: ПОЗИЦИЯ БЕЗ SL (реальная проверка биржи)! "
                            f"Auto-Healing активирован!"
                        )
                    health_monitor.record_error("missing_sl_order", symbol)
                    
                    # Создаем SL немедленно
                    entry_price = float(position['entryPrice'])
                    amount = float(position['contracts'])
                    side = position['side'].lower()
                    sl_price = entry_price * (0.9 if side == 'long' else 1.1)
                    
                    try:
                        sl_order = await exchange_manager.create_stop_market_order(
                            symbol=symbol,
                            side='sell' if side == 'long' else 'buy',
                            amount=amount,
                            stop_price=round(sl_price, 4)
                        )
                        
                        if sl_order and sl_order.get('status') == 'set':
                            logger.info(f"✅ {symbol}: SL восстановлен @ ${sl_price:.4f}")
                        else:
                            raise Exception("SL не создан")
                            
                    except Exception as e:
                        logger.critical(f"❌ {symbol}: Не смогли создать SL: {e}")
                        # КРИТИЧНО: не смогли создать SL - закрываем позицию!
                        close_side = "sell" if side == 'long' else "buy"
                        await exchange_manager.create_market_order(symbol, close_side, amount)
                        logger.critical(f"🚨 {symbol}: Позиция экстренно закрыта!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка health check: {e}")
            health_monitor.record_error("health_check", str(e))
    
    async def send_heartbeat(self):
        """Отправка heartbeat сообщения каждый час"""
        try:
            balance = await exchange_manager.get_balance()
            warsaw_time = datetime.now(pytz.timezone('Europe/Warsaw'))
            
            # Статус
            status_emoji = "🟢" if self.running and not self.paused else "⏸️" if self.paused else "🔴"
            test_mode_text = "🧪 ТЕСТОВЫЙ РЕЖИМ" if Config.TEST_MODE else "💰 РАБОЧИЙ РЕЖИМ"
            
            # Информация о позициях
            positions_text = ""
            if self.open_positions:
                for pos in self.open_positions:
                    positions_text += f"\n   • {pos['symbol']} {pos['side'].upper()}"
            else:
                positions_text = "\n   ✅ Нет открытых позиций"
            
            # Получаем отчеты агентов
            agent_report = trading_bot_agent.get_performance_report()
            health_report = health_monitor.get_health_report()
            
            # Статус здоровья
            health_emoji = "✅" if health_report['is_healthy'] else "⚠️"
            
            # СУПЕР AI АГЕНТ отчет
            super_ai_status = super_ai_agent.get_quick_status()
            
            # ML/LLM статус
            from bot_v3_ml_engine import ml_engine
            from bot_v3_llm_agent import llm_agent
            
            ml_status = ml_engine.get_status()
            llm_status = llm_agent.get_status()
            
            # ML строка
            ml_text = ""
            if ml_status['model_trained']:
                ml_text = f"🧠 *ML ENGINE (XGBoost):*\n"
                ml_text += f"   Точность: {ml_status['accuracy']:.1%}\n"
                ml_text += f"   Обучено на: {ml_status['training_samples']} сделок\n"
                if ml_status['accuracy'] >= 0.85:
                    ml_text += f"   🎯 Целевая точность достигнута!\n"
                ml_text += "\n"
            else:
                ml_text = f"🧠 *ML ENGINE:* Ожидание данных (0/{ml_status.get('min_samples', 50)} сделок)\n\n"
            
            # LLM строка
            llm_text = f"🤖 *LLM AGENT (GPT-4):* {'✅ Активен' if llm_status['enabled'] else '⚠️ Отключен'}\n"
            if llm_status['enabled'] and llm_status['total_analyses'] > 0:
                validation_rate = (llm_status['successful_validations'] / llm_status['total_analyses']) * 100
                llm_text += f"   Анализов: {llm_status['total_analyses']}\n"
                llm_text += f"   Одобрено: {llm_status['successful_validations']} ({validation_rate:.0f}%)\n"
                llm_text += f"   Отклонено: {llm_status['rejected_signals']}\n"
            llm_text += "\n"
            
            # AI агент для выбора сигналов
            ai_text = f"🎯 *AI СТРАТЕГ:*\n{super_ai_status}\n\n"
            
            # Базовый агент (риск-менеджмент)
            base_text = f"🛡️ *РИСК-МЕНЕДЖЕР:*\n"
            base_text += f"   Win Rate: {agent_report['win_rate']:.0%}\n"
            base_text += f"   Profit Factor: {agent_report['profit_factor']:.2f}\n"
            base_text += f"   Сделок записано: {agent_report['total_trades']}\n\n"
            
            # Система самомониторинга
            monitor_status = self_monitor.get_status()
            monitor_text = f"👁️ *САМОМОНИТОРИНГ:*\n"
            monitor_text += f"   Проверок: {monitor_status['total_checks']}\n"
            monitor_text += f"   Проблем найдено: {monitor_status['issues_found']}\n"
            monitor_text += f"   Автоисправлено: {monitor_status['auto_fixed']} ({monitor_status['fix_rate']})\n"
            monitor_text += f"   Улучшений: {monitor_status['improvement_actions']}\n\n"
            
            # Компактное сообщение
            status_text = 'Работает' if self.running and not self.paused else 'Пауза' if self.paused else 'Остановлен'
            pnl_value = -risk_manager.daily_loss
            pnl_emoji = "📈" if pnl_value >= 0 else "📉"
            
            # Статус ML/LLM кратко
            ml_short = f"Данные {len(ml_engine.training_data)}/50" if not ml_status['model_trained'] else f"Точн. {ml_status['accuracy']:.0%}"
            llm_short = "✅" if llm_status['enabled'] else "⚠️"
            
            # Основное сообщение (≤14 строк)
            message = (
                f"💓 *HEARTBEAT V3.6*\n"
                f"{status_emoji} {status_text} | ⏰ {warsaw_time.strftime('%H:%M:%S')}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Баланс: *${balance:.2f}* USDT\n"
                f"📊 Позиций: *{len(self.open_positions)}/{Config.MAX_POSITIONS}* | Сделок: *{risk_manager.trades_today}/{Config.MAX_TRADES_PER_DAY}*\n"
                f"{pnl_emoji} P&L: *${pnl_value:.2f}* | Убытков: *{risk_manager.consecutive_losses}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🧠 ML: {ml_short} | 🤖 LLM: {llm_short}\n"
                f"🎯 WinRate: *{agent_report['win_rate']:.0%}* | PF: *{agent_report['profit_factor']:.2f}*\n"
                f"🛡️ Здоровье: {health_emoji} | Ошибок: *{health_report['total_errors']}*"
            )
            
            # Добавляем позиции если есть
            if self.open_positions:
                message += f"\n━━━━━━━━━━━━━━━━━━━━━\n📊 *ПОЗИЦИИ:*"
                for pos in self.open_positions[:3]:  # Максимум 3
                    message += f"\n  • {pos['symbol']} {pos['side'].upper()}"
            
            # Если есть проблемы - добавляем детали
            if health_report['total_errors'] > 0 or monitor_status['issues_found'] > 0:
                message += f"\n━━━━━━━━━━━━━━━━━━━━━\n⚠️ *ВНИМАНИЕ:*"
                if health_report['total_errors'] > 0:
                    message += f"\n  Ошибок: {health_report['total_errors']}"
                if monitor_status['issues_found'] > 0:
                    message += f"\n  Проблем: {monitor_status['issues_found']}"
                    message += f"\n  Исправлено: {monitor_status['auto_fixed']}"
            
            await self.send_telegram(message)
            
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
            # КРИТИЧНО: Получаем РЕАЛЬНЫЕ позиции с биржи!
            real_positions = await exchange_manager.fetch_positions()
            
            # Закрыть все позиции
            for position in real_positions:
                side = position['side'].lower()
                close_side = "sell" if side == "long" else "buy"
                amount = float(position['contracts'])
                
                logger.info(f"🚀 Закрываю {position['symbol']}: {close_side} {amount}")
                
                await exchange_manager.create_market_order(
                    position['symbol'],
                    close_side,
                    amount
                )
            
            # Отменить все ордера
            await exchange_manager.cancel_all_orders()
            
            self.running = False
            self.paused = True  # Блокируем любую торговлю
            emergency_stop.activate(reason)
            
            await self.send_telegram(
                f"🚨🚨🚨 EMERGENCY STOP!\n\n"
                f"Причина: {reason}\n\n"
                f"✅ Все позиции закрыты\n"
                f"✅ Все ордера отменены\n"
                f"✅ Бот остановлен\n\n"
                f"⚠️ ТРЕБУЕТСЯ РУЧНОЙ ПЕРЕЗАПУСК!"
            )
            
            # КРИТИЧНО: Полностью останавливаем бота
            await self.shutdown()
            
            # Завершаем процесс
            import sys
            sys.exit(1)
            
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
            close_side = "sell" if side == "buy" else "buy"
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
            
            # 7. Удаляем из списка открытых
            self.open_positions = [p for p in self.open_positions if p['symbol'] != symbol]
            
            # 8. Уведомление
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
            "🤖 БОТ V2.0\n\n"
            "Команды:\n"
            "/status - статус\n"
            "/pause - пауза\n"
            "/resume - возобновить\n"
            "/stop - остановить"
        )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status"""
        balance = await exchange_manager.get_balance()
        
        status_text = (
            f"📊 СТАТУС БОТА V2.0\n\n"
            f"🟢 {'Работает' if self.running and not self.paused else '⏸️ Пауза' if self.paused else '🔴 Остановлен'}\n"
            f"💰 Баланс: ${balance:.2f}\n"
            f"📊 Позиций: {len(self.open_positions)}/{Config.MAX_POSITIONS}\n"
            f"🧪 Тест: {'ДА' if Config.TEST_MODE else 'НЕТ'}\n"
            f"📈 P&L день: ${-risk_manager.daily_loss:.2f}\n"
            f"🔢 Сделок: {risk_manager.trades_today}\n"
            f"⚠️ Ошибок: {self.bot_errors_count}"
        )
        
        await update.message.reply_text(status_text)
    
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
    
    async def _check_existing_positions_on_startup(self):
        """
        КРИТИЧНО: Проверка существующих позиций при запуске бота
        Если есть позиции без SL/TP - создаем их!
        """
        try:
            logger.info("🔍 Проверка существующих позиций при запуске...")
            
            positions = await exchange_manager.fetch_positions()
            
            if not positions:
                logger.info("✅ Нет открытых позиций")
                return
            
            logger.warning(f"⚠️ Найдено {len(positions)} открытых позиций при старте!")
            
            for position in positions:
                symbol = position['symbol']
                entry_price = float(position['entryPrice'])
                amount = float(position['contracts'])
                side = position['side'].lower()
                
                # Проверяем наличие SL на бирже
                stop_loss = position.get('stopLoss') or position.get('info', {}).get('stopLoss')
                
                # Проверяем и создаем SL
                if not stop_loss or stop_loss == "" or stop_loss == "0":
                    logger.critical(f"🚨 {symbol}: ПОЗИЦИЯ БЕЗ STOP LOSS!")
                    
                    # Создаем SL немедленно
                    sl_price = entry_price * (0.9 if side == 'long' else 1.1)
                    sl_order = await exchange_manager.create_stop_market_order(
                        symbol=symbol,
                        side='sell' if side == 'long' else 'buy',
                        amount=amount,
                        stop_price=round(sl_price, 4)
                    )
                    
                    if sl_order and sl_order.get('status') == 'set':
                        logger.info(f"✅ {symbol}: SL установлен @ ${sl_price:.4f}")
                    else:
                        logger.error(f"❌ {symbol}: НЕ УДАЛОСЬ УСТАНОВИТЬ SL!")
                else:
                    logger.info(f"✅ {symbol}: SL уже установлен @ ${stop_loss}")
                
                # Проверяем и создаем многоуровневый TP
                orders = await exchange_manager.fetch_open_orders(symbol)
                tp_orders = [o for o in orders if o.get('type') == 'limit' and o.get('reduceOnly')]
                
                if not tp_orders:
                    logger.warning(f"⚠️ {symbol}: НЕТ TAKE PROFIT ОРДЕРОВ!")
                    
                    # Создаем 5 уровней TP
                    tp_levels = [0.006, 0.013, 0.019, 0.026, 0.032]  # +0.6%, +1.3%, +1.9%, +2.6%, +3.2%
                    tp_amounts = [amount * 0.2] * 5  # По 20% на каждый уровень
                    
                    created_tp = []
                    for i, (level, tp_amount) in enumerate(zip(tp_levels, tp_amounts), 1):
                        tp_price = entry_price * (1 + level if side == 'long' else 1 - level)
                        
                        try:
                            tp_order = await exchange_manager.create_limit_order(
                                symbol=symbol,
                                side='sell' if side == 'long' else 'buy',
                                amount=tp_amount,
                                price=round(tp_price, 4)
                            )
                            
                            if tp_order and tp_order.get('id'):
                                created_tp.append(f"🎯 TP{i}: ${tp_price:.4f}")
                                logger.info(f"✅ {symbol}: TP{i} установлен @ ${tp_price:.4f}")
                        except Exception as e:
                            logger.error(f"❌ {symbol}: Ошибка создания TP{i}: {e}")
                    
                    if created_tp:
                        await self.send_telegram(
                            f"🛡️ *AUTO-FIX STARTUP*\n\n"
                            f"Позиция {symbol}:\n"
                            f"✅ Установлено {len(created_tp)} TP уровней\n" +
                            "\n".join(created_tp)
                        )
                else:
                    logger.info(f"✅ {symbol}: TP ордера уже есть ({len(tp_orders)} шт)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки позиций при старте: {e}")


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

