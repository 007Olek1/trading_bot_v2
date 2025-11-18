"""
📱 TELEGRAM HANDLER - Обработчик команд и уведомлений
"""

import asyncio
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import config


class TelegramHandler:
    """Обработчик Telegram команд и уведомлений"""
    
    def __init__(self, token: str, chat_id: str):
        """Инициализация"""
        self.token = token
        self.chat_id = chat_id
        self.bot_instance = None  # Ссылка на основной бот
        self.polling_task = None
        
        if token and chat_id:
            # Создаём приложение
            self.app = Application.builder().token(token).build()
            self._setup_handlers()
        else:
            self.app = None
    
    def set_bot_instance(self, bot):
        """Установка ссылки на основной бот"""
        self.bot_instance = bot
    
    def _setup_handlers(self):
        """Настройка обработчиков команд"""
        # Команды
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("history", self.cmd_history))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("regime", self.cmd_regime))
        self.app.add_handler(CommandHandler("adaptive", self.cmd_adaptive_stats))
        self.app.add_handler(CommandHandler("set_regime", self.cmd_set_regime))
        
        # Callback кнопки
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
    
    def start(self):
        """Запуск Telegram бота для обработки команд в отдельном потоке"""
        if not self.app:
            return
        
        def run_bot_thread():
            """Запуск бота в отдельном потоке"""
            try:
                print("✅ Telegram бот запускается в отдельном потоке...")
                
                # Создаём новый event loop для этого потока
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Инициализируем и запускаем бота вручную
                async def run():
                    async with self.app:
                        # Запускаем updater
                        await self.app.start()
                        await self.app.updater.start_polling(
                            allowed_updates=Update.ALL_TYPES,
                            drop_pending_updates=True
                        )
                        print("✅ Telegram бот слушает команды!")
                        
                        # Держим бота активным
                        while True:
                            await asyncio.sleep(1)
                
                loop.run_until_complete(run())
            except Exception as e:
                print(f"❌ Ошибка Telegram бота: {e}")
                import traceback
                traceback.print_exc()
        
        # Запускаем в отдельном потоке
        self.bot_thread = threading.Thread(target=run_bot_thread, daemon=True)
        self.bot_thread.start()
        print("✅ Telegram бот запущен в фоновом потоке")
    
    def stop(self):
        """Остановка Telegram бота"""
        if self.app:
            try:
                # Останавливаем приложение
                if hasattr(self.app, 'updater') and self.app.updater:
                    self.app.updater.stop()
                print("✅ Telegram бот остановлен")
            except Exception as e:
                print(f"❌ Ошибка остановки Telegram бота: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # КОМАНДЫ
    # ═══════════════════════════════════════════════════════════════════
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🟢 /start - Стартовое сообщение"""
        message = (
            "🚀 <b>TRADING BOT V4.0 MTF</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📊 <b>MTF Таймфреймы</b>\n"
            f"{'  ⏩  '.join(config.TIMEFRAMES.keys())}\n\n"
            "🎯 <b>Стратегии</b>\n"
            "💹 Тренд + Объём + Bollinger\n"
            "🎭 Детектор манипуляций\n"
            "🌍 Анализ глобального тренда\n\n"
            "💡 Используйте /help для списка команд"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("💰 Баланс", callback_data="balance")
            ],
            [
                InlineKeyboardButton("📈 Позиции", callback_data="positions"),
                InlineKeyboardButton("📊 Статистика", callback_data="stats")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📝 /help - Список команд"""
        message = (
            "📝 <b>СПИСОК КОМАНД</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>Основные:</b>\n"
            "🟢 /start - Стартовое сообщение\n"
            "📝 /help - Список команд\n"
            "📊 /status - Статус бота\n"
            "💰 /balance - Баланс\n"
            "📈 /positions - Открытые позиции\n"
            "📜 /history - История сделок\n"
            "⛔ /stop - Остановить торговлю\n"
            "▶️ /resume - Возобновить\n"
            "📊 /stats - Статистика\n\n"
            "<b>🧠 Адаптивная система:</b>\n"
            "🎯 /regime - Текущий режим рынка\n"
            "📊 /adaptive - Статистика по режимам\n"
            "⚙️ /set_regime - Установить режим вручную\n"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 /status - Статус бота"""
        if not self.bot_instance:
            await update.message.reply_text("⚠️ Бот не инициализирован")
            return
        
        status = "🟢 Активен" if self.bot_instance.active else "🔴 Остановлен"
        positions_count = len(self.bot_instance.open_positions)
        
        # Получаем баланс
        from utils import get_balance_info
        balance = get_balance_info(self.bot_instance.client)
        
        # Информация о сканировании
        scan_mode = "🔍 Динамическое" if config.USE_DYNAMIC_SCANNER else "📋 Статический"
        watchlist_count = len(self.bot_instance.current_watchlist) if self.bot_instance.current_watchlist else len(config.WATCHLIST)
        
        message = (
            f"📊 <b>СТАТУС БОТА</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🤖 Статус: {status}\n"
            f"📌 Позиций: {positions_count}/{config.MAX_CONCURRENT_POSITIONS}\n"
            f"🔄 Циклов: {self.bot_instance.cycle_count}\n\n"
        )
        
        if balance:
            message += (
                f"💰 <b>Баланс</b>\n"
                f"💵 Всего: ${balance['total']:.2f}\n"
                f"💸 Свободно: ${balance['available']:.2f}\n"
                f"🔒 В позициях: ${balance['used']:.2f}\n\n"
            )
        
        message += (
            f"📈 <b>Торговля</b>\n"
            f"⚡ Размер сделки: ${config.POSITION_SIZE_USD} x{config.LEVERAGE}\n"
            f"💰 Объём: ${config.POSITION_SIZE_USD * config.LEVERAGE}\n"
            f"📌 Макс. позиций: {config.MAX_CONCURRENT_POSITIONS}\n\n"
            f"🔍 <b>Сканирование</b>\n"
            f"{scan_mode}\n"
            f"📊 Монет в watchlist: {watchlist_count}\n\n"
            f"⏱️ Анализ: каждые {config.ANALYSIS_INTERVAL_SECONDS//60} мин\n"
            f"📊 Мониторинг: каждую минуту\n\n"
            f"💡 Команды: /help для списка\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """💰 /balance - Баланс"""
        if not self.bot_instance:
            await update.message.reply_text("⚠️ Бот не инициализирован")
            return
        
        from utils import get_balance_info
        balance = get_balance_info(self.bot_instance.client)
        
        if not balance:
            await update.message.reply_text("❌ Не удалось получить баланс")
            return
        
        positions_count = len(self.bot_instance.open_positions)
        
        message = (
            f"💰 <b>БАЛАНС</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💵 Всего: ${balance['total']:.2f}\n"
            f"💸 Свободно: ${balance['available']:.2f}\n"
            f"🔒 В позициях: ${balance['used']:.2f}\n\n"
            f"📈 <b>Торговля</b>\n"
            f"⚡ Сделка: ${config.POSITION_SIZE_USD} x{config.LEVERAGE} = ${config.POSITION_SIZE_USD * config.LEVERAGE}\n"
            f"📌 Позиции: {positions_count}/{config.MAX_CONCURRENT_POSITIONS}\n"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📈 /positions - Открытые позиции"""
        if not self.bot_instance:
            await update.message.reply_text("⚠️ Бот не инициализирован")
            return
        
        positions = self.bot_instance.open_positions
        
        if not positions:
            await update.message.reply_text("📭 Нет открытых позиций")
            return
        
        message = f"📈 <b>ОТКРЫТЫЕ ПОЗИЦИИ</b>\n\n"
        
        for symbol, pos in positions.items():
            direction_emoji = "🟢" if pos['direction'] == 'LONG' else "🔴"
            
            # Получаем текущую цену
            mtf_data = self.bot_instance.get_mtf_data(symbol)
            if mtf_data and config.PRIMARY_TIMEFRAME in mtf_data:
                current_price = mtf_data[config.PRIMARY_TIMEFRAME]['close'].iloc[-1]
                
                # Рассчитываем PnL
                from utils import calculate_pnl, format_duration, format_datetime
                pnl_usd, pnl_percent = calculate_pnl(
                    pos['entry_price'],
                    current_price,
                    pos['direction'],
                    pos['quantity']
                )
                
                pnl_emoji = "💚" if pnl_usd > 0 else "❤️" if pnl_usd < 0 else "💛"
                
                # Время удержания
                duration = (datetime.now(timezone.utc) - pos['open_time']).total_seconds()
                duration_str = format_duration(duration)
                open_time_str = format_datetime(pos['open_time'])
                
                # Рассчитываем максимальную прибыль
                max_profit_percent = pos.get('max_profit_percent', pnl_percent if pnl_percent > 0 else 0)
                
                # Формируем сообщение
                msg_parts = [
                    f"{direction_emoji} <b>{symbol}</b> {pos['direction']}\n\n",
                    f"💰 <b>ЦЕНЫ</b>\n",
                    f"🟢 Вход: ${pos['entry_price']:.6f}\n",
                    f"📊 Текущая: ${current_price:.6f}\n",
                ]
                
                # Stop Loss и Take Profit
                if 'sl_price' in pos and pos['sl_price']:
                    msg_parts.append(f"🛑 Stop Loss: ${pos['sl_price']:.6f}\n")
                
                # Рассчитываем целевую прибыль (ROI +10% при плече 10x = +1% от цены)
                leverage = pos.get('leverage', config.LEVERAGE)
                target_roi = 10.0  # Целевой ROI 10%
                price_change_percent = target_roi / leverage  # При 10x: 10% / 10 = 1%
                
                if pos['direction'] == 'LONG':
                    tp_price = pos['entry_price'] * (1 + price_change_percent / 100)
                else:
                    tp_price = pos['entry_price'] * (1 - price_change_percent / 100)
                msg_parts.append(f"🎯 Take Profit: ${tp_price:.6f} (+{target_roi:.0f}% ROI)\n")
                
                msg_parts.append(f"\n{pnl_emoji} <b>PnL</b>\n")
                msg_parts.append(f"💵 Текущий: ${pnl_usd:.2f} ({pnl_percent:+.2f}%)\n")
                
                if max_profit_percent > 0:
                    msg_parts.append(f"🚀 Максимум: +{max_profit_percent:.2f}%\n")
                
                msg_parts.append(f"\n📅 Открыта: {open_time_str}\n")
                msg_parts.append(f"⏱️ Удержание: {duration_str}\n")
                
                # Добавляем уверенность только если есть
                if 'confidence' in pos:
                    msg_parts.append(f"🎯 Уверенность: {pos['confidence']:.1%}\n")
                
                # Trailing SL статус
                if pos.get('trailing_sl_active'):
                    msg_parts.append(f"🔄 Trailing SL: Активен\n")
                
                # Флаг восстановленной позиции
                if pos.get('restored'):
                    msg_parts.append(f"🔄 Восстановлена при старте\n")
                
                msg_parts.append(f"\n")
                message += ''.join(msg_parts)
            else:
                message += (
                    f"{direction_emoji} <b>{symbol}</b> {pos['direction']}\n"
                    f"💰 Вход: ${pos['entry_price']:.6f}\n"
                    f"⚠️ Не удалось получить текущую цену\n\n"
                )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📜 /history - История сделок"""
        message = (
            "📜 <b>ИСТОРИЯ СДЕЛОК</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🚧 В разработке...\n"
            "История сделок будет доступна в следующей версии"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """⛔ /stop - Остановить торговлю"""
        if not self.bot_instance:
            await update.message.reply_text("⚠️ Бот не инициализирован")
            return
        
        self.bot_instance.active = False
        
        message = (
            "⛔ <b>ТОРГОВЛЯ ОСТАНОВЛЕНА</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔴 Бот больше не будет открывать новые позиции\n"
            "📌 Существующие позиции продолжают отслеживаться\n\n"
            "▶️ Используйте /resume для возобновления"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """▶️ /resume - Возобновить торговлю"""
        if not self.bot_instance:
            await update.message.reply_text("⚠️ Бот не инициализирован")
            return
        
        self.bot_instance.active = True
        
        message = (
            "▶️ <b>ТОРГОВЛЯ ВОЗОБНОВЛЕНА</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🟢 Бот снова активен и будет искать сигналы\n"
            "📊 Анализ рынка продолжается\n"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 /stats - Статистика"""
        if not self.bot_instance:
            await update.message.reply_text("⚠️ Бот не инициализирован")
            return
        
        # Читаем логи сделок
        from pathlib import Path
        import json
        
        trades_file = Path(config.LOG_DIR) / "trades.json"
        total_trades = 0
        winning_trades = 0
        total_pnl = 0.0
        
        if trades_file.exists():
            try:
                with open(trades_file, 'r') as f:
                    for line in f:
                        try:
                            trade = json.loads(line)
                            if trade.get('status') == 'CLOSED':
                                total_trades += 1
                                pnl = trade.get('pnl_percent', 0)
                                if pnl > 0:
                                    winning_trades += 1
                                total_pnl += pnl
                        except:
                            continue
            except:
                pass
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0
        
        # Статистика сканирования
        cycle_count = getattr(self.bot_instance, 'cycle_count', 0)
        scan_mode = "🔍 Динамическое" if config.USE_DYNAMIC_SCANNER else "📋 Статический"
        watchlist_count = len(self.bot_instance.current_watchlist) if self.bot_instance.current_watchlist else len(config.WATCHLIST)
        positions_count = len(self.bot_instance.open_positions)
        
        message = (
            "📊 <b>СТАТИСТИКА БОТА</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>🔍 СКАНИРОВАНИЕ</b>\n"
            f"🔄 Циклов: {cycle_count}\n"
            f"📊 Режим: {scan_mode}\n"
            f"📊 Монет в watchlist: {watchlist_count}\n"
            f"⏱️ Интервал: {config.ANALYSIS_INTERVAL_SECONDS//60} мин\n\n"
            "<b>📈 ТОРГОВЛЯ</b>\n"
            f"📌 Открытых позиций: {positions_count}/{config.MAX_CONCURRENT_POSITIONS}\n"
            f"💼 Всего сделок: {total_trades}\n"
            f"✅ Прибыльных: {winning_trades}\n"
            f"❌ Убыточных: {total_trades - winning_trades}\n"
            f"📊 Win Rate: {win_rate:.1f}%\n"
            f"💰 Средний PnL: {avg_pnl:+.2f}%\n\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатий на кнопки"""
        query = update.callback_query
        await query.answer()
        
        # Создаём новый update объект с message из callback query
        # Это необходимо для корректной работы команд
        class CallbackUpdate:
            def __init__(self, message):
                self.message = message
        
        callback_update = CallbackUpdate(query.message)
        
        # Вызываем соответствующую команду
        if query.data == "status":
            await self.cmd_status(callback_update, context)
        elif query.data == "balance":
            await self.cmd_balance(callback_update, context)
        elif query.data == "positions":
            await self.cmd_positions(callback_update, context)
        elif query.data == "stats":
            await self.cmd_stats(callback_update, context)
    
    # ═══════════════════════════════════════════════════════════════════
    # УВЕДОМЛЕНИЯ
    # ═══════════════════════════════════════════════════════════════════
    
    async def send_message(self, text: str, parse_mode: str = 'HTML'):
        """Отправка сообщения"""
        if not self.token or not self.chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        print(f"Ошибка отправки Telegram: HTTP {response.status}")
        except Exception as e:
            print(f"Ошибка отправки Telegram сообщения: {e}")
    
    async def send_startup_message(self):
        """Стартовое сообщение при запуске бота"""
        message = (
            "🚀 <b>TRADING BOT V4.0 MTF ЗАПУЩЕН</b>\n\n"
            f"📊 MTF Таймфреймы: {' ⏩ '.join(config.TIMEFRAMES.keys())}\n\n"
            "🎯 <b>Стратегии</b>\n"
            "💹 Тренд + Объём + Bollinger\n"
            "🎭 Детектор манипуляций\n"
            "🌍 Анализ глобального тренда\n\n"
            f"🎯 Минимальная цель: +{config.MIN_PROFIT_TARGET_PERCENT}% (R/R оптимизирован)\n"
            f"🔄 Trailing SL: после +{config.TRAILING_SL_ACTIVATION_PERCENT}% (откат {config.TRAILING_SL_CALLBACK_PERCENT}%)\n"
            f"📊 Уверенность: мин. {config.SIGNAL_THRESHOLDS['min_confidence']:.0%} (усилена фильтрация)\n"
            f"📈 MTF: минимум {config.MIN_TIMEFRAME_ALIGNMENT}/{len(config.TIMEFRAMES)} таймфреймов\n"
            f"🛑 SL макс: -${config.STOP_LOSS_MAX_USD}\n\n"
            f"⚡ Сделка: ${config.POSITION_SIZE_USD} x{config.LEVERAGE} = ${config.POSITION_SIZE_USD * config.LEVERAGE}\n"
            f"📌 Макс. позиций: {config.MAX_CONCURRENT_POSITIONS}\n\n"
            f"⏱️ Анализ: каждые {config.ANALYSIS_INTERVAL_SECONDS//60} мин\n"
            f"📊 Мониторинг: каждую минуту\n"
            f"💡 Команды: /help для списка\n"
            f"⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        )
        
        await self.send_message(message)
    
    async def send_candidate_alert(self, symbol: str, direction: str, confidence: float, tf_aligned: int, total_tf: int, price: float):
        """Уведомление о кандидате на вход"""
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        message = (
            f"🔍 <b>КАНДИДАТ НА ВХОД</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{direction_emoji} <b>{symbol}</b> → {direction}\n"
            f"💰 Цена: ${price:.6f}\n"
            f"🎯 Уверенность: {confidence:.1%}\n"
            f"📊 Таймфреймов: {tf_aligned}/{total_tf}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"💡 Ожидаю подтверждения для входа..."
        )
        
        await self.send_message(message)
    
    async def send_position_opened(self, symbol: str, direction: str, entry_price: float, quantity: float, confidence: float):
        """Уведомление об открытии позиции"""
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        # Рассчитываем минимальную цель (ROI +10% при плече 10x = +1% от цены)
        target_roi = 10.0  # Целевой ROI 10%
        price_change_percent = target_roi / config.LEVERAGE  # При 10x: 10% / 10 = 1%
        
        if direction == "LONG":
            target_price = entry_price * (1 + price_change_percent / 100)
        else:
            target_price = entry_price * (1 - price_change_percent / 100)
        
        # Рассчитываем прибыль в USD при достижении цели (ROI 10%)
        position_value = config.POSITION_SIZE_USD * config.LEVERAGE
        target_profit_usd = config.POSITION_SIZE_USD * (target_roi / 100)
        
        message = (
            f"{direction_emoji} <b>ПОЗИЦИЯ ОТКРЫТА</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💎 Монета: <b>{symbol}</b>\n"
            f"📊 Направление: {direction}\n"
            f"💰 Цена входа: ${entry_price:.6f}\n"
            f"📦 Количество: {quantity:.4f}\n"
            f"🎯 Уверенность: {confidence:.1%}\n"
            f"⚡ Размер: ${config.POSITION_SIZE_USD} x{config.LEVERAGE} = ${position_value}\n\n"
            f"🎯 <b>Стратегия:</b>\n"
            f"  🎯 Минимальная цель: +{target_roi:.0f}% ROI (+${target_profit_usd:.2f})\n"
            f"  💰 Цена цели: ${target_price:.6f}\n"
            f"  🔄 Trailing SL: активен сразу (откат {config.TRAILING_SL_CALLBACK_PERCENT}%)\n"
            f"  📈 Держим до разворота (НЕ закрываем по цели!)\n"
            f"  📉 Trailing SL закроет при развороте\n\n"
            f"🛑 Начальный SL: -${config.STOP_LOSS_MAX_USD} макс\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        )
        
        await self.send_message(message)
    
    async def send_position_closed(self, symbol: str, direction: str, entry_price: float, 
                                   exit_price: float, pnl_usd: float, pnl_percent: float, 
                                   duration: str, reason: str):
        """Уведомление о закрытии позиции с полной информацией"""
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        # Определяем результат сделки
        if pnl_usd > 0:
            result_emoji = "💰🚀"  # Прибыль
            result_text = "ПРИБЫЛЬ"
            pnl_emoji = "💚"
        elif pnl_usd < 0:
            result_emoji = "📉"  # Убыток
            result_text = "УБЫТОК"
            pnl_emoji = "❤️"
        else:
            result_emoji = "🔄"  # Безубыток
            result_text = "БЕЗУБЫТОК"
            pnl_emoji = "💛"
        
        # Причина закрытия
        reason_map = {
            "trailing_sl": "🔄 Trailing Stop Loss",
            "stop_loss": "🛑 Stop Loss",
            "max_duration": "⏰ Макс. время удержания",
            "manual": "🔧 Ручное закрытие",
            "signal_reversal": "🔄 Разворот сигнала"
        }
        reason_text = reason_map.get(reason, reason)
        
        message = (
            f"{result_emoji} <b>ПОЗИЦИЯ ЗАКРЫТА - {result_text}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{direction_emoji} <b>{symbol}</b> {direction}\n\n"
            f"💰 <b>РЕЗУЛЬТАТ</b>\n"
            f"{pnl_emoji} PnL: <b>${pnl_usd:.2f}</b> ({pnl_percent:+.2f}%)\n\n"
            f"📊 <b>ДЕТАЛИ</b>\n"
            f"🟢 Вход: ${entry_price:.6f}\n"
            f"🔴 Выход: ${exit_price:.6f}\n"
            f"⏱️ Время: {duration}\n"
            f"📝 Причина: {reason_text}\n\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        )
        
        await self.send_message(message)
    
    async def send_tp_hit(self, symbol: str, tp_level: int, price: float, pnl_usd: float, pnl_percent: float):
        """Уведомление о достижении TP уровня"""
        message = (
            f"🎯 <b>TP{tp_level} ДОСТИГНУТ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💎 Монета: <b>{symbol}</b>\n"
            f"💰 Цена: ${price:.6f}\n"
            f"💚 Прибыль: ${pnl_usd:.2f} (+{pnl_percent:.2f}%)\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        )
        
        await self.send_message(message)
    
    async def send_error(self, error_msg: str):
        """Уведомление об ошибке"""
        message = (
            f"❌ <b>ОШИБКА</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{error_msg}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        )
        
        await self.send_message(message)
    
    # ═══════════════════════════════════════════════════════════════════
    # КОМАНДЫ АДАПТИВНОЙ СИСТЕМЫ
    # ═══════════════════════════════════════════════════════════════════
    
    async def cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🎯 /regime - Текущий режим рынка"""
        if not self.bot_instance or not hasattr(self.bot_instance, 'regime_detector'):
            await update.message.reply_text("⚠️ Адаптивная система не инициализирована")
            return
        
        try:
            regime, confidence, details = self.bot_instance.regime_detector.detect_regime()
            stats = self.bot_instance.regime_detector.get_regime_stats()
            
            regime_names = {
                'bull': '🟢 БЫЧИЙ',
                'bear': '🔴 МЕДВЕЖИЙ',
                'sideways': '🟡 БОКОВОЙ',
                'volatile': '⚡ ВОЛАТИЛЬНЫЙ'
            }
            
            message = (
                f"🎯 <b>РЕЖИМ РЫНКА</b>\n\n"
                f"Текущий: <b>{regime_names.get(regime.value, regime.value.upper())}</b>\n"
                f"Уверенность: {confidence:.1%}\n\n"
                f"<b>📊 Детали:</b>\n"
                f"Тренд: {details.get('trend_score', 0):.2f}\n"
                f"Волатильность: {details.get('volatility_score', 0):.2f}\n"
                f"Объём: {details.get('volume_score', 0):.2f}\n"
                f"Моментум: {details.get('momentum_score', 0):.2f}\n"
            )
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_adaptive_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 /adaptive - Статистика адаптивной системы"""
        if not self.bot_instance or not hasattr(self.bot_instance, 'adaptive_config'):
            await update.message.reply_text("⚠️ Адаптивная система не инициализирована")
            return
        
        try:
            all_stats = self.bot_instance.adaptive_config.get_all_regimes_stats()
            
            message = "<b>📊 СТАТИСТИКА АДАПТИВНОЙ СИСТЕМЫ</b>\n\n"
            
            for regime_key, data in all_stats.items():
                perf = data.get('performance', {})
                if perf:
                    win_rate = perf.get('win_rate', 0) * 100
                    total_pnl = perf.get('total_pnl', 0)
                    total_trades = perf.get('total_trades', 0)
                    
                    emoji = {'bull': '🟢', 'bear': '🔴', 'sideways': '🟡', 'volatile': '⚡'}.get(regime_key, '⚪')
                    
                    message += (
                        f"{emoji} <b>{data['name']}</b>\n"
                        f"Сделок: {total_trades}\n"
                        f"Win Rate: {win_rate:.1f}%\n"
                        f"PnL: ${total_pnl:.2f}\n\n"
                    )
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_set_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """⚙️ /set_regime [bull|bear|sideways|volatile] - Ручное переопределение режима"""
        if not self.bot_instance or not hasattr(self.bot_instance, 'regime_detector'):
            await update.message.reply_text("⚠️ Адаптивная система не инициализирована")
            return
        
        if not context.args:
            await update.message.reply_text(
                "Использование: /set_regime [bull|bear|sideways|volatile]"
            )
            return
        
        regime_str = context.args[0].lower()
        
        if regime_str not in ['bull', 'bear', 'sideways', 'volatile']:
            await update.message.reply_text(f"❌ Неизвестный режим: {regime_str}")
            return
        
        try:
            from market_regime_detector import MarketRegime
            
            new_regime = MarketRegime(regime_str)
            self.bot_instance.regime_detector.current_regime = new_regime
            self.bot_instance.regime_detector.regime_confidence = 1.0
            self.bot_instance.regime_detector.regime_changed_at = datetime.now(timezone.utc)
            self.bot_instance.update_market_regime()
            
            await update.message.reply_text(f"✅ Режим изменён на: {regime_str.upper()}", parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
