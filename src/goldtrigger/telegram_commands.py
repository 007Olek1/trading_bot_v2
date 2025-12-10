#!/usr/bin/env python3
"""
Telegram Commands Handler for GoldTrigger Bot
Обработчик команд для управления ботом через Telegram
"""

import logging
from datetime import datetime
from typing import Optional, Set

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger(__name__)


class TelegramCommandsHandler:
    """Обработчик Telegram команд для бота"""
    
    def __init__(self, bot_instance, telegram_token: str, chat_id: str):
        self.bot = bot_instance
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.trading_enabled = True
        self.app = None
        self.allowed_chat_ids: Set[int] = self._parse_chat_ids(chat_id)
        
        logger.info("Telegram Commands Handler инициализирован")
    
    async def setup_commands(self):
        """Настройка команд Telegram"""
        try:
            self.app = Application.builder().token(self.telegram_token).build()
            
            # Регистрация команд
            self.app.add_handler(CommandHandler("start", self.cmd_start))
            self.app.add_handler(CommandHandler("help", self.cmd_help))
            self.app.add_handler(CommandHandler("status", self.cmd_status))
            self.app.add_handler(CommandHandler("balance", self.cmd_balance))
            self.app.add_handler(CommandHandler("positions", self.cmd_positions))
            self.app.add_handler(CommandHandler("history", self.cmd_history))
            self.app.add_handler(CommandHandler("stop", self.cmd_stop))
            self.app.add_handler(CommandHandler("resume", self.cmd_resume))
            self.app.add_handler(CommandHandler("stats", self.cmd_stats))
            self.app.add_handler(CommandHandler("ping", self.cmd_ping))
            
            # Запуск в фоновом режиме
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("Telegram команды настроены и активны")
        except Exception as e:
            logger.error(f"Ошибка настройки Telegram команд: {e}")

    def _parse_chat_ids(self, chat_id: Optional[str]) -> Set[int]:
        ids: Set[int] = set()
        if not chat_id:
            return ids
        for part in str(chat_id).replace(',', ' ').split():
            try:
                ids.add(int(part))
            except ValueError:
                continue
        return ids

    async def _check_authorized(self, update: Update) -> bool:
        if not self.allowed_chat_ids:
            return True
        chat = update.effective_chat
        if chat and chat.id in self.allowed_chat_ids:
            return True
        logger.warning("Telegram команда из неавторизованного чата: %s", chat.id if chat else 'unknown')
        await self._safe_reply(update, "⛔ Доступ запрещён. Добавьте этот chat_id в TELEGRAM_CHAT_ID")
        return False

    async def _safe_reply(self, update: Update, message: str):
        if update.message:
            await update.message.reply_text(message, parse_mode='HTML')
        elif update.effective_chat:
            await update.effective_chat.send_message(message, parse_mode='HTML')
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start - стартовое сообщение"""
        if not await self._check_authorized(update):
            return
        logger.info("/start от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
        message = (
            "🟢 <b>GoldTrigger Trend Trader V5</b>\n\n"
            "Добро пожаловать! Бот активен.\n\n"
            "🤖 <b>Система:</b>\n"
            "• Умный селектор: 145 монет\n"
            "• GoldTrigger логика: ✅\n"
            "• Disco57 (DiscoRL): ✅\n\n"
            "📈 <b>ТРЕНДОВАЯ СТРАТЕГИЯ:</b>\n"
            "• Плечо: 25x\n"
            "• SL: -1.5% от входа\n"
            "• TP: Trailing (без фикс.)\n"
            "• Trailing активация: +2%\n"
            "• Trailing step: 1%\n\n"
            "🎯 Держим позицию пока тренд идет!\n\n"
            "Используйте /help для списка команд"
        )
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help - список команд"""
        if not await self._check_authorized(update):
            return
        logger.info("/help от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
        message = (
            "📝 <b>Доступные команды:</b>\n\n"
            "/start - 🟢 Стартовое сообщение\n"
            "/help - 📝 Список команд\n"
            "/status - 📊 Статус бота\n"
            "/balance - 💰 Баланс\n"
            "/positions - 📈 Открытые позиции\n"
            "/history - 📜 История сделок\n"
            "/stop - ⛔ Остановить торговлю\n"
            "/resume - ▶️ Возобновить торговлю\n"
            "/stats - 📊 Статистика\n"
        )
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status - статус бота"""
        try:
            if not await self._check_authorized(update):
                return
            logger.info("/status от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
            status_emoji = "🟢" if self.trading_enabled else "🔴"
            status_text = "Активен" if self.trading_enabled else "Остановлен"
            
            open_positions = len(self.bot.positions)
            daily_pnl = self.bot.daily_pnl
            daily_trades = self.bot.daily_trades
            
            # Disco57 статус
            disco_status = "❌"
            if hasattr(self.bot, 'disco57') and self.bot.disco57:
                disco_wr = self.bot.disco57.get_win_rate()
                disco_status = f"✅ {disco_wr:.0f}%"
            
            # Нереализованный PnL
            unrealized_pnl = sum(p.current_pnl for p in self.bot.positions.values())
            
            message = (
                f"📊 <b>Статус бота</b>\n\n"
                f"{status_emoji} <b>Торговля:</b> {status_text}\n"
                f"🤖 <b>Disco57:</b> {disco_status}\n\n"
                f"📈 <b>Позиции:</b> {open_positions}/3\n"
                f"💵 <b>Нереализ. PnL:</b> ${unrealized_pnl:+.2f}\n"
                f"💰 <b>Дневной PnL:</b> ${daily_pnl:+.2f}\n"
                f"🔢 <b>Сделок:</b> {daily_trades}\n\n"
                f"⏰ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
            )
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /ping - проверка доступности бота"""
        if not await self._check_authorized(update):
            return
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        await update.message.reply_text(f"🏓 Pong!\n⏰ {ts}", parse_mode='HTML')
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /balance - баланс"""
        try:
            if not await self._check_authorized(update):
                return
            logger.info("/balance от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
            balance_info = await self.bot.api.get_account_balance()
            
            if balance_info and 'total' in balance_info:
                usdt_balance = balance_info['total'].get('USDT', 0)
                free_balance = balance_info['free'].get('USDT', 0)
                used_balance = balance_info['used'].get('USDT', 0)
                
                # Нереализованный PnL
                unrealized_pnl = sum(p.current_pnl for p in self.bot.positions.values())
                
                # Эффективный баланс
                effective_balance = usdt_balance + unrealized_pnl
                
                message = (
                    f"💰 <b>Баланс аккаунта</b>\n\n"
                    f"💵 <b>Всего USDT:</b> ${usdt_balance:.2f}\n"
                    f"✅ <b>Доступно:</b> ${free_balance:.2f}\n"
                    f"🔒 <b>В маржe:</b> ${used_balance:.2f}\n\n"
                    f"📊 <b>Нереализ. PnL:</b> ${unrealized_pnl:+.2f}\n"
                    f"💎 <b>Эффективный:</b> ${effective_balance:.2f}\n\n"
                    f"📈 <b>Дневной PnL:</b> ${self.bot.daily_pnl:+.2f}"
                )
            else:
                message = "❌ Не удалось получить баланс"
            
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения баланса: {e}")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /positions - открытые позиции"""
        try:
            if not await self._check_authorized(update):
                return
            logger.info("/positions от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
            if not self.bot.positions:
                message = (
                    "📭 <b>Нет открытых позиций</b>\n\n"
                    "🔍 Бот сканирует 145 монет...\n"
                    "⏳ Ожидание сильного сигнала"
                )
            else:
                total_pnl = sum(p.current_pnl for p in self.bot.positions.values())
                pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
                
                message = f"📈 <b>Открытые позиции ({len(self.bot.positions)}/3):</b>\n"
                message += f"{pnl_emoji} Общий PnL: ${total_pnl:+.2f}\n"
                message += "─" * 20 + "\n\n"
                
                for symbol, pos in self.bot.positions.items():
                    side_emoji = "🟢" if pos.side == 'long' else "🔴"
                    trailing_status = "🔄 TRAILING ACTIVE!" if pos.trailing_active else "⏳ Ждем +2%"
                    
                    # Расчет % прибыли
                    pnl_pct = pos.current_pnl / 25 * 100 if pos.current_pnl else 0
                    
                    # Время в позиции
                    import time
                    duration_sec = time.time() - pos.entry_time
                    duration_min = int(duration_sec // 60)
                    
                    # Короткое имя символа
                    short_symbol = symbol.replace('/USDT:USDT', '').replace('USDT', '')
                    
                    message += (
                        f"{side_emoji} <b>{short_symbol}</b> {pos.side.upper()} 25x\n"
                        f"   📍 Вход: ${pos.entry_price:.6f}\n"
                        f"   🛡 SL: ${pos.sl_price:.6f}\n"
                        f"   💰 PnL: <b>${pos.current_pnl:+.2f}</b> ({pnl_pct:+.1f}%)\n"
                        f"   ⏱ Время: {duration_min} мин\n"
                        f"   {trailing_status}\n\n"
                    )
            
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /history - история сделок из БД"""
        try:
            if not await self._check_authorized(update):
                return
            logger.info("/history от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
            message = f"📜 <b>История сделок (24ч)</b>\n\n"
            
            # Получаем историю из БД
            if hasattr(self.bot, 'trade_db') and self.bot.trade_db:
                trades = self.bot.trade_db.get_recent_trades(hours=24, limit=10)
                stats = self.bot.trade_db.get_stats(hours=24)
                
                if trades:
                    wins = stats.get('winning_trades', 0)
                    losses = stats.get('losing_trades', 0)
                    total_pnl = stats.get('total_pnl', 0)
                    win_rate = stats.get('win_rate', 0)
                    
                    message += f"📊 <b>Статистика:</b>\n"
                    message += f"   ✅ Прибыльных: {wins}\n"
                    message += f"   ❌ Убыточных: {losses}\n"
                    message += f"   🎯 Win Rate: {win_rate:.0f}%\n"
                    message += f"   💰 Итого: ${total_pnl:+.2f}\n\n"
                    message += "─" * 20 + "\n"
                    message += "<b>Последние сделки:</b>\n\n"
                    
                    for trade in trades[:5]:  # Последние 5
                        if trade['status'] == 'closed':
                            pnl = trade.get('pnl_usd', 0) or 0
                            emoji = "✅" if pnl > 0 else "❌"
                            reason = trade.get('reason', 'N/A')
                            short_sym = trade['symbol'].replace('/USDT:USDT', '')
                            message += f"{emoji} {short_sym} {trade['side'].upper()} ${pnl:+.2f} [{reason}]\n"
                else:
                    message += "📭 Нет закрытых сделок за 24ч\n"
            else:
                message += f"🔢 Сделок сегодня: {self.bot.daily_trades}\n"
                message += f"💰 Дневной PnL: ${self.bot.daily_pnl:+.2f}\n"
            
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /stop - остановить торговлю"""
        if not await self._check_authorized(update):
            return
        logger.warning("/stop от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
        self.trading_enabled = False
        if hasattr(self.bot, 'trading_enabled'):
            self.bot.trading_enabled = False
        
        open_pos = len(self.bot.positions)
        message = (
            "⛔ <b>Торговля ОСТАНОВЛЕНА</b>\n\n"
            "🚫 Новые позиции НЕ открываются\n"
            f"📈 Открытых позиций: {open_pos}\n\n"
            "⚠️ <i>Существующие позиции продолжат\n"
            "работать по своим SL/Trailing</i>\n\n"
            "▶️ /resume - возобновить торговлю"
        )
        await update.message.reply_text(message, parse_mode='HTML')
        logger.warning("Торговля остановлена через Telegram команду")
    
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /resume - возобновить торговлю"""
        if not await self._check_authorized(update):
            return
        logger.info("/resume от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
        self.trading_enabled = True
        if hasattr(self.bot, 'trading_enabled'):
            self.bot.trading_enabled = True
        
        message = (
            "▶️ <b>Торговля ВОЗОБНОВЛЕНА</b>\n\n"
            "✅ Бот активен\n"
            "🔍 Сканирование 145 монет...\n"
            "📈 Поиск трендовых сигналов\n\n"
            "🎯 Стратегия: Trend Following\n"
            "⚡ Плечо: 25x | SL: -1.5%"
        )
        await update.message.reply_text(message, parse_mode='HTML')
        logger.info("Торговля возобновлена через Telegram команду")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /stats - полная статистика"""
        try:
            if not await self._check_authorized(update):
                return
            logger.info("/stats от chat_id=%s", update.effective_chat.id if update.effective_chat else 'unknown')
            message = "📊 <b>СТАТИСТИКА</b>\n"
            message += "═" * 20 + "\n\n"
            
            # Данные из БД за 24ч
            if hasattr(self.bot, 'trade_db') and self.bot.trade_db:
                stats_24h = self.bot.trade_db.get_stats(hours=24)
                stats_72h = self.bot.trade_db.get_stats(hours=72)
                
                # 24 часа
                message += "📅 <b>За 24 часа:</b>\n"
                message += f"   🔢 Сделок: {stats_24h['total_trades']}\n"
                message += f"   ✅ Прибыльных: {stats_24h['winning_trades']}\n"
                message += f"   ❌ Убыточных: {stats_24h['losing_trades']}\n"
                message += f"   🎯 Win Rate: {stats_24h['win_rate']:.0f}%\n"
                message += f"   💰 PnL: ${stats_24h['total_pnl']:+.2f}\n"
                message += f"   📈 Лучшая: ${stats_24h['best_trade']:+.2f}\n"
                message += f"   📉 Худшая: ${stats_24h['worst_trade']:+.2f}\n\n"
                
                # 72 часа
                message += "📅 <b>За 72 часа:</b>\n"
                message += f"   🔢 Сделок: {stats_72h['total_trades']}\n"
                message += f"   🎯 Win Rate: {stats_72h['win_rate']:.0f}%\n"
                message += f"   💰 PnL: ${stats_72h['total_pnl']:+.2f}\n\n"
            else:
                message += f"🔢 Сделок сегодня: {self.bot.daily_trades}\n"
                message += f"💰 Дневной PnL: ${self.bot.daily_pnl:+.2f}\n\n"
            
            # Disco57
            if hasattr(self.bot, 'disco57') and self.bot.disco57:
                disco_wr = self.bot.disco57.get_win_rate()
                disco_trades = self.bot.disco57.total_trades
                message += "🤖 <b>Disco57 AI:</b>\n"
                message += f"   📚 Обучено на: {disco_trades} сделках\n"
                message += f"   🎯 Win Rate: {disco_wr:.1f}%\n\n"
            
            # Текущее состояние
            open_pos = len(self.bot.positions)
            unrealized = sum(p.current_pnl for p in self.bot.positions.values())
            message += "📈 <b>Сейчас:</b>\n"
            message += f"   🔓 Позиций: {open_pos}/3\n"
            message += f"   💵 Нереализ.: ${unrealized:+.2f}\n\n"
            
            message += f"⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    def is_trading_enabled(self) -> bool:
        """Проверка, включена ли торговля"""
        return self.trading_enabled
    
    async def shutdown(self):
        """Остановка обработчика команд"""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram Commands Handler остановлен")


# Пример использования
if __name__ == '__main__':
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Мок бота для тестирования
    class MockBot:
        def __init__(self):
            self.positions = {}
            self.daily_pnl = 0.0
            self.daily_trades = 0
            
            class MockAPI:
                async def get_account_balance(self):
                    return {
                        'total': {'USDT': 1000.0},
                        'free': {'USDT': 950.0},
                        'used': {'USDT': 50.0}
                    }
            
            self.api = MockAPI()
    
    async def test_commands():
        bot = MockBot()
        handler = TelegramCommandsHandler(
            bot,
            os.getenv('TELEGRAM_TOKEN'),
            os.getenv('TELEGRAM_CHAT_ID')
        )
        
        await handler.setup_commands()
        print("Telegram команды настроены. Бот готов к приему команд.")
        print("Нажмите Ctrl+C для остановки")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await handler.shutdown()
    
    asyncio.run(test_commands())
