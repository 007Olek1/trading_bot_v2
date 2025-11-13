"""Telegram bot application wiring command handlers to trading coordinator."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Callable, Dict, Iterable
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from bybit_bot.core.coordinator import TradingCoordinator
from bybit_bot.telegram.messages import (
    format_balance,
    format_help,
    format_history,
    format_opportunities,
    format_startup_notification,
    format_positions,
    format_start_message,
    format_stats,
    format_status,
    format_trade_close_event,
    format_trade_open_event,
)
from config.settings import settings

logger = logging.getLogger(__name__)

COMMANDS = [
    ("/start", "🟢 Стартовое сообщение"),
    ("/help", "📝 Список команд"),
    ("/status", "📊 Статус бота"),
    ("/balance", "💰 Баланс"),
    ("/positions", "📈 Открытые позиции"),
    ("/history", "📜 История сделок"),
    ("/stop", "⛔ Остановить торговлю"),
    ("/resume", "▶️ Возобновить торговлю"),
    ("/stats", "📊 Статистика"),
    ("/analysis", "🔍 Анализ рынка"),
]


class TelegramBot:
    def __init__(
        self,
        coordinator: TradingCoordinator,
        timezone_label: str = "Warsaw",
        timezone_id: str = "Europe/Warsaw",
    ) -> None:
        self.coordinator = coordinator
        self.timezone_label = timezone_label
        self.timezone = ZoneInfo(timezone_id)
        self.application = ApplicationBuilder().token(settings.telegram_token).build()
        self._task: asyncio.Task | None = None
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("balance", self.cmd_balance))
        self.application.add_handler(CommandHandler("positions", self.cmd_positions))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(CommandHandler("stop", self.cmd_stop))
        self.application.add_handler(CommandHandler("resume", self.cmd_resume))
        self.application.add_handler(CommandHandler("stats", self.cmd_stats))
        self.application.add_handler(CommandHandler("analysis", self.cmd_analysis))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(format_start_message(), parse_mode=ParseMode.MARKDOWN)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(format_help(COMMANDS), parse_mode=ParseMode.MARKDOWN)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        status = self.coordinator.status()
        await update.message.reply_text(
            format_status(status, self.timezone_label, self.timezone),
            parse_mode=ParseMode.MARKDOWN,
        )

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        balance = self.coordinator.get_balance()
        await update.message.reply_text(format_balance(balance), parse_mode=ParseMode.MARKDOWN)

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        positions = self.coordinator.get_positions()
        await update.message.reply_text(
            format_positions(positions.get("list", [])),
            parse_mode=ParseMode.MARKDOWN,
        )

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        history_payload = self.coordinator.get_history()
        history = history_payload.get("list", [])
        await update.message.reply_text(format_history(history), parse_mode=ParseMode.MARKDOWN)

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.coordinator.pause()
        await update.message.reply_text("⛔ Торговля остановлена.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.coordinator.resume()
        await update.message.reply_text("▶️ Торговля возобновлена.")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        status = self.coordinator.status()
        stats_payload = {
            "Последний сигнал": status.get("signal", "HOLD"),
            "Активен": "Да" if status.get("active") else "Нет",
            "Плечо": f"x{status.get('leverage', 0)}",
        }
        await update.message.reply_text(format_stats(stats_payload), parse_mode=ParseMode.MARKDOWN)

    async def cmd_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        opportunities = self.coordinator.scan_opportunities()
        await update.message.reply_text(
            format_opportunities(opportunities),
            parse_mode=ParseMode.MARKDOWN,
        )

    async def start(self) -> None:
        if self._task and not self._task.done():
            logger.info("Telegram bot already running")
            return
        logger.info("Initializing Telegram bot")
        await self.application.initialize()
        await self.application.start()
        logger.info("Starting Telegram polling loop")
        self._task = asyncio.create_task(self.application.updater.start_polling())

    async def stop(self) -> None:
        if self._task:
            logger.info("Stopping Telegram bot")
            await self.application.updater.stop()
            await self._task
            await self.application.stop()
            await self.application.shutdown()
            self._task = None

    async def notify_startup(self, status: Dict[str, object]) -> None:
        message = format_startup_notification(status, self.timezone_label, self.timezone)
        await self._send_message(message)

    async def handle_cycle(self, result: Dict[str, object]) -> None:
        try:
            execution = result.get("execution")
            if execution:
                message = format_trade_open_event(
                    execution,
                    execution_snapshot=result.get("execution_snapshot"),
                    probabilities=result.get("probabilities"),
                    component_support=result.get("component_support"),
                    risk_targets=result.get("risk_targets"),
                    leverage=self.coordinator.risk_manager.leverage(),
                    learning_rule=result.get("learning_rule", "Disco57"),
                    timezone=self.timezone,
                )
                await self._send_message(message)
            closed_positions: Iterable[dict] = result.get("closed_positions") or []
            for closed in closed_positions:
                message = format_trade_close_event(closed, self.timezone)
                await self._send_message(message)
        except Exception as exc:  # noqa: BLE001 - notification should not break cycle
            logger.exception("Failed to send Telegram cycle notification: %s", exc)

    async def _send_message(self, text: str) -> None:
        if not text:
            return
        await self.application.bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )

