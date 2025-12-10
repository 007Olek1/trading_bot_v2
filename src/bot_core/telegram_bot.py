from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Set, TYPE_CHECKING

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from trade_history_db import TradeHistoryDB
from .bybit_client import BybitClient
from .config import Config
from .models import Position
from .ml import Disco57Wrapper

if TYPE_CHECKING:
    from .trader import SwingBot

logger = logging.getLogger(__name__)


class TelegramController:
    """Обработчик команд Telegram для управления ботом."""

    def __init__(
        self,
        config: Config,
        swing_bot: "SwingBot",
        client: BybitClient,
        trade_db: TradeHistoryDB,
        ml: Optional[Disco57Wrapper] = None,
    ):
        self._config = config
        self._bot = swing_bot
        self._client = client
        self._trade_db = trade_db
        self._ml = ml

        self._app: Optional[Application] = None
        self._allowed_chat_ids: Set[int] = self._parse_chat_ids(config.telegram.chat_id)

    async def start(self):
        if not self._config.telegram.enabled:
            logger.warning("Telegram не настроен, пропускаем запуск команд")
            return

        self._app = Application.builder().token(self._config.telegram.token).build()
        self._register_handlers()

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram команды активны")

    async def stop(self):
        if not self._app:
            return
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram команды остановлены")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _register_handlers(self):
        assert self._app
        self._app.add_handler(CommandHandler("start", self.cmd_start))
        self._app.add_handler(CommandHandler("help", self.cmd_help))
        self._app.add_handler(CommandHandler("status", self.cmd_status))
        self._app.add_handler(CommandHandler("balance", self.cmd_balance))
        self._app.add_handler(CommandHandler("positions", self.cmd_positions))
        self._app.add_handler(CommandHandler("history", self.cmd_history))
        self._app.add_handler(CommandHandler("stop", self.cmd_stop))
        self._app.add_handler(CommandHandler("resume", self.cmd_resume))
        self._app.add_handler(CommandHandler("stats", self.cmd_stats))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        warsaw_time = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%d.%m.%Y %H:%M:%S")
        message = (
            "🚀 <b>GoldTrigger Bot запущен</b>\n"
            f"⏱ Варшава: {warsaw_time}\n\n"
            "• Плечо: 20x\n"
            "• Маржа: $1 (экспозиция $20)\n"
            "• Таймфреймы: 5m + 15m\n"
            "• TP1: +30% ROI (50% фиксация)\n"
            "• SL: -20…-25% ROI\n"
            "• Трейлинг: каждые +10% ROI\n"
            "• Макс. позиций: 3\n"
        )
        await self._reply(update, message)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        message = (
            "📝 <b>Команды:</b>\n"
            "/start — 🟢 Стартовое сообщение\n"
            "/help — 📝 Список команд\n"
            "/status — 📊 Статус бота\n"
            "/balance — 💰 Баланс аккаунта\n"
            "/positions — 📈 Открытые позиции\n"
            "/history — 📜 История сделок (24ч)\n"
            "/stop — ⛔ Остановить торговлю\n"
            "/resume — ▶️ Возобновить торговлю\n"
            "/stats — 📊 Статистика 24/72ч"
        )
        await self._reply(update, message)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return

        positions = list(self._bot.positions.positions.values())
        unrealized = await self._compute_unrealized(positions)
        trading = "🟢 Активен" if self._bot.trading_enabled else "🔴 Остановлен"

        message = (
            "📊 <b>Статус бота</b>\n\n"
            f"{trading}\n"
            f"📈 Позиций: {len(positions)}/{self._config.trading.max_positions}\n"
            f"💵 Нереализованный PnL: ${unrealized:+.2f}\n"
            f"⏰ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
        )
        await self._reply(update, message)

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        try:
            balance = await self._client.fetch_account_balance()
            total = float(balance.get("totalEquity", 0))
            avail = float(balance.get("availableBalance", 0))
            unrealized = float(balance.get("unrealisedPnl", 0))
            message = (
                "💰 <b>Баланс Bybit</b>\n\n"
                f"Всего: ${total:.2f}\n"
                f"Доступно: ${avail:.2f}\n"
                f"Нереализованный PnL: ${unrealized:+.2f}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Balance error: %s", exc)
            message = "❌ Не удалось получить баланс."
        await self._reply(update, message)

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        positions = list(self._bot.positions.positions.values())
        if not positions:
            await self._reply(update, "📭 <b>Нет открытых позиций</b>")
            return

        rows = []
        for pos in positions:
            pnl = await self._position_pnl(pos)
            side_emoji = "🟢" if pos.side == "long" else "🔴"
            rows.append(
                f"{side_emoji} <b>{pos.symbol}</b> {pos.side.upper()} 20x\n"
                f"   Вход: ${pos.entry_price:.6f}\n"
                f"   SL: ${pos.stop_loss:.6f}\n"
                f"   TP1: ${pos.take_profit_partial:.6f}\n"
                f"   PnL: ${pnl:+.2f}\n"
            )
        await self._reply(update, "📈 <b>Открытые позиции</b>\n\n" + "\n".join(rows))

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        trades = self._trade_db.get_recent_trades(hours=24, limit=10)
        if not trades:
            await self._reply(update, "📜 За последние 24ч закрытых сделок нет.")
            return
        lines = ["📜 <b>Последние сделки (24ч)</b>\n"]
        for trade in trades[:5]:
            emoji = "✅" if (trade.get("pnl_usd", 0) or 0) > 0 else "❌"
            lines.append(
                f"{emoji} {trade['symbol']} {trade['side'].upper()} "
                f"{trade.get('pnl_usd', 0):+.2f} USD [{trade.get('reason', '')}]"
            )
        await self._reply(update, "\n".join(lines))

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        self._bot.trading_enabled = False
        await self._reply(update, "⛔ Торговля остановлена. Новые позиции не открываются.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        self._bot.trading_enabled = True
        await self._reply(update, "▶️ Торговля возобновлена. Сканер снова активен.")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        stats_24 = self._trade_db.get_stats(hours=24)
        stats_72 = self._trade_db.get_stats(hours=72)
        message = (
            "📊 <b>Статистика</b>\n"
            "━━━ 24 часа ━━━\n"
            f"Сделок: {stats_24['total_trades']}\n"
            f"Win Rate: {stats_24['win_rate']:.0f}%\n"
            f"PnL: ${stats_24['total_pnl']:+.2f}\n\n"
            "━━━ 72 часа ━━━\n"
            f"Сделок: {stats_72['total_trades']}\n"
            f"Win Rate: {stats_72['win_rate']:.0f}%\n"
            f"PnL: ${stats_72['total_pnl']:+.2f}"
        )
        if self._ml:
            ml_stats = self._ml.stats()
            message += (
                "\n\n🤖 <b>Disco57</b>\n"
                f"Трейдов: {ml_stats.get('total_trades', 0)}\n"
                f"Win Rate: {ml_stats.get('win_rate', 0):.1f}%"
            )
        await self._reply(update, message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _position_pnl(self, pos: Position) -> float:
        ticker = await self._client.fetch_ticker(pos.symbol)
        price = float(ticker.get("lastPrice") or ticker.get("last") or 0)
        if price <= 0:
            return 0.0
        return pos.pnl_usd(price, pos.notional_remaining)

    async def _compute_unrealized(self, positions):
        total = 0.0
        for pos in positions:
            total += await self._position_pnl(pos)
        return total

    async def _reply(self, update: Update, text: str):
        if update.message:
            await update.message.reply_text(text, parse_mode="HTML")
        elif update.effective_chat:
            await update.effective_chat.send_message(text, parse_mode="HTML")

    def _parse_chat_ids(self, chat_id: Optional[str]) -> Set[int]:
        ids: Set[int] = set()
        if not chat_id:
            return ids
        for part in str(chat_id).replace(",", " ").split():
            try:
                ids.add(int(part))
            except ValueError:
                continue
        return ids

    async def _check_auth(self, update: Update) -> bool:
        if not self._allowed_chat_ids:
            return True
        chat = update.effective_chat
        if chat and chat.id in self._allowed_chat_ids:
            return True
        logger.warning("Неавторизованный доступ к Telegram командам: %s", chat.id if chat else "unknown")
        await self._reply(update, "⛔ Доступ запрещён. Добавьте chat_id в TELEGRAM_CHAT_ID.")
        return False


__all__ = ["TelegramController"]
