#!/usr/bin/env python3
"""
Telegram notifier для нового Swing Bot.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import aiohttp
import pytz
from dotenv import load_dotenv

from src.bot_core.risk_settings import FIXED_SL_USD, FIXED_TP_USD, RR_RATIO, TELEGRAM_STARTUP_MESSAGE

load_dotenv()

logger = logging.getLogger(__name__)
WARSAW_TZ = pytz.timezone("Europe/Warsaw")


def _fmt_usd(value: float) -> str:
    return f"${value:.2f}"


class TelegramNotifier:
    """Асинхронные уведомления в Telegram через Bot API."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage" if self.enabled else ""
        self.retry_count = 3
        if not self.enabled:
            logger.warning("Telegram уведомления отключены (не указан TELEGRAM_TOKEN или CHAT_ID)")

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        for attempt in range(self.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode}
                    async with session.post(self.api_url, json=payload) as response:
                        if response.status == 200:
                            return True
                        error_text = await response.text()
                        logger.warning("Telegram API error %s: %s", response.status, error_text)
            except Exception as exc:  # pylint: disable=broad-except
                if attempt < self.retry_count - 1:
                    logger.warning("Ошибка отправки в Telegram (попытка %s): %s", attempt + 1, exc)
                    await asyncio.sleep(1)
                else:
                    logger.error("Не удалось отправить Telegram сообщение: %s", exc)
        return False

    async def send_startup(self):
        await self.send_message(TELEGRAM_STARTUP_MESSAGE)

    async def send_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        leverage: int,
        margin_usd: float,
    ):
        side_emoji = "🟢" if side == "long" else "🔴"
        now_local = datetime.now(WARSAW_TZ).strftime("%H:%M:%S")
        message = (
            f"{side_emoji} <b>Вход в позицию</b>\n"
            f"🪙 {symbol} {side.upper()} | {leverage}x\n"
            f"⏰ {now_local} (Warsaw)\n"
            f"📍 Вход: ${entry_price:.6f}\n"
            f"🛡 SL: ${stop_loss:.6f} ({_fmt_usd(FIXED_SL_USD)})\n"
            f"🎯 TP: ${take_profit:.6f} ({_fmt_usd(FIXED_TP_USD)} | R/R 1:{RR_RATIO})\n"
            f"⚖️ Объём: {quantity:.4f} (маржа {_fmt_usd(margin_usd)})"
        )
        await self.send_message(message)

    async def send_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        reason: str,
        duration_min: int,
        daily_pnl: float,
    ):
        short_symbol = symbol.replace("/USDT:USDT", "")
        emoji = "✅" if pnl_usd >= 0 else "❌"
        direction = "LONG" if side == "long" else "SHORT"
        message = (
            f"{emoji} <b>Сделка закрыта</b>\n"
            f"🪙 {short_symbol} {direction}\n"
            f"Причина: {reason}\n"
            f"Вход: ${entry_price:.6f}\n"
            f"Выход: ${exit_price:.6f}\n"
            f"PnL: {_fmt_usd(pnl_usd)} | Дневной: {_fmt_usd(daily_pnl)}\n"
            f"⏱ Время в позиции: {duration_min} мин"
        )
        await self.send_message(message)

    async def send_health_alert(self, status: str, reason: str):
        message = (
            "🩺 <b>Health сигнал</b>\n"
            f"Статус: {status}\n"
            f"Причина: {reason}"
        )
        await self.send_message(message)

    async def send_shutdown(self, reason: str):
        message = (
            "⛔ <b>GoldTrigger_bot остановлен</b>\n"
            f"Причина: {reason}\n"
            "Все позиции закрыты, задачи остановлены."
        )
        await self.send_message(message)

    async def send_error(self, message: str):
        await self.send_message(f"⚠️ <b>Ошибка</b>\n{message}")


async def _debug():
    logging.basicConfig(level=logging.INFO)
    notifier = TelegramNotifier()
    if not notifier.enabled:
        print("Telegram не настроен")
        return
    await notifier.send_startup()


if __name__ == "__main__":
    asyncio.run(_debug())
