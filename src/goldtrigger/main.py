#!/usr/bin/env python3
"""
GoldTrigger Swing Bot entrypoint.
Initializes core services (logging, DB, selector, Disco57-Swing stub) and runs the scheduler.
"""

import argparse
import asyncio
import os
import signal
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from aiohttp import web
from dotenv import load_dotenv

from goldtrigger.bybit_api import BybitAPI
from goldtrigger.selector import Selector
from goldtrigger.strategy.swing_bot import SwingBot
from goldtrigger.disco_swing import DiscoSwingInterface
from goldtrigger.db import TradeHistoryDB
from goldtrigger.telegram_notifier import TelegramNotifier
from goldtrigger.telegram_commands import TelegramCommandsHandler
from goldtrigger.utils.logging import setup_logging, get_child_logger
from goldtrigger.disco import Disco57Learner


SCAN_INTERVAL_SECONDS = int(os.getenv("SWING_SCAN_INTERVAL_SEC", 1800))  # 30 минут
FULL_RESCAN_INTERVAL_SECONDS = int(os.getenv("SWING_FULL_RESCAN_SEC", 6 * 3600))
POSITION_POLL_INTERVAL_SECONDS = int(os.getenv("POSITION_POLL_INTERVAL_SEC", 30))
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8088"))


class GoldTriggerSwingApp:
    def __init__(self, mode: str = "paper"):
        load_dotenv()
        self.logger = setup_logging("goldtrigger.swing", "swing_system.log")
        self.mode = mode
        self.api = BybitAPI(paper_trading=(mode == "paper"))
        self.trade_db = TradeHistoryDB()
        self.telegram = TelegramNotifier()
        self.disco = DiscoSwingInterface(mode=os.getenv("DISCO_SWING_MODE", "shadow"))
        disco_model_path = os.getenv("DISCO57_MODEL_PATH", "/opt/GoldTrigger_bot/data/disco57_model.json")
        self.disco_learner = Disco57Learner(model_path=disco_model_path)
        self.selector = Selector(
            exchange=self.api.exchange,
            disco_interface=self.disco,
            config={"disco_learner": self.disco_learner},
        )
        self.bot = SwingBot(
            api=self.api,
            selector=self.selector,
            notifier=self.telegram,
            trade_db=self.trade_db,
            disco=self.disco,
            disco_learner=self.disco_learner,
            mode=mode,
        )
        self.telegram_commands: Optional[TelegramCommandsHandler] = None
        if self.telegram.enabled:
            token = os.getenv("TELEGRAM_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            if token and chat_id:
                self.telegram_commands = TelegramCommandsHandler(
                    bot_instance=self.bot,
                    telegram_token=token,
                    chat_id=chat_id,
                )
        self._scan_task: Optional[asyncio.Task] = None
        self._full_rescan_task: Optional[asyncio.Task] = None
        self._position_task: Optional[asyncio.Task] = None
        self._daily_report_task: Optional[asyncio.Task] = None
        self._health_runner: Optional[web.AppRunner] = None
        self._stop_event = asyncio.Event()
        self.metrics = {
            "open_positions": 0,
            "last_trade_ts": 0,
            "model_loaded": self.disco.health()["status"] == "ok",
            "api_latency_ms_avg": 0,
        }
        self.enable_daily_reports = (
            os.getenv("ENABLE_DAILY_REPORTS", "true").lower() == "true"
        )
        self.daily_report_hour = int(os.getenv("DAILY_REPORT_HOUR", "9"))
        tz_name = os.getenv("REPORT_TIMEZONE", "Europe/Warsaw")
        self.report_timezone = ZoneInfo(tz_name)
        self.last_daily_report_date: Optional[datetime.date] = None
        self.logger.info("GoldTrigger Swing App initialized (mode=%s)", mode)

    async def start(self):
        self.logger.info("Starting swing scheduler (scan interval %ss)", SCAN_INTERVAL_SECONDS)
        await self.selector.initialize()
        if self.telegram_commands:
            await self.telegram_commands.setup_commands()
        self._scan_task = asyncio.create_task(self._scan_loop(), name="swing-scan-loop")
        self._full_rescan_task = asyncio.create_task(
            self._full_rescan_loop(), name="swing-full-rescan"
        )
        self._position_task = asyncio.create_task(
            self._position_loop(), name="swing-position-loop"
        )
        if self.enable_daily_reports and self.telegram.enabled:
            self._daily_report_task = asyncio.create_task(
                self._daily_report_loop(), name="swing-daily-report"
            )
        else:
            self.logger.info("Daily reports disabled (ENABLE_DAILY_REPORTS=%s)", self.enable_daily_reports)
        await self._start_health_server()
        await self.bot.start()  # main run loop (async)

    async def stop(self):
        if self._stop_event.is_set():
            return
        self.logger.info("Stopping GoldTrigger Swing App...")
        self._stop_event.set()
        for task in (
            self._scan_task,
            self._full_rescan_task,
            self._position_task,
            self._daily_report_task,
        ):
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if self.telegram_commands:
            await self.telegram_commands.shutdown()
        if self._health_runner:
            await self._health_runner.cleanup()
        await self.api.close()
        self.trade_db.close()

    async def _scan_loop(self):
        while True:
            try:
                await self.bot.run_scan()
            except Exception as exc:  # pragma: no cover
                self.logger.exception("Scan loop error: %s", exc)
            sleep_seconds = self._seconds_until_next_scan()
            self.logger.debug("Next scan in %.1f minutes", sleep_seconds / 60)
            await asyncio.sleep(sleep_seconds)

    async def _full_rescan_loop(self):
        while True:
            try:
                await self.selector.perform_full_rescan()
            except Exception as exc:  # pragma: no cover
                self.logger.exception("Full rescan error: %s", exc)
            await asyncio.sleep(FULL_RESCAN_INTERVAL_SECONDS)

    async def _position_loop(self):
        while True:
            try:
                await self.bot.update_positions()
            except Exception as exc:
                self.logger.exception("Position loop error: %s", exc)
            await asyncio.sleep(POSITION_POLL_INTERVAL_SECONDS)

    async def _start_health_server(self):
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/metrics", self._handle_metrics)
        self._health_runner = web.AppRunner(app)
        await self._health_runner.setup()
        site = web.TCPSite(self._health_runner, "0.0.0.0", HEALTH_PORT)
        await site.start()
        self.logger.info("Health endpoint running on :%s", HEALTH_PORT)

    async def _handle_health(self, request: web.Request) -> web.Response:
        health = {
            "status": "ok",
            "mode": self.mode,
            "open_positions": self.bot.get_open_positions_count(),
            "model_loaded": self.metrics["model_loaded"],
        }
        return web.json_response(health)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        lines = [
            f"goldtrigger_open_positions {self.bot.get_open_positions_count()}",
            f"goldtrigger_last_trade_ts {self.bot.last_trade_timestamp}",
            f"goldtrigger_model_loaded {1 if self.metrics['model_loaded'] else 0}",
            f"goldtrigger_api_latency_ms_avg {self.metrics['api_latency_ms_avg']}",
        ]
        return web.Response(text="\n".join(lines), content_type="text/plain")

    async def _daily_report_loop(self):
        self.logger.info(
            "Daily report loop enabled (hour=%s tz=%s)", self.daily_report_hour, self.report_timezone
        )
        await self._maybe_send_report_immediately()
        while not self._stop_event.is_set():
            delay = self._seconds_until_next_report()
            self.logger.info("Next daily report in %.2f hours", delay / 3600)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass
            if self._stop_event.is_set():
                break
            await self._send_daily_report()

    async def _maybe_send_report_immediately(self):
        now = datetime.now(self.report_timezone)
        if (
            now.hour >= self.daily_report_hour
            and (self.last_daily_report_date is None or self.last_daily_report_date != now.date())
        ):
            self.logger.info("Sending catch-up daily report for %s", now.date())
            await self._send_daily_report()

    def _seconds_until_next_report(self) -> float:
        now = datetime.now(self.report_timezone)
        target = now.replace(
            hour=self.daily_report_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        return max((target - now).total_seconds(), 60)

    async def _send_daily_report(self):
        if not self.telegram.enabled:
            return
        now = datetime.now(self.report_timezone)
        try:
            balance = await self.api.get_account_balance()
        except Exception as exc:
            self.logger.error("Failed to fetch balance for daily report: %s", exc)
            balance = {}
        total_map = balance.get("total", {}) if isinstance(balance, dict) else {}
        free_map = balance.get("free", {}) if isinstance(balance, dict) else {}
        used_map = balance.get("used", {}) if isinstance(balance, dict) else {}
        usdt_total = float(total_map.get("USDT", 0) or 0)
        usdt_free = float(free_map.get("USDT", usdt_total) or usdt_total)
        usdt_used = float(used_map.get("USDT", max(usdt_total - usdt_free, 0)))

        stats_24h = self.trade_db.get_stats(hours=24)
        message = (
            "📊 <b>Ежедневный отчёт</b>\n"
            f"🕘 {now.strftime('%d.%m.%Y %H:%M')} ({self.report_timezone.key})\n\n"
            "💰 <b>Баланс Bybit</b>\n"
            f"• Всего: ${usdt_total:.2f}\n"
            f"• Свободно: ${usdt_free:.2f}\n"
            f"• В торговле: ${usdt_used:.2f}\n\n"
            "📈 <b>24ч статистика</b>\n"
            f"• Сделок: {stats_24h['total_trades']}\n"
            f"• Win Rate: {stats_24h['win_rate']:.1f}%\n"
            f"• PnL: {'+' if stats_24h['total_pnl'] >= 0 else ''}${stats_24h['total_pnl']:.2f}\n"
            f"• Лучшая: {'+' if stats_24h['best_trade'] >= 0 else ''}${stats_24h['best_trade']:.2f}\n"
            f"• Худшая: {'+' if stats_24h['worst_trade'] >= 0 else ''}${stats_24h['worst_trade']:.2f}\n\n"
            f"🔄 Активных позиций: {self.bot.get_open_positions_count()}/{self.bot.max_positions}"
        )
        await self.telegram.send_message(message)
        self.last_daily_report_date = now.date()
        self.logger.info("Daily report sent for %s", now.date())

    def _seconds_until_next_scan(self) -> float:
        now = datetime.now(timezone.utc)
        seconds_since_hour = (
            now.minute * 60 + now.second + now.microsecond / 1_000_000
        )
        remainder = seconds_since_hour % SCAN_INTERVAL_SECONDS
        if remainder == 0:
            return SCAN_INTERVAL_SECONDS
        return SCAN_INTERVAL_SECONDS - remainder


async def main():
    parser = argparse.ArgumentParser(description="GoldTrigger Swing Bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()
    app = GoldTriggerSwingApp(mode=args.mode)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_stop_signal():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_stop_signal)

    try:
        await app.start()
        await stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
