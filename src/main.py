from __future__ import annotations

import asyncio
import logging
import signal

from .bot_core.config import load_config
from .bot_core.bybit_client import BybitClient
from .bot_core.ml import Disco57Wrapper
from .bot_core.trader import SwingBot
from .bot_core.telegram_bot import TelegramController
from telegram_notifier import TelegramNotifier
from trade_history_db import TradeHistoryDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


async def run_bot():
    config = load_config()
    trade_db = TradeHistoryDB(db_path=str(config.paths.trade_db_path))
    bybit_client = BybitClient(config.bybit)
    notifier = TelegramNotifier()
    ml = Disco57Wrapper(config.paths.disco_model_path)

    bot = SwingBot(
        config=config,
        client=bybit_client,
        trade_db=trade_db,
        notifier=notifier,
        ml=ml,
    )

    telegram = TelegramController(
        config=config,
        swing_bot=bot,
        client=bybit_client,
        trade_db=trade_db,
        ml=ml,
    )

    stopper = asyncio.Event()

    def _handle_stop(*_):
        logger.info("Получен сигнал остановки")
        stopper.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_stop)

    await bot.start()
    await telegram.start()
    if notifier.enabled:
        await notifier.send_startup()

    await stopper.wait()
    await telegram.stop()
    await bot.stop()


def main():
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Завершение по Ctrl+C")


if __name__ == "__main__":
    main()
