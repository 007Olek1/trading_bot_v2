"""Entry point for running the Bybit trading bot with orchestrator and Telegram interface."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Добавляем корневую директорию в путь для импорта smart_coin_selector
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from bybit_bot.api.client import BybitClient
from bybit_bot.core.coordinator import TradingCoordinator
from bybit_bot.core.orchestrator import TradingOrchestrator
from bybit_bot.core.storage import StorageConfig, StorageManager
from bybit_bot.core.journal import TradeJournal
from bybit_bot.data.provider import MarketDataProvider
from bybit_bot.ml.pipeline import EnsemblePipeline
from bybit_bot.telegram.bot import TelegramBot

# Импортируем SmartCoinSelector для динамического выбора монет
try:
    from smart_coin_selector import SmartCoinSelector
    SMART_SELECTOR_AVAILABLE = True
except ImportError:
    SMART_SELECTOR_AVAILABLE = False
    logging.warning("SmartCoinSelector not available, using default watchlist")

MODEL_DIR = Path("models/ensemble")


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(log_dir / "bot.log", maxBytes=5 * 1024 * 1024, backupCount=5),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


def _load_pipeline() -> EnsemblePipeline:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory {MODEL_DIR} not found. Train the ensemble pipeline and place artifacts there."
        )
    return EnsemblePipeline.load(MODEL_DIR)


async def main() -> None:
    storage_manager = StorageManager(StorageConfig(base_dir=Path.cwd()))
    _configure_logging(storage_manager.config.logs_dir)
    logger = logging.getLogger("run_bot")
    logger.info("Starting Bybit Futures AI bot")

    client = BybitClient()
    data_provider = MarketDataProvider()
    pipeline = _load_pipeline()
    journal = TradeJournal(storage_manager.config.trades_dir)
    
    # 🎯 ДИНАМИЧЕСКИЙ ВЫБОР САМЫХ ЛИКВИДНЫХ МОНЕТ
    watchlist = None
    if SMART_SELECTOR_AVAILABLE:
        try:
            # Используем exchange из data_provider
            exchange = data_provider.exchange if hasattr(data_provider, 'exchange') else None
            
            if exchange:
                selector = SmartCoinSelector()
                # Определяем состояние рынка (можно улучшить, добавив анализ)
                market_condition = "normal"  # normal, bullish, bearish, volatile
                
                logger.info("🎯 Выбираем самые ликвидные монеты через SmartCoinSelector...")
                selected_symbols = await selector.get_smart_symbols(exchange, market_condition)
                
                if selected_symbols:
                    # Конвертируем формат символов для watchlist (BTCUSDT -> BTC/USDT)
                    watchlist = []
                    for symbol in selected_symbols[:50]:  # Берем топ-50 самых ликвидных
                        # Конвертируем формат: BTCUSDT -> BTC/USDT, 1000FLOKIUSDT -> FLOKI/USDT
                        if 'USDT' in symbol:
                            # Убираем USDT и префикс 1000 если есть
                            base = symbol.replace('USDT', '').replace('1000', '')
                            # Проверяем что base не пустой
                            if base:
                                watchlist.append(f"{base}/USDT")
                    logger.info(f"✅ Выбрано {len(watchlist)} самых ликвидных монет для торговли")
                    if watchlist:
                        logger.debug(f"   Примеры: {watchlist[:5]}")
                else:
                    logger.warning("⚠️ SmartCoinSelector не вернул монеты, используем fallback")
            else:
                logger.warning("⚠️ Exchange не доступен для SmartCoinSelector, используем fallback")
        except Exception as e:
            logger.error(f"❌ Ошибка при выборе монет через SmartCoinSelector: {e}", exc_info=True)
    
    # Fallback на топ-ликвидные монеты если SmartCoinSelector недоступен
    if not watchlist:
        watchlist = (
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
            "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT",
            "ATOM/USDT", "ETC/USDT", "XLM/USDT", "NEAR/USDT", "ICP/USDT",
            "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT",
            "TIA/USDT", "SEI/USDT", "TRX/USDT", "TON/USDT", "AAVE/USDT",
            "UNI/USDT", "HBAR/USDT", "BCH/USDT", "MATIC/USDT", "INJ/USDT",
            "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT", "BONK/USDT"
        )
        logger.info(f"📋 Используем fallback watchlist: {len(watchlist)} монет")
    
    coordinator = TradingCoordinator(
        client=client,
        pipeline=pipeline,
        data_provider=data_provider,
        watchlist=watchlist,
        journal=journal,
        analysis_dir=storage_manager.config.analysis_dir,
    )
    telegram_bot = TelegramBot(coordinator=coordinator)
    orchestrator = TradingOrchestrator(
        coordinator=coordinator,
        data_provider=data_provider,
        storage_manager=storage_manager,
        notifier=telegram_bot,
    )

    await telegram_bot.start()
    await telegram_bot.notify_startup(coordinator.status())
    await orchestrator.start()

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received")
    finally:
        await orchestrator.stop()
        await telegram_bot.stop()


if __name__ == "__main__":
    asyncio.run(main())

