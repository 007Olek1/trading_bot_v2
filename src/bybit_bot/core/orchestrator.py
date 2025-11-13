"""Async scheduler orchestrating market data updates, trading cycle, and notifications."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from bybit_bot.core.coordinator import TradingCoordinator
from bybit_bot.core.storage import StorageManager
from bybit_bot.data.provider import MarketDataProvider

logger = logging.getLogger(__name__)


class CycleNotifier(Protocol):
    async def handle_cycle(self, result: dict[str, object]) -> None:
        ...

@dataclass(slots=True)
class OrchestratorConfig:
    interval_seconds: int = 60
    timezone: str = "UTC"


class TradingOrchestrator:
    """Coordinates periodic data retrieval and trading cycles."""

    def __init__(
        self,
        coordinator: TradingCoordinator,
        data_provider: MarketDataProvider,
        config: OrchestratorConfig | None = None,
        storage_manager: StorageManager | None = None,
        notifier: "CycleNotifier | None" = None,
    ) -> None:
        self.coordinator = coordinator
        self.data_provider = data_provider
        self.config = config or OrchestratorConfig()
        self.storage_manager = storage_manager
        self.notifier = notifier
        self.scheduler = AsyncIOScheduler(timezone=self.config.timezone)
        self._job = None
        self._lock = asyncio.Lock()

    async def _cycle(self) -> None:
        async with self._lock:
            try:
                price_data = await asyncio.to_thread(self.data_provider.fetch)
                result = self.coordinator.run_cycle(price_data)
                if result and self.notifier:
                    await self.notifier.handle_cycle(result)
                if self.storage_manager:
                    await asyncio.to_thread(self.storage_manager.cleanup)
            except Exception as exc:  # noqa: BLE001 - top-level guard
                logger.exception("Trading cycle failed: %s", exc)

    async def start(self) -> None:
        if self._job:
            logger.info("Orchestrator already running")
            return
        logger.info("Starting trading orchestrator with %ss interval", self.config.interval_seconds)
        trigger = IntervalTrigger(seconds=self.config.interval_seconds)
        self._job = self.scheduler.add_job(lambda: asyncio.create_task(self._cycle()), trigger=trigger)
        self.scheduler.start()

    async def stop(self) -> None:
        if self.scheduler.running:
            logger.info("Stopping trading orchestrator")
            self.scheduler.shutdown(wait=False)
        self._job = None

