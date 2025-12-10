from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from disco57_simple import Disco57Simple

from .models import Position, SignalSnapshot

logger = logging.getLogger(__name__)


class Disco57Wrapper:
    """Обёртка над Disco57Simple для интеграции с новым ботом."""

    def __init__(self, model_path: Path):
        self._model_path = model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = Disco57Simple(model_path=str(model_path))
        self._buffer: list[tuple[SignalSnapshot, bool]] = []
        self._buffer_limit = 50
        self._learn_interval = 5

    def allow_signal(self, snapshot: SignalSnapshot) -> bool:
        decision = self._model.predict(
            price=snapshot.entry_price,
            volume_ratio=snapshot.volume_ratio,
            momentum=snapshot.momentum,
            volatility=snapshot.volatility,
        )
        logger.debug(
            "Disco57 decision for %s: %s (vol_ratio=%.2f momentum=%.4f volatility=%.4f)",
            snapshot.symbol,
            decision,
            snapshot.volume_ratio,
            snapshot.momentum,
            snapshot.volatility,
        )
        return decision == "ALLOW"

    def learn_from_position(self, pos: Position, success: bool):
        snapshot = pos.snapshot
        if not snapshot:
            return
        self._buffer.append((snapshot, success))
        if len(self._buffer) < self._learn_interval:
            return
        recent = self._buffer[-self._buffer_limit :]
        wins = sum(1 for _, ok in recent if ok)
        losses = len(recent) - wins
        if losses > 0 and wins / losses < 0.5:
            logger.info("Disco57: буфер показывает низкий WR (%.1f%%), пропускаем обучение", (wins / len(recent)) * 100)
            self._buffer = recent
            return
        for snap, ok in recent[-self._learn_interval :]:
            self._model.learn(
                price=snap.entry_price,
                volume_ratio=snap.volume_ratio,
                momentum=snap.momentum,
                volatility=snap.volatility,
                success=ok,
            )
        self._buffer = recent

    def stats(self) -> dict:
        return self._model.get_stats()


__all__ = ["Disco57Wrapper"]
