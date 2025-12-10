from __future__ import annotations

import time
from typing import Dict


class CooldownManager:
    """Отвечает за блокировку повторных входов по символу."""

    def __init__(self, cooldown_seconds: int):
        self._cooldown_seconds = cooldown_seconds
        self._cooldowns: Dict[str, float] = {}

    def set_cooldown(self, symbol: str) -> None:
        self._cooldowns[symbol] = time.time() + self._cooldown_seconds

    def is_on_cooldown(self, symbol: str) -> bool:
        expires_at = self._cooldowns.get(symbol)
        if not expires_at:
            return False
        if time.time() >= expires_at:
            del self._cooldowns[symbol]
            return False
        return True

    def remaining(self, symbol: str) -> int:
        expires_at = self._cooldowns.get(symbol, 0)
        remaining = expires_at - time.time()
        return int(remaining) if remaining > 0 else 0


__all__ = ["CooldownManager"]
