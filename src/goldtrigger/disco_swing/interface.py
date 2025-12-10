import logging
from typing import Dict, Tuple

from goldtrigger.utils.logging import get_child_logger, setup_logging


class DiscoSwingInterface:
    """
    Stub interface for Disco57-Swing model.
    In shadow mode it always BLOCKs but logs inputs for future training.
    """

    def __init__(self, mode: str = "shadow"):
        self.mode = mode
        self.logger = get_child_logger(setup_logging(), "disco_swing")
        self.model = None
        self.version = "stub-0.1"
        self.total_trades = 0
        self._wins = 0
        self.logger.info("DiscoSwingInterface initialized mode=%s", mode)

    def load_model(self, path: str | None = None):
        self.logger.info("Loading Disco57-Swing model (stub) from %s", path or "<default>")
        self.model = object()

    def health(self) -> Dict[str, str]:
        status = "ok" if self.model else "not_loaded"
        return {"status": status, "version": self.version}

    def infer_allow(self, features: Dict) -> Tuple[bool, str]:
        if self.mode == "shadow":
            self.logger.debug("Shadow mode features snapshot: %s", features)
            return False, "shadow_block"
        return False, "stub_block"

    def record_trade_result(self, pnl_usd: float):
        """Stub metric collector to mimic Disco57 stats."""
        self.total_trades += 1
        if pnl_usd > 0:
            self._wins += 1

    def get_win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self._wins / self.total_trades) * 100.0
