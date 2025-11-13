"""Storage utilities for managing directories and retention."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StorageConfig:
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    trades_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "trades")
    analysis_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "analysis")
    retention_days: int = 14

    def directories(self) -> Iterable[Path]:
        return (self.logs_dir, self.trades_dir, self.analysis_dir)


class StorageManager:
    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self.ensure_directories()

    def ensure_directories(self) -> None:
        for directory in self.config.directories():
            directory.mkdir(parents=True, exist_ok=True)

    def cleanup(self, now: datetime | None = None) -> None:
        cutoff = (now or datetime.utcnow()).replace(tzinfo=timezone.utc) - timedelta(days=self.config.retention_days)
        for directory in self.config.directories():
            if not directory.exists():
                continue
            for path in directory.glob("**/*"):
                if not path.is_file():
                    continue
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    try:
                        path.unlink()
                        logger.debug("Removed expired file %s", path)
                    except OSError as exc:
                        logger.warning("Failed to remove %s: %s", path, exc)

