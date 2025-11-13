from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from bybit_bot.core.storage import StorageConfig, StorageManager


def test_storage_cleanup_removes_old_files(tmp_path):
    config = StorageConfig(
        base_dir=tmp_path,
        logs_dir=tmp_path / "logs",
        trades_dir=tmp_path / "data" / "trades",
        analysis_dir=tmp_path / "data" / "analysis",
        retention_days=1,
    )
    manager = StorageManager(config)
    old_file = config.logs_dir / "old.log"
    old_file.write_text("old")
    old_time = (datetime.utcnow() - timedelta(days=2)).timestamp()
    os.utime(old_file, (old_time, old_time))

    manager.cleanup(now=datetime.utcnow().replace(tzinfo=timezone.utc))
    assert not old_file.exists()

