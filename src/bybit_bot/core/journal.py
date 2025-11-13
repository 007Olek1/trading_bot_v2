"""Trade journal logging executed orders to CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


@dataclass(slots=True)
class TradeJournal:
    directory: Path
    fieldnames: Iterable[str] = (
        "timestamp",
        "order_id",
        "symbol",
        "side",
        "size",
        "probability_buy",
        "probability_sell",
        "confidence",
        "threshold",
        "execution_context",
    )

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, entry: Dict[str, object]) -> None:
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        payload = {"timestamp": timestamp, **entry}
        file_path = self.directory / f"{datetime.utcnow():%Y-%m-%d}.csv"
        is_new = not file_path.exists()
        with file_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow({key: payload.get(key, "") for key in self.fieldnames})

