from __future__ import annotations

import csv
from pathlib import Path

from bybit_bot.core.journal import TradeJournal


def test_trade_journal_writes_entry(tmp_path):
    journal_dir = tmp_path / "trades"
    journal = TradeJournal(journal_dir)
    entry = {
        "order_id": "123",
        "symbol": "BTCUSDT",
        "side": "LONG",
        "size": 1.0,
        "probability_buy": 0.75,
        "probability_sell": 0.25,
        "confidence": 0.75,
        "threshold": 0.6,
        "execution_context": "BTCUSDT:BUY",
    }
    journal.record(entry)

    files = list(journal_dir.glob("*.csv"))
    assert files, "Journal did not create a CSV file"
    with files[0].open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["side"] == "LONG"

