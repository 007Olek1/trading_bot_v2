from __future__ import annotations

import pytest

from bybit_bot.core.risk import RiskConfig, RiskManager


def test_position_size_respects_account_risk():
    config = RiskConfig(base_position_size_usd=100, max_account_risk_pct=1.0)
    manager = RiskManager(config)
    size = manager.position_size(5000)
    assert size == pytest.approx(49.94, rel=1e-3)  # учитываем комиссию обеих сторон


def test_can_open_position():
    manager = RiskManager(RiskConfig(max_concurrent_positions=2))
    assert manager.can_open_position(1) is True
    assert manager.can_open_position(2) is False


def test_adjust_position_size_respects_bounds():
    manager = RiskManager(RiskConfig(base_position_size_usd=1.0, min_position_size_usd=0.5, max_position_size_usd=5.0))
    assert manager.adjust_position_size(1.5) == 1.5
    assert manager.adjust_position_size(10) == 5.0
    assert manager.adjust_position_size(0.05) == 0.5

