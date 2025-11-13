from __future__ import annotations

import pytest

from bybit_bot.api.client import BybitClient, OrderRequest


class DummyHTTP:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.calls = {}

    def get_server_time(self):
        self.calls.setdefault("get_server_time", 0)
        self.calls["get_server_time"] += 1
        return {"retCode": 0, "result": {"timeSecond": 1234567890}}

    def get_wallet_balance(self, **kwargs):
        self.calls.setdefault("get_wallet_balance", [])
        self.calls["get_wallet_balance"].append(kwargs)
        return {"retCode": 0, "result": {"list": []}}

    def place_order(self, **kwargs):
        self.calls.setdefault("place_order", [])
        self.calls["place_order"].append(kwargs)
        return {"retCode": 0, "result": {"orderId": "abc123"}}


@pytest.fixture(autouse=True)
def patch_http(monkeypatch):
    dummy_instances = []

    def factory(*args, **kwargs):
        instance = DummyHTTP(*args, **kwargs)
        dummy_instances.append(instance)
        return instance

    monkeypatch.setattr("bybit_bot.api.client.HTTP", factory)
    return dummy_instances


def test_client_initialization_uses_settings(patch_http):
    client = BybitClient()
    assert patch_http
    dummy_http = patch_http[0]
    assert dummy_http.kwargs["testnet"] is False
    assert dummy_http.kwargs["recv_window"] == 5000
    assert client.ping() is True
    assert dummy_http.calls["get_server_time"] == 1


def test_place_order_returns_result(patch_http):
    client = BybitClient()
    order = OrderRequest(symbol="BTCUSDT", side="Buy", order_type="Market", qty=0.01)
    result = client.place_order(order)
    assert result["orderId"] == "abc123"
    dummy_http = patch_http[0]
    assert dummy_http.calls["place_order"][0]["symbol"] == "BTCUSDT"

