from __future__ import annotations

from bybit_bot.telegram import messages


def test_format_help_includes_commands():
    text = messages.format_help([("/start", "Start bot")])
    assert "📋 *ДОСТУПНЫЕ КОМАНДЫ:*" in text
    assert "Start bot" in text
    assert "💡 Отправьте команду" in text


def test_format_balance_formats_values():
    balance = {"totalEquity": 1234.56, "availableBalance": 789.0, "walletBalance": 1000.0}
    text = messages.format_balance(balance)
    assert "1,234.56" in text
    assert "789.00" in text


def test_format_positions_handles_empty():
    text = messages.format_positions([])
    assert "ОТКРЫТЫЕ ПОЗИЦИИ" in text
    assert "Нет активных позиций" in text


def test_format_positions_renders_details():
    pos = [
        {
            "symbol": "BTCUSDT",
            "side": "Long",
            "size": 1,
            "avgPrice": 110929.3,
            "lastPrice": 110240.53,
            "unrealisedPnl": -0.16,
            "takeProfitPrice": 113147.8,
            "stopLossPrice": 107601.4,
        }
    ]
    text = messages.format_positions(pos)
    assert "🟢 BTCUSDT LONG" in text
    assert "🎯 TP" in text
    assert "🛑 SL" in text


def test_format_opportunities_message():
    opportunities = [
        {"symbol": "BTCUSDT", "direction": "LONG", "confidence": 0.78},
        {"symbol": "ETHUSDT", "direction": "SHORT", "confidence": 0.65},
    ]
    text = messages.format_opportunities(opportunities)
    assert "🔍 *Анализ рынка*" in text
    assert "🟢 BTCUSDT" in text
    assert "BTCUSDT LONG" in text
    assert "ETHUSDT SHORT" in text


def test_format_startup_notification_contains_super_emoji():
    status = {
        "balance": {"totalEquity": 1200.5, "availableBalance": 800.25},
        "leverage": 10,
        "risk_targets": {"tp": 1.0},
        "max_positions": 3,
        "server_host": "185.70.199.244",
    }
    text = messages.format_startup_notification(status, "Warsaw")
    assert "🚀 *БОТ — ЗАПУЩЕН!*" in text
    assert "📋 ДОСТУПНЫЕ КОМАНДЫ" in text
    assert "Позиция: `$1.00`" in text


def test_format_trade_open_event_highlights_probabilities():
    execution = {"symbol": "BTCUSDT", "side": "BUY", "size": 1.0}
    snapshot = {"avgPrice": 110000.0}
    text = messages.format_trade_open_event(
        execution,
        execution_snapshot=snapshot,
        probabilities=[0.2, 0.8],
        component_support={"RandomForest": 0.9, "LightGBM": 0.7},
        risk_targets={"tp": 1.0, "sl": -1.0},
        leverage=10,
        learning_rule="Disco57",
    )
    assert "🚀 *НОВАЯ СДЕЛКА ОТКРЫТА!*" in text
    assert "📊 Вероятности — BUY: 80.00% | SELL: 20.00%" in text
    assert "🤖 Модели Disco57" in text


def test_format_trade_close_event_manual_reason():
    event = {
        "reason": "manual",
        "symbol": "ETHUSDT",
        "side": "SELL",
        "entry_price": 3000.0,
        "exit_price": 2900.0,
        "pnl": 5.25,
        "size": 1.0,
        "timestamp": "2025-11-12T12:00:00Z",
    }
    text = messages.format_trade_close_event(event)
    assert "✅ *СДЕЛКА ЗАКРЫТА!*" in text
    assert "Ручное закрытие на бирже" in text
    assert "ETHUSUT SHORT" not in text  # ensure formatting handles side
    assert "ETHUSDT SHORT" in text

