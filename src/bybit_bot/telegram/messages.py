"""Formatting helpers for Telegram responses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo


def _format_now(tz: ZoneInfo | None, fmt: str) -> str:
    return datetime.now(tz or timezone.utc).strftime(fmt)


def format_header(title: str) -> str:
    return f"*{title}*"


def format_start_message() -> str:
    return "\n".join(
        [
            "🚀 *БОТ — ЗАПУЩЕН!*",
            "",
            "💠 Добро пожаловать! Я управляю торговлей на Bybit фьючерсах с использованием ML/AI ансамбля и тысяч микропаттернов, собранных в единую логику 🧠.",
            "",
            "📊 *MTF Таймфреймы:*",
            "15m ⏩ 30m ⏩ 1h ⏩ 4h ⏩ 24h",
            "",
            "🤖 *Логика:*",
            "🧠 Ансамбль Disco57 с автоадаптацией",
            "📐 EMA(4h) ↔ EMA(24h) | 🎯 Уверенность ≥ 0.70",
            "",
            "📋 Используй `/help`, чтобы увидеть список доступных команд.",
        ]
    )


def format_help(commands: Sequence[tuple[str, str]]) -> str:
    lines = [
        "📋 *ДОСТУПНЫЕ КОМАНДЫ:*",
        "",
        "🔹 *Основные:*",
    ]
    for cmd, description in commands[:6]:
        lines.append(f"`{cmd}` — {description}")
    lines.append("")
    lines.append("🔹 *Управление:*")
    for cmd, description in commands[6:]:
        lines.append(f"`{cmd}` — {description}")
    lines.append("")
    lines.append("💡 Отправьте команду для выполнения!")
    return "\n".join(lines)


def format_status(status: dict, timezone: str, tz: ZoneInfo | None = None) -> str:
    signal = status.get("signal", "HOLD")
    active = status.get("active", False)
    leverage = status.get("leverage", 0)
    symbol = status.get("symbol", "BTCUSDT")
    server = status.get("server_host", "N/A")
    probabilities = status.get("probabilities")
    prob_line = ""
    if probabilities:
        prob_line = f"\nВероятность BUY: {probabilities[-1][1]:.2f}"
    threshold = status.get("threshold")
    weights = status.get("model_weights") or {}
    opportunities = status.get("opportunities") or []
    balance = status.get("balance") or {}
    risk_targets = status.get("risk_targets") or {}
    timestamp = _format_now(tz, "%Y-%m-%d %H:%M:%S")
    lines = [
        "📊 *Статус бота*",
        f"Режим: {'Активен' if active else 'Пауза'}",
        f"Сигнал: `{signal}`{prob_line}",
        "Таймфреймы: 15m ⏩ 30m ⏩ 1h ⏩ 4h ⏩ 24h",
        f"Символ: `{symbol}` | Плечо: x{leverage}",
    ]
    if threshold is not None:
        lines.append(f"Порог сигнала: {threshold:.2f}")
    if weights:
        weight_str = ", ".join(f"{name}: {weight:.2f}" for name, weight in weights.items())
        lines.append(f"Весы моделей: {weight_str}")
    if risk_targets:
        tp = float(risk_targets.get("tp", 0.0))
        sl = float(risk_targets.get("sl", 0.0))
        lines.append("")
        lines.append("🎯 TP: +${:.2f} + Trailing".format(tp))
        lines.append("🛑 SL: ${:.2f}".format(sl))
    lines.append("")
    total_equity = float(balance.get("totalEquity", 0.0))
    available = float(balance.get("availableBalance", 0.0))
    lines.extend(
        [
            "💰 *Баланс*",
            f"💵 Всего: `${total_equity:,.2f}`",
            f"💸 Свободно: `${available:,.2f}`",
        ]
    )
    lines.append("")
    lines.append("📌 Позиции: используйте `/positions`")
    if opportunities:
        lines.append("")
        lines.append(_format_opportunities_section(opportunities, heading=False))
    else:
        lines.append("")
        lines.append("🚀 *Монеты с высокой точностью:*")
        lines.append("_Нет подходящих возможностей._")
    lines.extend(
        [
            f"Сервер: `{server}`",
            f"Время ({timezone}): {timestamp}",
            f"⏱️ Анализ: каждые {status.get('analysis_interval', '15m')}",
            f"📊 Мониторинг: каждую {status.get('monitoring_interval', '1m')}",
        ]
    )
    return "\n".join(lines)


def format_balance(balance: dict) -> str:
    total = balance.get("totalEquity", 0.0)
    free = balance.get("availableBalance", 0.0)
    wallet = balance.get("walletBalance", total)
    return "\n".join(
        [
            "💰 *Баланс аккаунта*",
            f"Общий: `${total:,.2f}`",
            f"Доступно: `${free:,.2f}`",
            f"В кошельке: `${wallet:,.2f}`",
        ]
    )


def format_positions(positions: Iterable[dict]) -> str:
    lines = ["📊 *ОТКРЫТЫЕ ПОЗИЦИИ*"]
    entries = list(positions)
    if not entries:
        lines.append("_Нет активных позиций._")
        return "\n".join(lines)
    for pos in entries:
        symbol = pos.get("symbol", "N/A")
        side = pos.get("side", "N/A").upper()
        size = pos.get("size", 0)
        entry = float(pos.get("avgPrice", 0.0))
        last_raw = pos.get("lastPrice") or pos.get("markPrice") or entry
        last_price = float(last_raw)
        pnl = float(pos.get("unrealisedPnl", 0.0))
        tp = pos.get("takeProfitPrice")
        sl = pos.get("stopLossPrice")

        arrow = "🟢" if side in {"BUY", "LONG"} else "🔴"
        lines.append(f"{arrow} {symbol} {side}")
        lines.append(f"💵 Вход: ${entry:,.5f} | Текущая: ${last_price:,.5f}")
        lines.append(f"📊 uPnL: {pnl:+.2f} USDT")
        if tp:
            tp_value = float(tp)
            if tp_value:
                tp_pct = ((tp_value - entry) / entry) * 100 if entry else 0.0
                lines.append(f"🎯 TP: ${tp_value:,.5f} ({tp_pct:+.3f}%)")
        if sl:
            sl_value = float(sl)
            if sl_value:
                sl_pct = ((sl_value - entry) / entry) * 100 if entry else 0.0
                lines.append(f"🛑 SL: ${sl_value:,.5f} ({sl_pct:+.3f}%)")
        if size:
            lines.append(f"⚖️ Размер: {size}")
        lines.append("")
    return "\n".join(lines).strip()


def format_history(orders: Iterable[dict], limit: int = 10) -> str:
    lines = ["📜 *История сделок*"]
    data = list(orders)[:limit]
    if not data:
        lines.append("_История пуста._")
        return "\n".join(lines)
    for order in data:
        symbol = order.get("symbol", "N/A")
        side = order.get("side", "N/A")
        qty = order.get("qty", 0)
        price = order.get("avgPrice", order.get("price", 0))
        status = order.get("orderStatus", "N/A")
        lines.append(f"- `{symbol}` {side} {qty} @ {price} — {status}")
    return "\n".join(lines)


def format_stats(stats: dict) -> str:
    lines = ["📊 *Статистика*"]
    if not stats:
        lines.append("_Нет данных._")
        return "\n".join(lines)
    for key, value in stats.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def format_opportunities(opportunities: Sequence[dict]) -> str:
    return _format_opportunities_section(opportunities, heading=True)


def _format_opportunities_section(opportunities: Sequence[dict], heading: bool) -> str:
    lines = []
    if heading:
        lines.append("🔍 *Анализ рынка*")
    if not opportunities:
        lines.append("_Нет подходящих возможностей._")
        return "\n".join(lines)
    for opp in opportunities:
        arrow = "🟢" if opp.get("direction") == "LONG" else "🔴"
        confidence = float(opp.get("confidence", 0.0)) * 100
        symbol = opp.get("symbol", "")
        direction = opp.get("direction", "")
        marker = "" if opp.get("above_threshold", True) else " ⚠️"
        lines.append(f"{arrow}{marker} {symbol} {direction} — {confidence:.2f}%")
        if not opp.get("above_threshold", True):
            lines.append("   ↳ ниже порога уверенности, требуется ручная проверка")
    return "\n".join(lines)


def format_startup_notification(status: dict, timezone_label: str, tz: ZoneInfo | None = None) -> str:
    balance = status.get("balance") or {}
    total_equity = float(balance.get("totalEquity", 0.0))
    available = float(balance.get("availableBalance", 0.0))
    leverage = status.get("leverage", 10)
    risk_targets = status.get("risk_targets") or {}
    position_size = float(risk_targets.get("tp", 1.0))
    max_positions = status.get("max_positions") or 3
    server = status.get("server_host", "N/A")
    timestamp = _format_now(tz, "%d.%m.%Y %H:%M:%S")
    lines = [
        "💠 🚀 *БОТ ЗАПУЩЕН* | MULTI_OPTIMIZE",
        "",
        "⚡ 📊 *Таймфреймы:*",
        "🪩 15m ⏩ 30m ⏩ 1h ⏩ 4h ⏩ 24h",
        "",
        "⚙️ *Параметры:*",
        f"🔸 Плечо: ×{leverage}",
        f"🔸 Позиция: ${position_size:.2f}",
        f"🔸 Макс. позиций: {max_positions}",
        "",
        "🤖 *Логика:*",
        "🧠 Ансамбль Disco57 с автоадаптацией",
        "📐 EMA(4h) ↔ EMA(24h) | 🎯 Уверенность ≥ 0.70",
        "",
        "💰 *Баланс:*",
        f"💵 Всего: ${total_equity:,.2f}",
        f"💸 Свободно: ${available:,.2f}",
        "",
        "🧩 *Команды:*",
        "🔹 /start · /status · /positions",
        "🔹 /balance · /stats · /history",
        "🔹 /analysis · /stop · /resume",
        "",
        f"⏰ Warsaw: {timestamp}",
        f"📡 Server: {server}",
    ]
    return "\n".join(lines)


def format_trade_open_event(
    execution: dict,
    *,
    execution_snapshot: dict | None,
    probabilities: Sequence[float] | None,
    component_support: dict | None,
    risk_targets: dict | None,
    leverage: int,
    learning_rule: str,
    timezone: ZoneInfo | None = None,
) -> str:
    symbol = execution.get("symbol", "N/A")
    side = execution.get("side", "BUY")
    human_side = "LONG" if side.upper() == "BUY" else "SHORT"
    arrow = "🟢" if side.upper() == "BUY" else "🔴"
    size = float(execution.get("size", 0.0))
    entry = 0.0
    if execution_snapshot:
        entry = float(execution_snapshot.get("avgPrice") or 0.0)
    tp_val = float((risk_targets or {}).get("tp", 0.0))
    sl_val = float((risk_targets or {}).get("sl", 0.0))
    buy_prob = None
    sell_prob = None
    if probabilities and len(probabilities) >= 2:
        sell_prob, buy_prob = probabilities[0], probabilities[1]
    components = component_support or {}
    top_components = sorted(components.items(), key=lambda item: item[1], reverse=True)[:3]
    timestamp = _format_now(timezone, "%d.%m.%Y %H:%M:%S %Z")
    lines = [
        "🚀 *НОВАЯ СДЕЛКА ОТКРЫТА!*",
        f"{arrow} {symbol} {human_side}",
        f"💵 Вход: ${entry:,.5f}",
        f"⚖️ Размер: ${size:,.2f} | ⚡ Плечо: x{leverage}",
        f"🎯 TP: +${tp_val:.2f} + Trailing",
        f"🛑 SL: ${sl_val:.2f}",
    ]
    if buy_prob is not None and sell_prob is not None:
        lines.append(f"📊 Вероятности — BUY: {buy_prob*100:.2f}% | SELL: {sell_prob*100:.2f}%")
    if top_components:
        comp_lines = ", ".join(f"{name}: {value*100:.1f}%" for name, value in top_components)
        lines.append(f"🤖 Модели Disco57 → {comp_lines}")
    lines.extend(
        [
            f"🧠 Правило обучения: {learning_rule}",
            "📈 Анализируем рынок для новых возможностей.",
            f"⏰ {timestamp}",
        ]
    )
    return "\n".join(lines)


def format_trade_close_event(event: dict, tz: ZoneInfo | None = None) -> str:
    reason = event.get("reason", "manual")
    symbol = event.get("symbol", "N/A")
    side = event.get("side", "BUY")
    human_side = "LONG" if str(side).upper() in {"BUY", "LONG"} else "SHORT"
    arrow = "🟢" if human_side == "LONG" else "🔴"
    entry = float(event.get("entry_price") or 0.0)
    exit_price = float(event.get("exit_price") or 0.0)
    pnl = float(event.get("pnl") or 0.0)
    size = float(event.get("size") or 0.0)
    reason_map = {
        "manual": "Ручное закрытие на бирже",
        "timeout": "Авто-таймер 24 часа",
        "strategy": "Стратегия бота",
    }
    reason_text = reason_map.get(reason, reason)
    raw_timestamp = event.get("timestamp")
    if isinstance(raw_timestamp, str):
        try:
            parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
            timestamp = parsed.astimezone(tz or timezone.utc).strftime("%d.%m.%Y %H:%M:%S %Z")
        except ValueError:
            timestamp = _format_now(tz, "%d.%m.%Y %H:%M:%S %Z")
    else:
        timestamp = _format_now(tz, "%d.%m.%Y %H:%M:%S %Z")
    lines = [
        "✅ *СДЕЛКА ЗАКРЫТА!*",
        f"{arrow} {symbol} {human_side}",
        f"📊 Результат: {pnl:+.2f} USDT",
        f"💵 Вход: ${entry:,.5f} | Выход: ${exit_price:,.5f}",
        f"⚖️ Размер: ${size:,.2f}",
        f"📝 Причина: {reason_text}",
        f"⏰ {timestamp}",
    ]
    return "\n".join(lines)

