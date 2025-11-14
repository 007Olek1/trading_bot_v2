"""
🛠️ DISCO57 BOT - УТИЛИТЫ
Вспомогательные функции для работы бота
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any


def setup_logging(log_file: Path, log_level: str = "INFO") -> logging.Logger:
    """Настройка логирования для бота"""
    logger = logging.getLogger("Disco57Bot")
    logger.setLevel(getattr(logging, log_level))
    
    # Форматтер
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler для файла
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def save_trade_log(trade_data: Dict[str, Any], log_file: Path) -> None:
    """Сохранение информации о сделке в JSON"""
    try:
        # Загружаем существующие записи
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                trades = json.load(f)
        else:
            trades = []
        
        # Добавляем новую сделку
        trade_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        trades.append(trade_data)
        
        # Сохраняем (последние 1000 сделок)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(trades[-1000:], f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logging.getLogger("Disco57Bot").error(f"Ошибка сохранения лога сделки: {e}")


def calculate_position_size(balance: float, position_size_usd: float, leverage: int, price: float) -> float:
    """
    Рассчитывает размер позиции в монетах
    
    Args:
        balance: Доступный баланс в USD
        position_size_usd: Желаемый размер позиции в USD
        leverage: Плечо
        price: Текущая цена монеты
    
    Returns:
        Количество монет для ордера
    """
    # Проверяем достаточность баланса
    required_margin = position_size_usd / leverage
    
    if balance < required_margin:
        return 0.0
    
    # Расчет количества монет
    notional = position_size_usd * leverage
    qty = notional / price
    
    return qty


def calculate_sl_tp_prices(
    entry_price: float,
    side: str,
    sl_percent: float,
    tp_percent: float
) -> Dict[str, float]:
    """
    Рассчитывает цены Stop Loss и Take Profit
    
    Args:
        entry_price: Цена входа
        side: "Buy" или "Sell"
        sl_percent: Процент Stop Loss
        tp_percent: Процент Take Profit
    
    Returns:
        {"stop_loss": price, "take_profit": price}
    """
    if side == "Buy":
        stop_loss = entry_price * (1 - sl_percent / 100)
        take_profit = entry_price * (1 + tp_percent / 100)
    else:  # Sell
        stop_loss = entry_price * (1 + sl_percent / 100)
        take_profit = entry_price * (1 - tp_percent / 100)
    
    return {
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit, 6)
    }


def format_telegram_message(data: Dict[str, Any]) -> str:
    """Форматирование сообщения для Telegram"""
    msg_type = data.get("type", "status")
    
    if msg_type == "trade_open":
        return f"""
🚀 НОВАЯ ПОЗИЦИЯ ОТКРЫТА

Символ: {data.get('symbol')}
Направление: {data.get('side')}
Размер: ${data.get('size', 0):.2f}
Вход: ${data.get('entry_price', 0):.6f}

🎯 TP: ${data.get('take_profit', 0):.6f} (+{data.get('tp_percent', 0):.1f}%)
🛑 SL: ${data.get('stop_loss', 0):.6f} (-{data.get('sl_percent', 0):.1f}%)

Уверенность: {data.get('confidence', 0):.1f}%
Таймфреймы: {data.get('timeframes_aligned', 0)}/4

Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    elif msg_type == "trade_close":
        pnl = data.get('pnl', 0)
        emoji = "💰" if pnl > 0 else "📉"
        return f"""
{emoji} ПОЗИЦИЯ ЗАКРЫТА

Символ: {data.get('symbol')}
Направление: {data.get('side')}
Вход: ${data.get('entry_price', 0):.6f}
Выход: ${data.get('exit_price', 0):.6f}

PnL: ${pnl:.2f} ({data.get('pnl_percent', 0):.2f}%)
Причина: {data.get('reason', 'N/A')}

Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    elif msg_type == "status":
        return f"""
📊 СТАТУС БОТА DISCO57

Режим: {'Активен ✅' if data.get('active') else 'Пауза ⏸'}
Анализируется: {data.get('symbols_count', 0)} монет
Открыто позиций: {data.get('open_positions', 0)}/{data.get('max_positions', 3)}

Последний сигнал: {data.get('last_signal', 'HOLD')}
Уверенность: {data.get('confidence', 0):.1f}%

💰 Баланс: ${data.get('balance', 0):.2f}
Свободно: ${data.get('available', 0):.2f}

Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return str(data)


def round_price(price: float, tick_size: float = 0.01) -> float:
    """Округление цены до tick size биржи"""
    return round(price / tick_size) * tick_size


def round_quantity(quantity: float, qty_step: float = 0.001) -> float:
    """
    Округление количества до qty step биржи
    Убирает лишние знаки после запятой
    """
    if qty_step <= 0:
        qty_step = 0.001
    
    # Округляем до нужного шага
    rounded = round(quantity / qty_step) * qty_step
    
    # Определяем количество знаков после запятой на основе qty_step
    # Например: 0.001 -> 3 знака, 0.01 -> 2 знака, 1 -> 0 знаков
    if qty_step >= 1:
        decimals = 0
    else:
        # Считаем количество знаков после запятой
        qty_str = str(qty_step).rstrip('0')
        if '.' in qty_str:
            decimals = len(qty_str.split('.')[1])
        else:
            decimals = 0
    
    # Округляем до нужного количества знаков и убираем лишние нули
    rounded = round(rounded, decimals)
    
    return rounded


print("✅ Утилиты Disco57 загружены")

