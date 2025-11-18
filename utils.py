"""
🛠️ УТИЛИТЫ - Вспомогательные функции
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """Настройка логирования"""
    logger = logging.getLogger("TradingBotV4")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Создаем директорию для логов если её нет
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Файловый хендлер
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    
    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # Добавляем хендлеры
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_trade_log(log_file: Path, trade_data: Dict):
    """Сохранение лога сделки"""
    try:
        # Читаем существующие логи
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Добавляем новую сделку
        trade_data['timestamp'] = datetime.now().isoformat()
        logs.append(trade_data)
        
        # Сохраняем
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    
    except Exception as e:
        print(f"Ошибка сохранения лога сделки: {e}")


def calculate_position_size(client, symbol: str, position_size_usd: float, leverage: int, current_price: float) -> Optional[float]:
    """Расчёт размера позиции в монетах"""
    try:
        # Получаем информацию об инструменте
        instrument_info = client.get_instruments_info(
            category="linear",
            symbol=symbol
        )
        
        if instrument_info['retCode'] != 0:
            return None
        
        info = instrument_info['result']['list'][0]
        
        # Минимальный размер позиции
        min_qty = float(info['lotSizeFilter']['minOrderQty'])
        qty_step = float(info['lotSizeFilter']['qtyStep'])
        
        # Рассчитываем количество монет
        # position_size_usd * leverage / current_price
        quantity = (position_size_usd * leverage) / current_price
        
        # Округляем до шага
        quantity = round(quantity / qty_step) * qty_step
        
        # Проверяем минимум
        if quantity < min_qty:
            quantity = min_qty
        
        return quantity
    
    except Exception as e:
        print(f"Ошибка расчёта размера позиции: {e}")
        return None


def calculate_multi_tp_levels(entry_price: float, direction: str, levels: List[Dict]) -> List[Dict]:
    """
    Расчёт множественных TP уровней
    
    Args:
        entry_price: Цена входа
        direction: 'LONG' или 'SHORT'
        levels: Список уровней [{'percent': 4.0, 'size': 0.40}, ...]
    
    Returns:
        Список TP уровней с ценами
    """
    tp_levels = []
    
    for level in levels:
        percent = level['percent'] / 100
        size = level['size']
        
        if direction == 'LONG':
            price = entry_price * (1 + percent)
        else:  # SHORT
            price = entry_price * (1 - percent)
        
        tp_levels.append({
            'price': price,
            'size': size,
            'percent': level['percent'],
            'hit': False
        })
    
    return tp_levels


def calculate_trailing_sl(entry_price: float, direction: str, max_loss_usd: float, position_size_usd: float) -> float:
    """
    Расчёт начального Stop Loss
    
    Args:
        entry_price: Цена входа
        direction: 'LONG' или 'SHORT'
        max_loss_usd: Максимальный убыток в USD
        position_size_usd: Размер позиции в USD (с учётом плеча)
    
    Returns:
        Цена SL
    """
    # Рассчитываем процент убытка
    loss_percent = max_loss_usd / position_size_usd
    
    if direction == 'LONG':
        sl_price = entry_price * (1 - loss_percent)
    else:  # SHORT
        sl_price = entry_price * (1 + loss_percent)
    
    return sl_price


def round_price(symbol: str, price: float) -> float:
    """Округление цены до нужного шага"""
    # Для большинства пар USDT используется 2-4 знака после запятой
    if price >= 1000:
        return round(price, 2)
    elif price >= 100:
        return round(price, 3)
    elif price >= 1:
        return round(price, 4)
    else:
        return round(price, 6)


def round_quantity(symbol: str, quantity: float) -> float:
    """Округление количества до нужного шага"""
    # Для большинства пар используется 0-3 знака после запятой
    if quantity >= 100:
        return round(quantity, 0)
    elif quantity >= 10:
        return round(quantity, 1)
    elif quantity >= 1:
        return round(quantity, 2)
    else:
        return round(quantity, 3)


def format_telegram_message(text: str, **kwargs) -> str:
    """Форматирование сообщения для Telegram"""
    # Заменяем плейсхолдеры
    for key, value in kwargs.items():
        placeholder = f"{{{key}}}"
        text = text.replace(placeholder, str(value))
    
    return text


def calculate_pnl(entry_price: float, current_price: float, direction: str, quantity: float) -> Tuple[float, float]:
    """
    Расчёт PnL (прибыль/убыток)
    
    Returns:
        (pnl_usd, pnl_percent)
    """
    if direction == 'LONG':
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        pnl_usd = (current_price - entry_price) * quantity
    else:  # SHORT
        pnl_percent = ((entry_price - current_price) / entry_price) * 100
        pnl_usd = (entry_price - current_price) * quantity
    
    return pnl_usd, pnl_percent


def format_duration(seconds: float) -> str:
    """Форматирование длительности"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}д")
    if hours > 0:
        parts.append(f"{hours}ч")
    if minutes > 0 or not parts:  # Показываем минуты если это единственное значение
        parts.append(f"{minutes}м")
    
    return " ".join(parts)


def format_datetime(dt) -> str:
    """Форматирование даты и времени для отображения"""
    from datetime import datetime, timezone
    
    # Конвертируем в UTC+1 (ваш часовой пояс)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Форматируем: 17.11.2025 21:30
    return dt.strftime("%d.%m.%Y %H:%M")


def get_balance_info(client) -> Optional[Dict]:
    """Получение информации о балансе"""
    try:
        # Пробуем UNIFIED аккаунт
        response = client.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"
        )
        
        if response['retCode'] != 0:
            print(f"Ошибка получения баланса: {response.get('retMsg', 'Unknown error')}")
            return None
        
        if not response['result']['list']:
            print("Нет данных о балансе")
            return None
        
        balance_data = response['result']['list'][0]['coin'][0]
        
        wallet_balance = float(balance_data.get('walletBalance', 0))
        available = balance_data.get('availableToWithdraw', '')
        
        # Если availableToWithdraw пустое, используем totalAvailableBalance из аккаунта
        if not available or available == '':
            available = response['result']['list'][0].get('totalAvailableBalance', wallet_balance)
        
        available = float(available) if available else wallet_balance
        
        return {
            'total': wallet_balance,
            'available': available,
            'used': wallet_balance - available
        }
    
    except Exception as e:
        print(f"Ошибка получения баланса: {e}")
        return None
