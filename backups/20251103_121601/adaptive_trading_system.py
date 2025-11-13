from dataclasses import dataclass
from typing import Dict
import logging
from datetime import datetime, timedelta
import json
from adaptive_parameters import AdaptiveParameterSystem
from order_manager import OrderManager
from coin_analyzer import CoinAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """Настройки торговли"""
    position_size: float
    take_profit_percent: float
    trailing_percent: float
    stop_loss_amount: float
    leverage: int

    @classmethod
    def from_parameters(cls, params: Dict) -> 'TradeSetup':
        return cls(
            position_size=params['position_size'],
            take_profit_percent=params['take_profit_percent'],
            trailing_percent=params['trailing_percent'],
            stop_loss_amount=params['stop_loss_amount'],
            leverage=params['leverage']
        )


class AdaptiveTradingSystem:
    """🔄 Адаптивная торговая система"""

    def __init__(self):
        self.parameter_system = AdaptiveParameterSystem()
        self.coin_analyzer = CoinAnalyzer()
        self.trade_setup = TradeSetup(
            position_size=30.0,
            take_profit_percent=2.0,
            trailing_percent=1.0,
            stop_loss_amount=1.0,
            leverage=10
        )

        # Загрузка конфигурации биржи
        with open('config/exchange_config.json', 'r') as f:
            exchange_config = json.load(f)

        self.order_manager = OrderManager(exchange_config)
        self.last_trade_time = {}
        self.min_time_between_trades = timedelta(hours=1)
        self.required_confirmations = 3
        self.signal_history = {}

    def update_trade_setup(self):
        """Обновление торговых настроек из параметров"""
        params = self.parameter_system.get_trading_parameters()
        self.trade_setup = TradeSetup.from_parameters(params)
        logger.info(f"Trade setup updated: {self.trade_setup}")

    def calculate_position_size(self, market_data: Dict) -> float:
        """Расчет размера позиции с учетом рыночных условий"""
        base_size = self.trade_setup.position_size

        # Получаем рекомендации системы
        recommendations = self.parameter_system.get_parameter_recommendations()
        market_condition = recommendations["market_condition"]

        # Корректировка размера позиции
        if market_condition["trend"] == "bullish":
            # В бычьем тренде можем немного увеличить позицию
            return base_size * 1.1
        elif market_condition["trend"] == "bearish":
            # В медвежьем тренде уменьшаем риск
            return base_size * 0.9

        return base_size

    def calculate_take_profit(self, entry_price: float, market_data: Dict) -> Dict:
        """Расчет тейк-профита с учетом рыночных условий"""
        base_tp_percent = self.trade_setup.take_profit_percent
        trailing_percent = self.trade_setup.trailing_percent

        # Получаем рекомендации системы
        recommendations = self.parameter_system.get_parameter_recommendations()
        market_condition = recommendations["market_condition"]

        # Корректировка тейк-профита
        if float(market_condition["volatility"]) > 0.7:
            # При высокой волатильности увеличиваем цель
            tp_percent = base_tp_percent * 1.2
            trailing = trailing_percent * 1.2
        else:
            tp_percent = base_tp_percent
            trailing = trailing_percent

        take_profit_price = entry_price * (1 + tp_percent / 100)

        return {
            "price": take_profit_price,
            "trailing_percent": trailing
        }

    def calculate_stop_loss(self, entry_price: float, position_size: float) -> float:
        """Расчет стоп-лосса на основе фиксированной суммы"""
        # Стоп-лосс = Точка входа - (Фиксированная сумма / Размер позиции)
        stop_loss_percent = (self.trade_setup.stop_loss_amount / position_size) * 100
        return entry_price * (1 - stop_loss_percent / 100)

    def should_enter_trade(self, market_data: Dict) -> bool:
        """Проверка условий входа в сделку"""
        # Получаем адаптивные параметры
        params = self.parameter_system.get_adaptive_parameters(market_data)

        # Проверяем основные индикаторы
        if market_data.get('rsi', 50) < params.rsi_oversold:
            return False

        if market_data.get('rsi', 50) > params.rsi_overbought:
            return False

        if market_data.get('volume_ratio', 1.0) < params.volume_filter:
            return False

        # Проверяем полосы Боллинджера
        if market_data.get('bb_position', 0.5) > params.bb_upper_threshold:
            return False

        if market_data.get('bb_position', 0.5) < params.bb_lower_threshold:
            return False

        # Проверяем MACD
        if abs(market_data.get('macd', 0)) < params.macd_threshold:
            return False

        return True

    def process_market_update(self, market_data: Dict) -> Dict:
        """Обработка обновления рыночных данных"""
        # Обновляем настройки торговли
        self.update_trade_setup()

        # Проверяем условия для входа
        if not self.should_enter_trade(market_data):
            return {"action": "wait", "reason": "Условия входа не выполнены"}

        # Рассчитываем параметры сделки
        position_size = self.calculate_position_size(market_data)
        entry_price = market_data.get('current_price', 0)

        if entry_price == 0:
            return {"action": "error", "reason": "Неверная цена входа"}

        # Рассчитываем тейк-профит и стоп-лосс
        take_profit = self.calculate_take_profit(entry_price, market_data)
        stop_loss = self.calculate_stop_loss(entry_price, position_size)

        return {
            "action": "enter_trade",
            "setup": {
                "position_size": position_size,
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "leverage": self.trade_setup.leverage
            }
        }
