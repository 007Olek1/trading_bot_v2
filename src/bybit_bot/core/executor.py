"""Trading executor orchestrating signal, risk, and API actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

from bybit_bot.api.client import BybitClient, OrderRequest
from bybit_bot.core.risk import RiskManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionContext:
    symbol: str
    category: str = "linear"
    leverage: int = 10


class TradeExecutor:
    def __init__(
        self,
        client: BybitClient,
        risk_manager: RiskManager,
        context: ExecutionContext,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.context = context

    def sync_leverage(self) -> None:
        logger.info("Sync leverage for %s to x%s", self.context.symbol, self.context.leverage)
        self.client.set_leverage(self.context.symbol, self.context.leverage, self.context.leverage, self.context.category)

    def close_position(self, position: Dict[str, object]) -> Dict[str, str] | None:
        size = float(position.get("size", 0))
        if size <= 0:
            return None
        side = position.get("side", "Sell")
        logger.info("Closing position %s size %s", position.get("symbol"), size)
        response = self.client.close_position(
            symbol=self.context.symbol,
            qty=size,
            side=side,
            category=self.context.category,
        )
        return {
            "orderId": response.get("orderId", ""),
            "symbol": self.context.symbol,
            "side": "CLOSE",
            "size": size,
            "category": self.context.category,
        }

    def execute_signal(
        self,
        signal: str,
        balance_info: Dict[str, float],
        open_positions_count: int,
    ) -> Dict[str, str] | None:
        if signal not in {"BUY", "SELL"}:
            logger.debug("No actionable signal (%s)", signal)
            return None

        if not self.risk_manager.can_open_position(open_positions_count):
            logger.warning("Max concurrent positions reached: %s", open_positions_count)
            return None

        balance = balance_info.get("availableBalance", 0)
        size = self.risk_manager.position_size(balance)
        side = "Buy" if signal == "BUY" else "Sell"

        order = OrderRequest(
            symbol=self.context.symbol,
            side=side,
            order_type="Market",
            qty=size,
            category=self.context.category,
        )
        logger.info("Executing %s order with size %s USDT", signal, size)
        response = self.client.place_order(order)
        
        # 🛑 АВТОМАТИЧЕСКАЯ УСТАНОВКА SL/TP ПОСЛЕ ОТКРЫТИЯ ПОЗИЦИИ
        try:
            # Небольшая задержка для появления позиции на бирже
            import time
            time.sleep(0.5)
            
            # Получаем цену входа из позиции (с retry)
            current_position = None
            for attempt in range(3):  # 3 попытки
                positions = self.client.get_positions(category=self.context.category, symbol=self.context.symbol)
                position_list = positions.get("list", [])
                for pos in position_list:
                    pos_size = float(pos.get("size", 0) or 0)
                    if pos_size > 0:
                        current_position = pos
                        break
                if current_position:
                    break
                if attempt < 2:
                    time.sleep(0.3)  # Небольшая задержка перед повтором
            
            if current_position:
                entry_price = float(current_position.get("avgPrice", 0) or current_position.get("entryPrice", 0))
                if entry_price > 0:
                    # Рассчитываем SL и TP
                    position_notional = size * self.context.leverage  # $1 × 10x = $10
                    max_sl_usd = 1.0  # Максимальный SL: -$1
                    sl_percent = (max_sl_usd / position_notional) * 100  # ~10%
                    tp_percent = 1.0  # TP: +1% первый уровень
                    
                    if side == "Buy":
                        stop_loss_price = entry_price * (1 - sl_percent / 100.0)
                        take_profit_price = entry_price * (1 + tp_percent / 100.0)
                    else:  # Sell
                        stop_loss_price = entry_price * (1 + sl_percent / 100.0)
                        take_profit_price = entry_price * (1 - tp_percent / 100.0)
                    
                    # Устанавливаем SL/TP
                    self.client.set_trading_stop(
                        symbol=self.context.symbol,
                        category=self.context.category,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                    )
                    logger.info("✅ SL/TP установлены: SL=%.6f (%.2f%%), TP=%.6f (+%.2f%%)", 
                              stop_loss_price, sl_percent, take_profit_price, tp_percent)
        except Exception as e:
            logger.warning("⚠️ Не удалось установить SL/TP автоматически: %s", e)
        
        return {
            "orderId": response.get("orderId", ""),
            "symbol": self.context.symbol,
            "side": side.upper(),
            "size": size,
            "category": self.context.category,
        }

