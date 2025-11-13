"""Thin wrapper around the official PyBit unified trading client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pybit.unified_trading import HTTP

from bybit_bot.api.exceptions import (
    BybitConfigurationError,
    BybitRequestError,
    BybitValidationError,
)
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrderRequest:
    """Normalized order request payload for Bybit."""

    symbol: str
    side: str  # Buy / Sell
    order_type: str  # Market / Limit
    qty: float
    price: Optional[float] = None
    time_in_force: str = "GoodTillCancel"
    reduce_only: bool = False
    close_on_trigger: bool = False
    position_idx: int = 0  # 0 default, 1 long, 2 short
    category: str = "linear"  # linear (USDT perp), inverse, option

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "category": self.category,
            "symbol": self.symbol,
            "side": self.side.title(),
            "orderType": self.order_type.title(),
            "qty": str(self.qty),
            "timeInForce": self.time_in_force,
            "reduceOnly": self.reduce_only,
            "closeOnTrigger": self.close_on_trigger,
            "positionIdx": self.position_idx,
        }
        if self.price is not None:
            payload["price"] = str(self.price)
        return payload


class BybitClient:
    """High-level client encapsulating PyBit usage and error handling."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        testnet: bool = False,
        recv_window: int = 5000,
    ) -> None:
        api_key = api_key or settings.bybit_api_key
        api_secret = api_secret or settings.bybit_api_secret

        if not api_key or not api_secret:
            raise BybitConfigurationError("Bybit API key/secret must be provided")

        logger.debug("Initializing Bybit HTTP client (testnet=%s)", testnet)
        self._client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            recv_window=recv_window,
        )

    # --------------------------------------------------------------------- #
    # Utility methods
    # --------------------------------------------------------------------- #
    def _handle_response(self, response: Dict[str, Any], *, context: str) -> Dict[str, Any]:
        ret_code = response.get("retCode")
        if ret_code != 0:
            message = response.get("retMsg", "Unknown error")
            logger.error("Bybit request failed (%s): %s", context, message)
            raise BybitRequestError(
                f"{context} failed: {message}",
                status_code=ret_code,
                payload=response,
            )
        return response.get("result", {})

    def ping(self) -> bool:
        """Check connectivity by fetching server time."""
        try:
            response = self._client.get_server_time()
            _ = self._handle_response(response, context="get_server_time")
            return True
        except Exception as exc:  # noqa: BLE001 - surface all errors
            logger.exception("Bybit ping failed: %s", exc)
            return False

    # --------------------------------------------------------------------- #
    # Account endpoints
    # --------------------------------------------------------------------- #
    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        logger.debug("Fetching wallet balance for account type %s", account_type)
        response = self._client.get_wallet_balance(accountType=account_type)
        return self._handle_response(response, context="get_wallet_balance")

    def get_positions(self, category: str = "linear", symbol: str | None = None) -> Dict[str, Any]:
        logger.debug("Fetching positions (category=%s, symbol=%s)", category, symbol)
        params: Dict[str, Any] = {"category": category}
        if symbol:
            params["symbol"] = symbol
        response = self._client.get_positions(**params)
        return self._handle_response(response, context="get_positions")

    def get_history_orders(
        self,
        category: str = "linear",
        limit: int = 50,
        symbol: str | None = None,
    ) -> Dict[str, Any]:
        logger.debug("Fetching order history (category=%s, symbol=%s)", category, symbol)
        params: Dict[str, Any] = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        response = self._client.get_order_history(**params)
        return self._handle_response(response, context="get_order_history")

    # --------------------------------------------------------------------- #
    # Trading endpoints
    # --------------------------------------------------------------------- #
    def place_order(self, request: OrderRequest) -> Dict[str, Any]:
        payload = request.to_payload()
        logger.info("Placing order: %s", {k: v for k, v in payload.items() if k != "price"})
        try:
            response = self._client.place_order(**payload)
        except ValueError as exc:  # pybit raises ValueError for validation issues
            logger.error("Order validation error: %s", exc)
            raise BybitValidationError(str(exc)) from exc
        return self._handle_response(response, context="place_order")

    def cancel_order(
        self,
        order_id: str | None = None,
        *,
        category: str = "linear",
        symbol: str,
        client_order_id: str | None = None,
    ) -> Dict[str, Any]:
        if not order_id and not client_order_id:
            raise BybitValidationError("Either order_id or client_order_id must be provided")

        payload: Dict[str, Any] = {"category": category, "symbol": symbol}
        if order_id:
            payload["orderId"] = order_id
        if client_order_id:
            payload["orderLinkId"] = client_order_id

        logger.info("Cancelling order: %s", payload)
        response = self._client.cancel_order(**payload)
        return self._handle_response(response, context="cancel_order")

    def close_position(self, symbol: str, qty: float, side: str, category: str = "linear") -> Dict[str, Any]:
        """Close position by placing an opposite market order."""
        side_norm = side.title()
        closing_side = "Sell" if side_norm == "Buy" else "Buy"
        payload = {
            "category": category,
            "symbol": symbol,
            "side": closing_side,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": True,
        }
        logger.info("Closing position: %s", payload)
        response = self._client.place_order(**payload)
        return self._handle_response(response, context="close_position")

    def set_leverage(self, symbol: str, buy_leverage: int, sell_leverage: int, category: str = "linear") -> Dict[str, Any]:
        if buy_leverage <= 0 or sell_leverage <= 0:
            raise BybitValidationError("Leverage must be positive")
        payload = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        }
        logger.info("Setting leverage: %s", payload)
        response = self._client.set_leverage(**payload)
        return self._handle_response(response, context="set_leverage")

    def set_trading_stop(
        self,
        symbol: str,
        category: str = "linear",
        stop_loss: float | None = None,
        take_profit: float | None = None,
        position_idx: int = 0,
        sl_trigger_by: str = "LastPrice",
        tp_trigger_by: str = "LastPrice",
    ) -> Dict[str, Any]:
        """Set stop loss and/or take profit for a position."""
        payload: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "positionIdx": position_idx,
        }
        if stop_loss is not None:
            payload["stopLoss"] = str(stop_loss)
            payload["slTriggerBy"] = sl_trigger_by
        if take_profit is not None:
            payload["takeProfit"] = str(take_profit)
            payload["tpTriggerBy"] = tp_trigger_by
        
        logger.info("Setting trading stop: %s", {k: v for k, v in payload.items() if k in ["symbol", "stopLoss", "takeProfit"]})
        response = self._client.set_trading_stop(**payload)
        return self._handle_response(response, context="set_trading_stop")

    # --------------------------------------------------------------------- #
    # Factory helper
    # --------------------------------------------------------------------- #
    @classmethod
    def from_settings(cls, *, testnet: bool = False) -> "BybitClient":
        return cls(testnet=testnet)

