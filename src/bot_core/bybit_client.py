from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from pybit.unified_trading import HTTP

from .config import BybitConfig

logger = logging.getLogger(__name__)

CATEGORY = "linear"


class BybitClientError(Exception):
    """Исключение верхнего уровня для ошибок клиента Bybit."""


class BybitClient:
    """Обёртка над pybit HTTP клиентом с удобными async-методами."""

    def __init__(self, config: BybitConfig):
        self._config = config
        self._timeout = float(os.getenv("BYBIT_API_TIMEOUT_SEC", "90"))
        self._http = HTTP(
            testnet=config.testnet,
            api_key=config.api_key,
            api_secret=config.api_secret,
        )
        self._symbol_filters: Dict[str, Dict[str, float]] = {}

    async def _call(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Запустить блокирующий pybit вызов в отдельном потоке с расширенным таймаутом."""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise BybitClientError(
                f"Bybit API call timed out after {self._timeout:.0f}s for {func.__name__}"
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise BybitClientError(str(exc)) from exc

        ret_code = response.get("retCode", 0)
        if ret_code != 0:
            raise BybitClientError(
                f"Bybit API error {ret_code}: {response.get('retMsg', 'Unknown error')}"
            )
        return response

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self._call(
            self._http.set_leverage,
            category=CATEGORY,
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )

    async def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
        resp = await self._call(
            self._http.get_kline,
            category=CATEGORY,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        return resp.get("result", {}).get("list", []) or []

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        resp = await self._call(
            self._http.get_tickers,
            category=CATEGORY,
            symbol=symbol,
        )
        tickers = resp.get("result", {}).get("list", [])
        if not tickers:
            raise BybitClientError(f"Ticker not found for {symbol}")
        return tickers[0]

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        resp = await self._call(
            self._http.get_positions,
            category=CATEGORY,
            settleCoin="USDT",
        )
        return resp.get("result", {}).get("list", []) or []

    async def fetch_account_balance(self) -> Dict[str, Any]:
        resp = await self._call(
            self._http.get_wallet_balance,
            accountType="UNIFIED",
            coin="USDT",
        )
        result = resp.get("result", {})
        list_data = result.get("list") or []
        return list_data[0] if list_data else {}

    async def create_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        position_idx: int,
        reduce_only: bool = False,
        order_type: str = "Market",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": f"{qty:.8f}",
            "timeInForce": "GoodTillCancel",
            "reduceOnly": reduce_only,
            "positionIdx": position_idx,
        }
        if price is not None:
            params["price"] = f"{price:.8f}"
        return await self._call(self._http.place_order, **params)

    async def close_position(self, symbol: str, side: str, qty: float, position_idx: int) -> Dict[str, Any]:
        opposite_side = "Sell" if side.lower() == "long" else "Buy"
        return await self.create_order(
            symbol=symbol,
            side=opposite_side,
            qty=qty,
            position_idx=position_idx,
            reduce_only=True,
        )

    async def set_trading_stop(
        self,
        symbol: str,
        position_idx: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "positionIdx": position_idx,
        }
        if stop_loss is not None:
            params["stopLoss"] = f"{stop_loss:.8f}"
        if take_profit is not None:
            params["takeProfit"] = f"{take_profit:.8f}"
        if trailing_stop is not None:
            params["trailingStop"] = f"{trailing_stop:.8f}"

        return await self._call(self._http.set_trading_stop, **params)

    async def get_symbol_filters(self, symbol: str) -> Dict[str, float]:
        symbol_key = symbol.upper()
        cached = self._symbol_filters.get(symbol_key)
        if cached:
            return cached

        resp = await self._call(
            self._http.get_instruments_info,
            category=CATEGORY,
            symbol=symbol,
        )
        instruments = resp.get("result", {}).get("list", [])
        if not instruments:
            raise BybitClientError(f"Instruments info not found for {symbol}")
        info = instruments[0]
        lot_filter = info.get("lotSizeFilter", {}) or {}
        price_filter = info.get("priceFilter", {}) or {}
        filters = {
            "qty_step": float(lot_filter.get("qtyStep") or 0.0),
            "min_qty": float(lot_filter.get("minOrderQty") or 0.0),
            "min_notional": float(lot_filter.get("minNotionalValue") or 0.0),
            "price_tick": float(price_filter.get("tickSize") or 0.0),
        }
        self._symbol_filters[symbol_key] = filters
        return filters

    async def normalize_quantity(self, symbol: str, qty: float) -> Tuple[float, Dict[str, float]]:
        filters = await self.get_symbol_filters(symbol)
        step = filters.get("qty_step") or 0.0
        min_qty = filters.get("min_qty") or 0.0
        if step <= 0:
            return qty, filters
        rounded = math.floor(qty / step) * step
        rounded = float(f"{rounded:.8f}")
        if rounded < min_qty:
            return 0.0, filters
        return rounded, filters


__all__ = ["BybitClient", "BybitClientError", "CATEGORY"]
