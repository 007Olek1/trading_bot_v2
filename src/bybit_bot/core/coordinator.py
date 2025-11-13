"""Coordinator tying together data ingestion, ML predictions, and execution."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from bybit_bot.api.client import BybitClient
from bybit_bot.core.executor import ExecutionContext, TradeExecutor
from bybit_bot.core.risk import RiskConfig, RiskManager
from bybit_bot.core.signals import SignalConfig, SignalGenerator
from bybit_bot.core.adaptation import AdaptationConfig, AdaptationManager
from bybit_bot.core.scanner import MarketScanner, ScannerConfig
from bybit_bot.core.journal import TradeJournal
from bybit_bot.data.provider import MarketDataProvider
from bybit_bot.ml.pipeline import EnsemblePipeline

logger = logging.getLogger(__name__)


class TradingCoordinator:
    def __init__(
        self,
        client: BybitClient,
        pipeline: EnsemblePipeline,
        signal_config: SignalConfig | None = None,
        risk_config: RiskConfig | None = None,
        adaptation_config: AdaptationConfig | None = None,
        data_provider: MarketDataProvider | None = None,
        watchlist: Sequence[str] | None = None,
        analysis_interval: str = "15m",
        monitoring_interval: str = "1m",
        journal: TradeJournal | None = None,
        analysis_dir: Path | None = None,
        symbol: str = "BTCUSDT",
        category: str = "linear",
        server_host: str = "185.70.199.244",
    ) -> None:
        self.client = client
        self.pipeline = pipeline
        self.signal_generator = SignalGenerator(signal_config)
        self.risk_manager = RiskManager(risk_config)
        context = ExecutionContext(symbol=symbol, category=category, leverage=self.risk_manager.leverage())
        self.executor = TradeExecutor(client=client, risk_manager=self.risk_manager, context=context)
        self.adaptation_manager = AdaptationManager(
            pipeline=self.pipeline,
            signal_generator=self.signal_generator,
            risk_manager=self.risk_manager,
            config=adaptation_config or AdaptationConfig(),
        )
        self.data_provider = data_provider
        self.scanner = (
            MarketScanner(
                data_provider=self.data_provider,
                pipeline=self.pipeline,
                config=ScannerConfig(watchlist=tuple(watchlist) if watchlist else None),
            )
            if self.data_provider
            else None
        )
        self.symbol = symbol
        self.category = category
        self.server_host = server_host
        self.analysis_interval = analysis_interval
        self.monitoring_interval = monitoring_interval
        self.journal = journal
        self.analysis_dir = analysis_dir
        if self.analysis_dir:
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.active: bool = True
        self.last_signal: str = "HOLD"
        self.last_probabilities: np.ndarray | None = None
        self.last_execution: Dict[str, str] | None = None
        self.last_component_support: Dict[str, float] | None = None
        self.previous_total_equity: Optional[float] = None
        self.last_opportunities: list[dict] = []
        self._previous_positions: dict[str, dict] = {}

    def run_cycle(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, object] | None:
        if not self.active:
            logger.info("Trading is currently paused; skipping cycle")
            return None
        ensemble_probs, component_probs = self.pipeline.predict_with_components(price_data)
        self.last_probabilities = ensemble_probs
        component_support = {name: probs[-1][1] for name, probs in component_probs.items()}
        self.last_component_support = component_support
        signal = self.signal_generator.generate_signal(ensemble_probs, price_data)
        balance = self.client.get_wallet_balance()
        positions = self.client.get_positions(category=self.category, symbol=self.symbol)
        positions_map = self._normalize_positions(positions)
        closed_positions: list[dict] = []
        manual_closures = self._detect_manual_closures(positions_map)
        if manual_closures:
            closed_positions.extend(manual_closures)
        timeout_closures = self._enforce_time_limit(positions)
        if timeout_closures:
            closed_positions.extend(timeout_closures)
            for closure in timeout_closures:
                key = closure.get("position_key")
                if key and key in positions_map:
                    positions_map.pop(key)
            positions = self.client.get_positions(category=self.category, symbol=self.symbol)
            positions_map = self._normalize_positions(positions)
        open_positions_count = len(positions.get("list", []))
        balance_info = balance.get("list", [{}])[0]
        total_equity = float(balance_info.get("totalEquity", 0.0))
        logger.info("Signal generated: %s", signal)
        execution = self.executor.execute_signal(signal, balance_info, open_positions_count)
        self.last_signal = signal
        self.last_execution = execution
        execution_snapshot: dict | None = None
        if execution is not None:
            positions_after = self.client.get_positions(category=self.category, symbol=self.symbol)
            positions = positions_after
            positions_map = self._normalize_positions(positions_after)
            execution_snapshot = self._extract_position_snapshot(positions_after, execution.get("side", ""))
        if execution is not None and self.previous_total_equity is not None and component_support:
            profit = total_equity - self.previous_total_equity
            if profit != 0:
                self.adaptation_manager.record_trade(profit, signal, component_support)
            if self.journal:
                buy_prob = float(ensemble_probs[-1][1])
                sell_prob = float(ensemble_probs[-1][0])
                record = {
                    "order_id": execution.get("orderId", ""),
                    "symbol": execution.get("symbol", ""),
                    "side": execution.get("side", ""),
                    "size": execution.get("size", 0),
                    "probability_buy": buy_prob,
                    "probability_sell": sell_prob,
                    "confidence": max(buy_prob, sell_prob),
                    "threshold": self.signal_generator.config.probability_threshold,
                    "execution_context": f"{self.symbol}:{signal}",
                }
                self.journal.record(record)
            if self.analysis_dir:
                analysis_entry = {
                    "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                    "symbol": self.symbol,
                    "signal": signal,
                    "confidence": float(max(ensemble_probs[-1])),
                    "weights": self.pipeline.get_weights(),
                    "threshold": self.signal_generator.config.probability_threshold,
                }
                with (self.analysis_dir / "adaptation.log").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(analysis_entry) + "\n")
        self.previous_total_equity = total_equity
        if self.scanner:
            self.last_opportunities = self.scanner.rank()
        self._previous_positions = positions_map
        result: dict[str, object] = {
            "signal": signal,
            "execution": execution,
            "execution_snapshot": execution_snapshot,
            "closed_positions": closed_positions,
            "probabilities": ensemble_probs[-1].tolist() if len(ensemble_probs) else None,
            "component_support": component_support,
            "balance": balance_info,
            "risk_targets": {
                "tp": self.risk_manager.config.base_position_size_usd,
                "sl": -self.risk_manager.config.base_position_size_usd,
            },
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "learning_rule": getattr(self.pipeline, "learning_rule", "Disco57"),
        }
        return result

    def pause(self) -> None:
        logger.info("Trading paused by user")
        self.active = False

    def resume(self) -> None:
        logger.info("Trading resumed by user")
        self.active = True

    def get_balance(self) -> Dict[str, str | float]:
        balance = self.client.get_wallet_balance()
        return balance.get("list", [{}])[0]

    def get_positions(self) -> Dict[str, pd.DataFrame]:
        return self.client.get_positions(category=self.category, symbol=self.symbol)

    def get_history(self, limit: int = 50) -> Dict[str, pd.DataFrame]:
        return self.client.get_history_orders(limit=limit, category=self.category, symbol=self.symbol)

    def scan_opportunities(self, top_n: int = 5) -> list[dict]:
        if not self.scanner:
            return []
        self.last_opportunities = self.scanner.rank(top_n=top_n)
        return self.last_opportunities

    def status(self) -> Dict[str, object]:
        balance = self.get_balance()
        positions = self.get_positions()
        return {
            "active": self.active,
            "signal": self.last_signal,
            "probabilities": self.last_probabilities.tolist() if self.last_probabilities is not None else None,
            "balance": balance,
            "positions": positions.get("list", []),
            "symbol": self.symbol,
            "server_host": self.server_host,
            "leverage": self.risk_manager.leverage(),
            "threshold": self.signal_generator.config.probability_threshold,
            "model_weights": self.pipeline.get_weights() if self.last_probabilities is not None else None,
            "analysis_interval": self.analysis_interval,
            "monitoring_interval": self.monitoring_interval,
            "risk_targets": {
                "tp": self.risk_manager.config.base_position_size_usd,
                "sl": -self.risk_manager.config.base_position_size_usd,
            },
            "opportunities": self.last_opportunities,
            "max_positions": self.risk_manager.config.max_concurrent_positions,
        }

    def _enforce_time_limit(self, positions: Dict[str, object]) -> list[dict]:
        """Ensure positions do not remain open longer than 24 hours."""
        position_list = positions.get("list", []) if isinstance(positions, dict) else []
        cutoff_ms = 24 * 60 * 60 * 1000
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        closed: list[dict] = []
        for pos in position_list:
            created = int(pos.get("createdTime", pos.get("updatedTime", now_ms)))
            if now_ms - created > cutoff_ms:
                logger.info("Position %s exceeded 24h, closing.", pos.get("symbol"))
                result = self.executor.close_position(pos)
                if result:
                    closed.append(
                        self._build_closed_event(
                            pos,
                            reason="timeout",
                            execution=result,
                        )
                    )
        return closed

    def _normalize_positions(self, positions: Dict[str, object]) -> dict[str, dict]:
        position_list = positions.get("list", []) if isinstance(positions, dict) else []
        mapping: dict[str, dict] = {}
        for pos in position_list:
            key = self._position_key(pos)
            mapping[key] = pos
        return mapping

    def _detect_manual_closures(self, current_positions: dict[str, dict]) -> list[dict]:
        closed: list[dict] = []
        if not self._previous_positions:
            return closed
        for key, previous in self._previous_positions.items():
            if key not in current_positions:
                closed.append(self._build_closed_event(previous, reason="manual"))
        return closed

    def _position_key(self, position: dict[str, object]) -> str:
        symbol = str(position.get("symbol", "")).upper()
        side = str(position.get("side", "")).upper()
        return f"{symbol}:{side}"

    def _build_closed_event(
        self,
        position: dict[str, object],
        *,
        reason: str,
        execution: dict | None = None,
    ) -> dict:
        entry_price = float(position.get("avgPrice") or 0.0)
        exit_raw = position.get("lastPrice") or position.get("markPrice") or entry_price
        size = float(position.get("size") or 0.0)
        pnl = float(position.get("unrealisedPnl") or 0.0)
        event = {
            "reason": reason,
            "symbol": position.get("symbol", ""),
            "side": str(position.get("side", "")).upper(),
            "size": size,
            "entry_price": entry_price,
            "exit_price": float(exit_raw or 0.0),
            "pnl": pnl,
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "execution": execution,
            "position_key": self._position_key(position),
        }
        return event

    def _extract_position_snapshot(self, positions: Dict[str, object], side: str) -> dict | None:
        target_side = str(side).upper()
        for pos in positions.get("list", []) if isinstance(positions, dict) else []:
            if self._position_key(pos) == f"{self.symbol.upper()}:{target_side}":
                return pos
        return None

