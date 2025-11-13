"""Backtest trading strategy using stored historical data."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from bybit_bot.ml.pipeline import EnsemblePipeline
from bybit_bot.ml.features import FeatureConfig
from bybit_bot.core.signals import SignalGenerator, SignalConfig
from bybit_bot.core.risk import RiskConfig, RiskManager

HISTORICAL_DIR = Path("data/historical")
RESULTS_DIR = Path("data/analysis")
MODEL_DIR = Path("models/ensemble")
FEE_RATE = 0.0006
MAX_HOLD = timedelta(hours=24)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


@dataclass(slots=True)
class BacktestParams:
    starting_balance: float = 1000.0
    max_positions: int = 3
    fee_rate: float = FEE_RATE


def load_pipeline() -> EnsemblePipeline:
    path = MODEL_DIR / "ensemble.joblib"
    if not path.exists():
        raise FileNotFoundError("Train ensemble first (scripts/train_ensemble.py)")
    return EnsemblePipeline.load(MODEL_DIR)


def load_dataset(file_path: Path) -> dict[str, pd.DataFrame]:
    with file_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    frames = {}
    for label, rows in raw.items():
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        frames[label] = df
    return frames


def run_backtest(file_path: Path, pipeline: EnsemblePipeline, params: BacktestParams) -> dict:
    data = load_dataset(file_path)
    base_index = data["15m"].index
    min_rows = max(
        pipeline.feature_config.atr_period,
        pipeline.feature_config.bb_period,
        pipeline.feature_config.adx_period if hasattr(pipeline.feature_config, "adx_period") else 14,
    ) + 10
    for label, df in data.items():
        if len(df) < min_rows:
            raise ValueError(f"Not enough candles for timeframe {label}; need at least {min_rows}")
    features = pipeline._build_features(data)  # pylint: disable=protected-access
    features = features.reindex(base_index).dropna()
    feature_columns = list(pipeline._feature_columns or features.columns)  # type: ignore[attr-defined]
    features = features.reindex(columns=feature_columns, fill_value=0.0)
    scaler = pipeline._model.named_steps["scaler"]  # type: ignore[attr-defined]
    classifier = pipeline._model.named_steps["classifier"]  # type: ignore[attr-defined]
    X_scaled = scaler.transform(features.to_numpy())
    ensemble_probs = classifier.predict_proba(X_scaled)

    signal_generator = SignalGenerator(SignalConfig())
    risk_manager = RiskManager(RiskConfig(max_concurrent_positions=params.max_positions))

    balance = params.starting_balance
    equity_curve = []
    positions = []
    trade_log = []

    for timestamp, prob in zip(features.index[min_rows:], ensemble_probs[min_rows:]):
        sample = {label: df.loc[:timestamp] for label, df in data.items()}
        signal = signal_generator.generate_signal(np.array([prob]), sample)

        # Close positions exceeding 24h
        still_open = []
        for trade in positions:
            age = timestamp - trade["entry_time"]
            price = float(data["15m"].loc[timestamp, "close"])
            pnl = (price - trade["entry_price"]) * trade["direction"] * trade["size"]
            if age >= MAX_HOLD:
                balance += pnl - params.fee_rate * abs(trade["size"]) * abs(price)
                trade["exit_time"] = timestamp
                trade["exit_price"] = price
                trade["pnl"] = pnl
                trade_log.append(trade)
            else:
                still_open.append(trade)
        positions = still_open

        price = float(data["15m"].loc[timestamp, "close"])
        if signal in {"BUY", "SELL"} and len(positions) < params.max_positions:
            direction = 1 if signal == "BUY" else -1
            size = risk_manager.position_size(balance)
            fee = params.fee_rate * size * price * 2
            if balance - fee <= 0:
                continue
            positions.append(
                {
                    "entry_time": timestamp,
                    "entry_price": price,
                    "direction": direction,
                    "size": size / price,  # convert to contracts (approx)
                    "signal": signal,
                }
            )
            balance -= fee

        equity = balance
        for trade in positions:
            price = float(data["15m"].loc[timestamp, "close"])
            pnl = (price - trade["entry_price"]) * trade["direction"] * trade["size"]
            equity += pnl
        equity_curve.append({"timestamp": timestamp, "equity": equity})

    equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
    returns = equity_df["equity"].pct_change().dropna()
    pnl = equity_df["equity"].iloc[-1] - params.starting_balance
    if not returns.empty and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)
    else:
        sharpe = 0.0
    drawdown = (equity_df["equity"].cummax() - equity_df["equity"]).max()

    result = {
        "file": file_path.name,
        "final_equity": equity_df["equity"].iloc[-1],
        "pnl": pnl,
        "sharpe": float(sharpe) if not math.isnan(sharpe) else 0.0,
        "max_drawdown": float(drawdown) if not math.isnan(drawdown) else 0.0,
        "trades": len(trade_log),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"backtest_{file_path.stem}.json"
    with output.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Backtest result saved to {output}")
    return result


def main() -> None:
    pipeline = load_pipeline()
    params = BacktestParams()
    files = sorted(HISTORICAL_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError("No historical files found, run scripts/fetch_data.py first.")
    summary = []
    for file_path in files:
        summary.append(run_backtest(file_path, pipeline, params))
    summary_path = RESULTS_DIR / f"backtest_summary_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

