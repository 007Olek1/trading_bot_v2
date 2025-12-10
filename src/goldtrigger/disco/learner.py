#!/usr/bin/env python3
"""
Disco57 (DiscoRL) - Адаптивная система обучения
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TradeFeatures:
    ema_trend: float
    momentum: float
    volume_ratio: float
    rsi: float
    spread: float
    volatility: float
    price_position: float
    candle_strength: float
    stop_speed: float = 0.0
    book_imbalance: float = 0.5
    book_delta: float = 0.5


class Disco57Learner:
    def __init__(self, model_path: str = "/opt/GoldTrigger_bot/data/disco57_model.json"):
        self.model_path = model_path
        self.learning_rate = 0.05
        self.min_confidence = 0.6
        self.weights = {
            "ema_trend": 0.5,
            "momentum": 0.5,
            "volume_ratio": 0.5,
            "rsi": 0.5,
            "spread": 0.5,
            "volatility": 0.5,
            "price_position": 0.5,
            "candle_strength": 0.5,
            "stop_speed": 0.4,
            "book_imbalance": 0.4,
            "book_delta": 0.4,
        }
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_history: List[Dict] = []
        self.load_model()

    def load_model(self):
        try:
            path = Path(self.model_path)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                self.weights = data.get("weights", self.weights)
                self.total_trades = data.get("total_trades", 0)
                self.winning_trades = data.get("winning_trades", 0)
                self.total_pnl = data.get("total_pnl", 0.0)
                self.trade_history = data.get("trade_history", [])[-100:]
                logger.info("Disco57 модель загружена: %s сделок", self.total_trades)
        except Exception as exc:
            logger.warning("Не удалось загрузить модель Disco57: %s", exc)

    def save_model(self):
        try:
            path = Path(self.model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "weights": self.weights,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "total_pnl": self.total_pnl,
                "trade_history": self.trade_history[-100:],
                "updated_at": time.time(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.error("Ошибка сохранения модели Disco57: %s", exc)

    def predict(self, features: TradeFeatures, direction: str) -> tuple[bool, float]:
        score = 0.0
        if direction == "long":
            score += self.weights["ema_trend"] * (1 if features.ema_trend > 0 else 0)
            score += self.weights["momentum"] * (1 if features.momentum > 0.005 else 0)
            score += self.weights["volume_ratio"] * features.volume_ratio
            score += self.weights["rsi"] * (1 if 0.3 < features.rsi < 0.7 else 0)
            score += self.weights["spread"] * (1 - features.spread)
            score += self.weights["volatility"] * features.volatility
            score += self.weights["price_position"] * (1 - features.price_position)
            score += self.weights["candle_strength"] * features.candle_strength
            score += self.weights["book_imbalance"] * features.book_imbalance
            score += self.weights["book_delta"] * features.book_delta
        else:
            score += self.weights["ema_trend"] * (1 if features.ema_trend < 0 else 0)
            score += self.weights["momentum"] * (1 if features.momentum < -0.005 else 0)
            score += self.weights["volume_ratio"] * features.volume_ratio
            score += self.weights["rsi"] * (1 if 0.3 < features.rsi < 0.7 else 0)
            score += self.weights["spread"] * (1 - features.spread)
            score += self.weights["volatility"] * features.volatility
            score += self.weights["price_position"] * features.price_position
            score += self.weights["candle_strength"] * features.candle_strength
            score += self.weights["book_imbalance"] * (1 - features.book_imbalance)
            score += self.weights["book_delta"] * (1 - features.book_delta)

        stop_component = 1 - max(0.0, min(features.stop_speed, 1.0))
        score += self.weights["stop_speed"] * stop_component

        max_score = sum(self.weights.values())
        confidence = score / max_score if max_score > 0 else 0.5
        allow = confidence >= self.min_confidence
        return allow, confidence

    def learn(self, features: TradeFeatures, direction: str, pnl: float):
        self.total_trades += 1
        self.total_pnl += pnl
        reward = 1.0 if pnl > 0 else -1.0
        if pnl > 0:
            self.winning_trades += 1
        feature_dict = asdict(features)
        for key in self.weights:
            if key not in feature_dict:
                continue
            value = feature_dict[key]
            if direction == "long":
                if key == "ema_trend":
                    value = 1 if value > 0 else 0
                elif key == "momentum":
                    value = 1 if value > 0.005 else 0
                elif key == "price_position":
                    value = 1 - value
                elif key == "stop_speed":
                    value = 1 - max(0.0, min(value, 1.0))
            else:
                if key == "ema_trend":
                    value = 1 if value < 0 else 0
                elif key == "momentum":
                    value = 1 if value < -0.005 else 0
                elif key == "book_imbalance":
                    value = 1 - value
                elif key == "book_delta":
                    value = 1 - value
                elif key == "stop_speed":
                    value = 1 - max(0.0, min(value, 1.0))
            adjustment = self.learning_rate * reward * value
            self.weights[key] = max(0.1, min(1.0, self.weights[key] + adjustment))

        self.trade_history.append(
            {"time": time.time(), "direction": direction, "pnl": pnl, "features": feature_dict}
        )
        self.save_model()

    def get_win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def get_stats(self) -> Dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.get_win_rate(),
            "total_pnl": self.total_pnl,
            "weights": self.weights,
        }
