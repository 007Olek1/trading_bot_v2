"""Model constructors for the ML ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier  # Excluded due to libomp dependency issues


@dataclass(slots=True)
class EnsembleConfig:
    """Configuration for ensemble model hyperparameters."""

    rf_params: Dict[str, int | float] = field(
        default_factory=lambda: {"n_estimators": 300, "max_depth": 8, "random_state": 42, "n_jobs": -1}
    )
    xgb_params: Dict[str, int | float] = field(default_factory=dict)  # Deprecated
    lgbm_params: Dict[str, int | float] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
    )
    svm_params: Dict[str, int | float] = field(
        default_factory=lambda: {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
        }
    )
    mlp_params: Dict[str, int | float | tuple[int, ...]] = field(
        default_factory=lambda: {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": 256,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "random_state": 42,
        }
    )


def build_base_models(config: EnsembleConfig | None = None):
    """Return dictionary of initialized base learners."""
    cfg = config or EnsembleConfig()
    models = {
        "random_forest": RandomForestClassifier(**cfg.rf_params),
        "lightgbm": LGBMClassifier(**cfg.lgbm_params),
        "svm": SVC(**cfg.svm_params),
        "neural_net": MLPClassifier(**cfg.mlp_params),
    }
    return models

