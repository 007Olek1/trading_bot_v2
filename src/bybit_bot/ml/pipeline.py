"""Training pipeline coordinating feature engineering and ensemble modelling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bybit_bot.ml.features import FeatureConfig, compute_indicators
from bybit_bot.ml.models import EnsembleConfig, build_base_models

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingArtifacts:
    model: VotingClassifier
    feature_columns: Tuple[str, ...]
    meta: Dict[str, Any]


class EnsemblePipeline:
    """Coordinate feature engineering, model training, and inference."""

    def __init__(
        self,
        *,
        feature_config: FeatureConfig | None = None,
        ensemble_config: EnsembleConfig | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        learning_rule: str = "Disco57",
    ) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.test_size = test_size
        self.random_state = random_state
        self.learning_rule = learning_rule
        self._model: VotingClassifier | None = None
        self._feature_columns: Tuple[str, ...] | None = None
        self._estimator_names: Tuple[str, ...] | None = None
        self._weights: list[float] | None = None

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    def _build_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute indicators for each timeframe and concatenate them."""
        if not data:
            raise ValueError("No price data provided")
        base_label = "m15" if "m15" in data else next(iter(data))
        base_index = data[base_label].index
        feature_frames = []
        for timeframe, df in data.items():
            logger.debug("Computing features for timeframe %s (len=%s)", timeframe, len(df))
            prefix = f"{timeframe}_"
            indicators = compute_indicators(df, config=self.feature_config, prefix=prefix)
            if timeframe == base_label:
                aligned = indicators.reindex(base_index).ffill()
            else:
                aligned = indicators.reindex(base_index, method="ffill")
            feature_frames.append(aligned)
        features = pd.concat(feature_frames, axis=1).dropna()
        logger.debug("Final feature matrix shape: %s", features.shape)
        return features

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        price_data: Dict[str, pd.DataFrame],
        labels: pd.Series,
        *,
        sample_weights: np.ndarray | None = None,
    ) -> TrainingArtifacts:
        """Train ensemble on provided price data across timeframes."""
        if not price_data:
            raise ValueError("No price data provided for training")

        features = self._build_features(price_data)
        aligned_labels = labels.loc[features.index].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            aligned_labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=aligned_labels,
        )

        base_models = build_base_models(self.ensemble_config)
        self._estimator_names = tuple(base_models.keys())
        self._weights = [1.0 for _ in base_models]
        estimators = list(base_models.items())
        voting_clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", voting_clf),
            ]
        )

        logger.info("Training ensemble pipeline with %s features", X_train.shape[1])
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info("Validation report: %s", json.dumps(report, indent=2))

        self._model = pipeline
        self._feature_columns = tuple(features.columns)
        return TrainingArtifacts(
            model=pipeline,
            feature_columns=self._feature_columns,
            meta={"report": report, "learning_rule": self.learning_rule},
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(
        self,
        price_data: Dict[str, pd.DataFrame],
    ) -> np.ndarray:
        ensemble, _ = self.predict_with_components(price_data)
        return (ensemble.argmax(axis=1)).astype(int)

    def predict_proba(self, price_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        ensemble, _ = self.predict_with_components(price_data)
        return ensemble

    def predict_with_components(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self._model is None or self._feature_columns is None or self._weights is None:
            raise RuntimeError("Model not fitted. Call `fit` before `predict`.")

        features = self._build_features(price_data)
        missing = [col for col in self._feature_columns if col not in features.columns]
        if missing:
            features = features.reindex(columns=list(features.columns) + missing, fill_value=0.0)
        features = features.reindex(columns=list(self._feature_columns)).fillna(0)
        scaler = self._model.named_steps["scaler"]
        classifier: VotingClassifier = self._model.named_steps["classifier"]
        X_scaled = scaler.transform(features.to_numpy())
        ensemble_probs = classifier.predict_proba(X_scaled)

        component_probs: Dict[str, np.ndarray] = {}
        estimator_names = [name for name, _ in classifier.estimators]
        for name, estimator in zip(estimator_names, classifier.estimators_):
            if hasattr(estimator, "predict_proba"):
                component_probs[name] = estimator.predict_proba(X_scaled)
            else:
                decision = estimator.decision_function(X_scaled)
                probs = 1 / (1 + np.exp(-decision))
                component_probs[name] = np.vstack([1 - probs, probs]).T
        return ensemble_probs, component_probs

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, directory: Path) -> None:
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("Nothing to save. Train the model first.")
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "feature_columns": self._feature_columns,
                "feature_config": self.feature_config,
                "ensemble_config": self.ensemble_config,
                "learning_rule": self.learning_rule,
                "estimator_names": self._estimator_names,
                "weights": self._weights,
            },
            directory / "ensemble.joblib",
        )
        logger.info("Saved ensemble model to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "EnsemblePipeline":
        artifact = joblib.load(directory / "ensemble.joblib")
        pipeline = cls(
            feature_config=artifact["feature_config"],
            ensemble_config=artifact["ensemble_config"],
            learning_rule=artifact.get("learning_rule", "Disco57"),
        )
        pipeline._model = artifact["model"]
        pipeline._feature_columns = tuple(artifact["feature_columns"])
        pipeline._estimator_names = tuple(artifact.get("estimator_names", ()))
        pipeline._weights = list(artifact.get("weights", [1.0] * len(pipeline._estimator_names)))
        classifier: VotingClassifier = pipeline._model.named_steps["classifier"]
        if pipeline._weights:
            classifier.set_params(weights=pipeline._weights)
        return pipeline

    def get_weights(self) -> Dict[str, float]:
        if self._estimator_names is None or self._weights is None:
            raise RuntimeError("Model weights are unavailable before training.")
        return {name: weight for name, weight in zip(self._estimator_names, self._weights)}

    def set_weights(self, new_weights: Dict[str, float]) -> None:
        if self._estimator_names is None or self._weights is None or self._model is None:
            raise RuntimeError("Cannot set weights before training.")
        updated = []
        for name in self._estimator_names:
            updated.append(max(0.1, float(new_weights.get(name, 1.0))))
        self._weights = updated
        classifier: VotingClassifier = self._model.named_steps["classifier"]
        classifier.set_params(weights=self._weights)

