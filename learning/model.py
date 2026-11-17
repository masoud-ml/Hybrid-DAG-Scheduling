from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Protocol, Optional
from sklearn.ensemble import RandomForestClassifier


class ProbabilisticBinaryClassifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilisticBinaryClassifier": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RFModelConfig:
    n_estimators: int = 100
    rf_seed: int = 123
    n_jobs: int = -1


class RFBinaryClassifier:
    """Random Forest binary classifier."""
    def __init__(self, cfg: RFModelConfig):
        self.cfg = cfg
        self._model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.rf_seed,
            n_jobs=cfg.n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFBinaryClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def sklearn_model(self) -> RandomForestClassifier:
        return self._model


def build_default_model(
    *,
    model_name: str = "rf",
    rf_cfg: Optional[RFModelConfig] = None,
) -> ProbabilisticBinaryClassifier:
    if model_name == "rf":
        return RFBinaryClassifier(rf_cfg or RFModelConfig())

    raise ValueError(f"Unknown model_name: {model_name}")
