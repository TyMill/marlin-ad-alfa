"""Isolation Forest detector with deterministic, metadata-rich outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from marlin_ad.detection.base import BaseDetector
from marlin_ad.detection.scoring import score_threshold
from marlin_ad.types.errors import DataError
from marlin_ad.types.protocols import DetectionResult, MetadataValue
from marlin_ad.types.validation import ensure_2d_array, validate_quantile


@dataclass
class IsolationForestDetector(BaseDetector):
    """Isolation Forest detector for multivariate maritime anomaly detection.

    Notes
    -----
    The score orientation is standardized to the MARLIN-AD convention where
    **higher score means more anomalous**.
    """

    n_estimators: int = 200
    contamination: float | str = "auto"
    random_state: int = 42
    quantile: float = 0.95
    _n_features: int | None = field(init=False, default=None)
    _reference_size: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise DataError(f"n_estimators must be > 0. Got {self.n_estimators}.")
        validate_quantile(self.quantile)
        if isinstance(self.contamination, str):
            if self.contamination != "auto":
                raise DataError(
                    "contamination must be 'auto' or a float in the interval (0, 0.5]."
                )
        else:
            if not 0.0 < float(self.contamination) <= 0.5:
                raise DataError(
                    f"contamination must be in (0, 0.5]. Got {self.contamination}."
                )

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any | None = None) -> "IsolationForestDetector":
        """Fit the Isolation Forest detector on reference data."""

        arr = ensure_2d_array(X, name="X")
        self._model.fit(arr)
        self._n_features = arr.shape[1]
        self._reference_size = arr.shape[0]
        self._is_fitted = True
        return self

    def score(self, X: Any) -> DetectionResult:
        """Score current data and generate labels via quantile thresholding."""

        self._check_is_fitted()
        arr = ensure_2d_array(X, name="X")
        if self._n_features is not None and arr.shape[1] != self._n_features:
            raise DataError(
                f"Expected {self._n_features} features, got {arr.shape[1]} features."
            )

        raw_decision = self._model.decision_function(arr)
        scores = -np.asarray(raw_decision, dtype=float)
        threshold = score_threshold(scores, quantile=self.quantile)
        labels = (scores >= threshold).astype(int)

        metadata: dict[str, MetadataValue] = {
            "method": "isolation_forest",
            "score_orientation": "higher_is_more_anomalous",
            "thresholding": {
                "kind": "quantile",
                "quantile": self.quantile,
                "threshold": threshold,
            },
            "model": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "random_state": self.random_state,
            },
            "data": {
                "n_features": arr.shape[1],
                "n_samples_scored": arr.shape[0],
                "reference_size": self._reference_size,
            },
            "score_summary": {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
            },
        }
        return DetectionResult(scores=scores, labels=labels, metadata=metadata)
