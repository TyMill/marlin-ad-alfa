"""Robust statistics based detectors for maritime anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from marlin_ad.detection.base import BaseDetector
from marlin_ad.types.errors import DataError
from marlin_ad.types.protocols import DetectionResult, MetadataValue
from marlin_ad.types.validation import ensure_2d_array


_MAD_SCALING_CONSTANT: float = 0.6744897501960817


@dataclass
class RobustZScoreDetector(BaseDetector):
    """MAD-based robust z-score detector for tabular maritime signals.

    The detector estimates per-feature robust location and scale from reference
    data, then computes sample-wise anomaly scores as the maximum absolute robust
    z-score across features.

    Score orientation follows MARLIN-AD convention: higher is more anomalous.
    """

    threshold: float = 3.5
    epsilon: float = 1e-12
    _n_features: int | None = field(init=False, default=None)
    _reference_size: int | None = field(init=False, default=None)
    _median: npt.NDArray[np.float64] | None = field(init=False, default=None, repr=False)
    _mad: npt.NDArray[np.float64] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.threshold <= 0.0:
            raise DataError(f"threshold must be > 0. Got {self.threshold}.")
        if self.epsilon <= 0.0:
            raise DataError(f"epsilon must be > 0. Got {self.epsilon}.")

    def fit(self, X: Any, y: Any | None = None) -> "RobustZScoreDetector":
        """Fit robust location (median) and scale (MAD) from reference data."""

        arr = ensure_2d_array(X, name="X")
        median = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - median), axis=0)
        safe_mad = np.where(mad <= self.epsilon, self.epsilon, mad)

        self._median = median.astype(np.float64, copy=False)
        self._mad = safe_mad.astype(np.float64, copy=False)
        self._n_features = arr.shape[1]
        self._reference_size = arr.shape[0]
        self._is_fitted = True
        return self

    def score(self, X: Any) -> DetectionResult:
        """Score data with robust z-scores and generate binary labels."""

        self._check_is_fitted()
        arr = ensure_2d_array(X, name="X")
        if self._n_features is not None and arr.shape[1] != self._n_features:
            raise DataError(
                f"Expected {self._n_features} features, got {arr.shape[1]} features."
            )
        if self._median is None or self._mad is None:
            raise DataError("Detector has not been fitted with valid robust statistics.")

        robust_z = _MAD_SCALING_CONSTANT * np.abs(arr - self._median) / self._mad
        scores = np.max(robust_z, axis=1)
        labels = (scores >= self.threshold).astype(int)

        metadata: dict[str, MetadataValue] = {
            "method": "robust_zscore",
            "score_orientation": "higher_is_more_anomalous",
            "thresholding": {
                "kind": "fixed",
                "threshold": self.threshold,
            },
            "estimation": {
                "mad_scaling_constant": _MAD_SCALING_CONSTANT,
                "epsilon": self.epsilon,
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
        explanations: dict[str, npt.NDArray[np.float64]] = {
            "feature_scores": robust_z.astype(np.float64, copy=False)
        }

        return DetectionResult(
            scores=scores.astype(np.float64, copy=False),
            labels=labels,
            metadata=metadata,
            explanations=explanations,
        )
