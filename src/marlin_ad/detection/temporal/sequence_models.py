from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from marlin_ad.detection.base import BaseDetector
from marlin_ad.types.protocols import DetectionResult
from marlin_ad.types.validation import ensure_2d_array


@dataclass
class RollingZScoreDetector(BaseDetector):
    """Simple rolling z-score detector for sequence data."""

    window: int = 20
    threshold: float = 3.0

    def fit(self, X: Any, y: Any | None = None) -> "RollingZScoreDetector":
        arr = ensure_2d_array(X)
        self._mean = np.mean(arr, axis=0)
        self._std = np.std(arr, axis=0) + 1e-12
        self._is_fitted = True
        return self

    def score(self, X: Any) -> DetectionResult:
        self._check_is_fitted()
        arr = ensure_2d_array(X)
        z = np.abs((arr - self._mean) / self._std)
        scores = np.max(z, axis=1)
        labels = (scores >= self.threshold).astype(int)
        meta = {"method": "rolling_zscore", "window": self.window, "threshold": self.threshold}
        explanations: dict[str, npt.NDArray[np.floating]] = {"feature_scores": z}
        return DetectionResult(scores=scores, labels=labels, metadata=meta, explanations=explanations)
