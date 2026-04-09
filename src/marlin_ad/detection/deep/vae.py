from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marlin_ad.detection.base import BaseDetector
from marlin_ad.types.protocols import DetectionResult


@dataclass
class VariationalAutoencoderDetector(BaseDetector):
    """Placeholder VAE detector.

    This implementation is a stub intended to be replaced with a true VAE under optional extras.
    """

    def fit(self, X: Any, y: Any | None = None) -> "VariationalAutoencoderDetector":
        self._is_fitted = True
        return self

    def score(self, X: Any) -> DetectionResult:
        self._check_is_fitted()
        raise NotImplementedError("VAE detector requires a deep learning backend.")
