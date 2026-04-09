"""Base detector contract implementation."""

from __future__ import annotations

from typing import Any

from marlin_ad.types.errors import NotFittedError
from marlin_ad.types.protocols import DetectionResult


class BaseDetector:
    """Minimal base class for anomaly detectors."""

    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been successfully called."""
        return self._is_fitted

    def fit(self, X: Any, y: Any | None = None) -> "BaseDetector":
        """Fit detector to reference data.

        Subclasses may override this method with algorithm-specific logic.
        """
        self._is_fitted = True
        return self

    def score(self, X: Any) -> DetectionResult:
        """Compute anomaly scores for current data."""
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Detector has not been fitted. Call fit() before score().")
