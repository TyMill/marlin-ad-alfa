"""Base monitoring contract implementation."""

from __future__ import annotations

from typing import Any

from marlin_ad.types.errors import NotFittedError
from marlin_ad.types.protocols import MonitorResult


class BaseMonitor:
    """Minimal base class for model/data monitors."""

    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been successfully called."""
        return self._is_fitted

    def fit(self, reference: Any) -> "BaseMonitor":
        """Fit monitor to reference data."""
        self._is_fitted = True
        return self

    def evaluate(self, current: Any) -> MonitorResult:
        """Evaluate current data against fitted reference behavior."""
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Monitor has not been fitted. Call fit() before evaluate().")
