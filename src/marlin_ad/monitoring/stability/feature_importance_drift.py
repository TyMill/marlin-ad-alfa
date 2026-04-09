from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.errors import DataError
from marlin_ad.types.protocols import MonitorResult
from marlin_ad.types.validation import ensure_1d_array


@dataclass
class FeatureImportanceDriftMonitor(BaseMonitor):
    """Track drift in feature importance vectors."""

    threshold: float = 0.2
    _reference: npt.NDArray[np.floating] | None = None

    def fit(self, reference: Any) -> "FeatureImportanceDriftMonitor":
        self._reference = ensure_1d_array(reference, name="reference_importance")
        self._is_fitted = True
        return self

    def evaluate(self, current: Any) -> MonitorResult:
        self._check_is_fitted()
        if self._reference is None:
            raise DataError("Reference importance missing; call fit() first.")
        current_arr = ensure_1d_array(current, name="current_importance")
        if current_arr.shape != self._reference.shape:
            raise DataError("Current importance vector has different shape than reference.")
        delta = np.abs(current_arr - self._reference)
        max_delta = float(np.max(delta))
        metrics = {"max_importance_shift": max_delta}
        alerts = ["feature_importance_drift"] if max_delta > self.threshold else []
        meta: Mapping[str, Any] = {"method": "importance_shift", "threshold": self.threshold}
        return MonitorResult(metrics=metrics, alerts=alerts, metadata=meta)
