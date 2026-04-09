from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.errors import DataError
from marlin_ad.types.protocols import MonitorResult


@dataclass
class SelfTestMonitor(BaseMonitor):
    """Minimal health monitor for AI self-monitoring pipelines.

    The monitor validates that incoming prediction streams are finite and not
    severely under-populated relative to the fitted baseline.
    """

    min_sample_ratio: float = 0.5
    max_nonfinite_fraction: float = 0.0
    _reference_size: int = field(init=False, default=0, repr=False)

    def fit(self, reference: Any) -> "SelfTestMonitor":
        if not 0.0 <= self.max_nonfinite_fraction < 1.0:
            raise DataError("max_nonfinite_fraction must be in [0, 1).")
        if self.min_sample_ratio <= 0.0:
            raise DataError("min_sample_ratio must be > 0.")

        reference_values = np.asarray(reference, dtype=float).reshape(-1)
        if reference_values.size == 0:
            raise DataError("reference must contain at least one value.")
        self._reference_size = int(reference_values.size)
        self._is_fitted = True
        return self

    def evaluate(self, current: Any) -> MonitorResult:
        self._check_is_fitted()
        current_values = np.asarray(current, dtype=float).reshape(-1)
        if current_values.size == 0:
            raise DataError("current must contain at least one value.")

        finite_mask = np.isfinite(current_values)
        finite_fraction = float(np.mean(finite_mask))
        nonfinite_fraction = 1.0 - finite_fraction
        sample_ratio = float(current_values.size / max(self._reference_size, 1))

        alerts: list[str] = []
        if sample_ratio < self.min_sample_ratio:
            alerts.append("self_test:sample_volume")
        if nonfinite_fraction > self.max_nonfinite_fraction:
            alerts.append("self_test:non_finite_values")

        metrics = {
            "health.sample_ratio": sample_ratio,
            "health.nonfinite_fraction": nonfinite_fraction,
            "health.finite_fraction": finite_fraction,
            "summary.alert_count": float(len(alerts)),
        }
        metadata: Mapping[str, Any] = {
            "method": "stream_health_self_test",
            "reference_size": self._reference_size,
            "min_sample_ratio": self.min_sample_ratio,
            "max_nonfinite_fraction": self.max_nonfinite_fraction,
        }
        return MonitorResult(metrics=metrics, alerts=tuple(alerts), metadata=metadata)
