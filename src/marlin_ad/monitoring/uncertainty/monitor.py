from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.protocols import MonitorResult


class UncertaintyBatch(TypedDict):
    """Expected payload for future uncertainty monitoring checks."""

    predicted_mean: list[float]
    predicted_std: list[float]


@dataclass
class UncertaintyMonitor(BaseMonitor):
    """Structured V1 stub for predictive uncertainty health monitoring."""

    def fit(self, reference: UncertaintyBatch) -> "UncertaintyMonitor":
        _ = reference
        self._is_fitted = True
        return self

    def evaluate(self, current: UncertaintyBatch) -> MonitorResult:
        self._check_is_fitted()
        _ = current
        raise NotImplementedError(
            "UncertaintyMonitor is a V1 stub. Expected payload keys are "
            "{'predicted_mean', 'predicted_std'} for fit/evaluate payloads."
        )
