from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.protocols import MonitorResult


class CalibrationBatch(TypedDict):
    """Expected payload for future calibration checks."""

    y_true: list[int]
    y_prob: list[float]


@dataclass
class CalibrationMonitor(BaseMonitor):
    """Structured V1 stub for calibration monitoring.

    Planned V2 behavior: monitor expected calibration error and reliability-curve drift.
    """

    def fit(self, reference: CalibrationBatch) -> "CalibrationMonitor":
        _ = reference
        self._is_fitted = True
        return self

    def evaluate(self, current: CalibrationBatch) -> MonitorResult:
        self._check_is_fitted()
        _ = current
        raise NotImplementedError(
            "CalibrationMonitor is a V1 stub. Expected payload keys are "
            "{'y_true', 'y_prob'} for both fit(reference) and evaluate(current)."
        )
