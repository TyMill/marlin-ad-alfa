from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.protocols import MonitorResult


class ConceptDriftBatch(TypedDict):
    """Expected payload for future supervised concept-drift checks."""

    y_true: list[float]
    y_pred: list[float]


@dataclass
class ConceptDriftMonitor(BaseMonitor):
    """Structured V1 stub for concept drift monitoring.

    Planned V2 behavior: compare error and calibration characteristics between
    reference and current labeled batches.
    """

    def fit(self, reference: ConceptDriftBatch) -> "ConceptDriftMonitor":
        _ = reference
        self._is_fitted = True
        return self

    def evaluate(self, current: ConceptDriftBatch) -> MonitorResult:
        self._check_is_fitted()
        _ = current
        raise NotImplementedError(
            "ConceptDriftMonitor is a V1 stub. Expected payload keys are "
            "{'y_true', 'y_pred'} for both fit(reference) and evaluate(current)."
        )
