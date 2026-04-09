"""Typed protocol contracts and result envelopes used across MARLIN-AD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt

from marlin_ad.types.errors import ContractError


MetadataValue = object
ExplanationValue = object


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Stable detector output contract."""

    scores: npt.NDArray[np.floating[Any]]
    labels: npt.NDArray[np.int_] | None
    metadata: Mapping[str, MetadataValue]
    explanations: Mapping[str, ExplanationValue] | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=float).reshape(-1)
        if scores.size == 0:
            raise ContractError("DetectionResult.scores must be non-empty.")
        if not np.isfinite(scores).all():
            raise ContractError("DetectionResult.scores must be finite.")

        labels = self.labels
        if labels is not None:
            label_arr = np.asarray(labels, dtype=int).reshape(-1)
            if label_arr.shape[0] != scores.shape[0]:
                raise ContractError(
                    "DetectionResult.labels must match DetectionResult.scores length."
                )
            object.__setattr__(self, "labels", label_arr)

        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.explanations is not None:
            object.__setattr__(self, "explanations", dict(self.explanations))


@dataclass(frozen=True, slots=True)
class MonitorResult:
    """Stable monitor output contract."""

    metrics: Mapping[str, float]
    alerts: Sequence[str]
    metadata: Mapping[str, MetadataValue]

    def __post_init__(self) -> None:
        coerced_metrics: dict[str, float] = {k: float(v) for k, v in self.metrics.items()}
        object.__setattr__(self, "metrics", coerced_metrics)
        object.__setattr__(self, "alerts", tuple(self.alerts))
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class Detector(Protocol):
    @property
    def is_fitted(self) -> bool: ...

    def fit(self, X: Any, y: Any | None = None) -> "Detector": ...

    def score(self, X: Any) -> DetectionResult: ...


@runtime_checkable
class Monitor(Protocol):
    @property
    def is_fitted(self) -> bool: ...

    def fit(self, reference: Any) -> "Monitor": ...

    def evaluate(self, current: Any) -> MonitorResult: ...
