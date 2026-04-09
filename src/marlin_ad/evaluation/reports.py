from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from marlin_ad.detection.scoring import labels_from_threshold
from marlin_ad.evaluation.metrics import precision_recall_f1
from marlin_ad.types.protocols import DetectionResult


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any]


def evaluate_detection(
    result: DetectionResult,
    y_true: npt.NDArray[np.int_],
    *,
    threshold: float | None = None,
) -> EvaluationReport:
    """Evaluate a detector result against ground truth labels."""
    if threshold is not None:
        y_pred = labels_from_threshold(result.scores, threshold)
    elif result.labels is not None:
        y_pred = result.labels
    else:
        raise ValueError("DetectionResult missing labels and no threshold was provided.")
    metrics = precision_recall_f1(y_true.astype(int), y_pred.astype(int))
    metadata = {"threshold": threshold, "metadata": dict(result.metadata)}
    return EvaluationReport(metrics=metrics, metadata=metadata)
