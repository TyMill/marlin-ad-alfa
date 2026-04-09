from __future__ import annotations

from typing import Any

import numpy as np

from marlin_ad.types.protocols import DetectionResult, Detector, Monitor, MonitorResult


def score_detector(detector: Detector, X: Any) -> DetectionResult:
    """Score samples with a fitted detector."""

    return detector.score(X)


def evaluate_monitor(monitor: Monitor, current: Any) -> MonitorResult:
    """Evaluate current behavior with a fitted monitor."""

    return monitor.evaluate(current)


def detection_anomaly_rate(result: DetectionResult) -> float:
    """Compute anomaly rate using labels when available, otherwise score sign."""

    if result.labels is not None:
        return float(np.mean(result.labels))
    return float(np.mean(result.scores > 0.0))
