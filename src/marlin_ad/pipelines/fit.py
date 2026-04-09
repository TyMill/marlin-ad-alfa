from __future__ import annotations

from typing import Any

from marlin_ad.types.protocols import Detector, Monitor


def fit_detector(detector: Detector, X: Any, y: Any | None = None) -> Detector:
    """Fit a detector and return the same instance for fluent orchestration."""

    return detector.fit(X, y=y)


def fit_monitor(monitor: Monitor, reference: Any) -> Monitor:
    """Fit a monitor on reference behavior and return the same instance."""

    return monitor.fit(reference)
