"""Canonical default configuration for MARLIN-AD."""

from __future__ import annotations

from typing import Final

DEFAULT_DETECTION_THRESHOLD: Final[float] = 0.95
DEFAULT_MONITORING_DRIFT_THRESHOLD: Final[float] = 0.2
DEFAULT_MONITORING_MIN_SAMPLES: Final[int] = 25
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOGGER_NAME: Final[str] = "marlin_ad"

DEFAULTS: Final[dict[str, object]] = {
    "detection": {"threshold": DEFAULT_DETECTION_THRESHOLD},
    "monitoring": {
        "drift_threshold": DEFAULT_MONITORING_DRIFT_THRESHOLD,
        "min_samples": DEFAULT_MONITORING_MIN_SAMPLES,
    },
    "logging": {"level": DEFAULT_LOG_LEVEL, "logger_name": DEFAULT_LOGGER_NAME},
}
