"""Configuration schema and defaults for MARLIN-AD."""

from marlin_ad.config.defaults import (
    DEFAULTS,
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_LOGGER_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MONITORING_DRIFT_THRESHOLD,
    DEFAULT_MONITORING_MIN_SAMPLES,
)
from marlin_ad.config.schema import DetectionConfig, LoggingConfig, MarlinConfig, MonitoringConfig

__all__ = [
    "DEFAULTS",
    "DEFAULT_DETECTION_THRESHOLD",
    "DEFAULT_MONITORING_DRIFT_THRESHOLD",
    "DEFAULT_MONITORING_MIN_SAMPLES",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOGGER_NAME",
    "DetectionConfig",
    "MonitoringConfig",
    "LoggingConfig",
    "MarlinConfig",
]
