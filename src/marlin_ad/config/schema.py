"""Pydantic configuration schema for MARLIN-AD core contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marlin_ad.config.defaults import (
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_LOGGER_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MONITORING_DRIFT_THRESHOLD,
    DEFAULT_MONITORING_MIN_SAMPLES,
)


class DetectionConfig(BaseModel):
    """Detector-level baseline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(DEFAULT_DETECTION_THRESHOLD, gt=0.0, lt=1.0)


class MonitoringConfig(BaseModel):
    """Monitoring baseline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    drift_threshold: float = Field(DEFAULT_MONITORING_DRIFT_THRESHOLD, ge=0.0)
    min_samples: int = Field(DEFAULT_MONITORING_MIN_SAMPLES, ge=1)


class LoggingConfig(BaseModel):
    """Application logging baseline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: str = Field(DEFAULT_LOG_LEVEL)
    logger_name: str = Field(DEFAULT_LOGGER_NAME, min_length=1)

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in allowed:
            msg = f"Unsupported log level: {value!r}. Allowed values: {sorted(allowed)}"
            raise ValueError(msg)
        return normalized


class MarlinConfig(BaseModel):
    """Top-level immutable config object for MARLIN-AD."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection: DetectionConfig = Field(
        default_factory=lambda: DetectionConfig(threshold=DEFAULT_DETECTION_THRESHOLD)
    )
    monitoring: MonitoringConfig = Field(
        default_factory=lambda: MonitoringConfig(
            drift_threshold=DEFAULT_MONITORING_DRIFT_THRESHOLD,
            min_samples=DEFAULT_MONITORING_MIN_SAMPLES,
        )
    )
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(level=DEFAULT_LOG_LEVEL, logger_name=DEFAULT_LOGGER_NAME)
    )
