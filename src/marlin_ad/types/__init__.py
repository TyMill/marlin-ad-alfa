"""Core typed primitives for MARLIN-AD."""

from marlin_ad.types.errors import (
    ConfigError,
    ContractError,
    DataError,
    MarlinError,
    ModelError,
    NotFittedError,
)
from marlin_ad.types.protocols import DetectionResult, Detector, Monitor, MonitorResult
from marlin_ad.types.validation import ensure_1d_array, ensure_2d_array, validate_quantile

__all__ = [
    "ConfigError",
    "ContractError",
    "DataError",
    "MarlinError",
    "ModelError",
    "NotFittedError",
    "DetectionResult",
    "MonitorResult",
    "Detector",
    "Monitor",
    "ensure_1d_array",
    "ensure_2d_array",
    "validate_quantile",
]
