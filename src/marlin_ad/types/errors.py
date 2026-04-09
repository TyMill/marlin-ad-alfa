"""Shared error hierarchy for MARLIN-AD."""

from __future__ import annotations


class MarlinError(RuntimeError):
    """Base error for MARLIN-AD."""


class ConfigError(MarlinError):
    """Raised when configuration is invalid or inconsistent."""


class DataError(MarlinError):
    """Raised when input or reference data violates contract requirements."""


class ModelError(MarlinError):
    """Raised when a model/monitor lifecycle or state is invalid."""


class NotFittedError(ModelError):
    """Raised when inference/evaluation is attempted before fit()."""


class ContractError(MarlinError):
    """Raised when a core detector/monitor contract is violated."""
