"""Logging helpers for MARLIN-AD."""

from __future__ import annotations

import logging

from marlin_ad.config.defaults import DEFAULT_LOGGER_NAME, DEFAULT_LOG_LEVEL
from marlin_ad.config.schema import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Create or reuse a MARLIN-AD logger from typed configuration."""
    active_config = config or LoggingConfig(level=DEFAULT_LOG_LEVEL, logger_name=DEFAULT_LOGGER_NAME)
    logger = logging.getLogger(active_config.logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, active_config.level))
    return logger
