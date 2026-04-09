from __future__ import annotations

import numpy as np
import pytest

from marlin_ad.config.schema import LoggingConfig, MarlinConfig
from marlin_ad.detection.base import BaseDetector
from marlin_ad.logging import setup_logging
from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.errors import ContractError, NotFittedError
from marlin_ad.types.protocols import DetectionResult, MonitorResult


class _DummyDetector(BaseDetector):
    def score(self, X: object) -> DetectionResult:
        self._check_is_fitted()
        _ = X
        return DetectionResult(scores=np.array([0.1, 0.2]), labels=np.array([0, 1]), metadata={})


class _DummyMonitor(BaseMonitor):
    def evaluate(self, current: object) -> MonitorResult:
        self._check_is_fitted()
        _ = current
        return MonitorResult(metrics={"drift": 0.1}, alerts=("ok",), metadata={})


def test_detection_result_enforces_alignment() -> None:
    with pytest.raises(ContractError):
        DetectionResult(scores=np.array([0.1, 0.2]), labels=np.array([1]), metadata={})


def test_monitor_result_is_immutable_envelope() -> None:
    out = MonitorResult(metrics={"a": 1}, alerts=("alert",), metadata={"method": "x"})
    assert out.metrics["a"] == 1.0
    assert out.alerts == ("alert",)


def test_base_detector_not_fitted_error() -> None:
    det = _DummyDetector()
    with pytest.raises(NotFittedError):
        det.score(np.array([1.0]))


def test_base_monitor_not_fitted_error() -> None:
    mon = _DummyMonitor()
    with pytest.raises(NotFittedError):
        mon.evaluate(np.array([1.0]))


def test_marlin_config_defaults_and_validation() -> None:
    cfg = MarlinConfig()
    assert 0.0 < cfg.detection.threshold < 1.0
    assert cfg.monitoring.min_samples >= 1
    assert cfg.logging.level == "INFO"

    with pytest.raises(ValueError):
        LoggingConfig(level="verbose")


def test_setup_logging_uses_logging_config() -> None:
    logger = setup_logging(LoggingConfig(level="DEBUG", logger_name="marlin_ad.tests"))
    assert logger.name == "marlin_ad.tests"
    assert logger.level > 0
