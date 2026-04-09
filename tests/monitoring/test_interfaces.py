from __future__ import annotations

import pytest

from marlin_ad.monitoring.drift.calibration import CalibrationMonitor
from marlin_ad.monitoring.drift.concept_drift import ConceptDriftMonitor
from marlin_ad.monitoring.stability.feature_importance_drift import FeatureImportanceDriftMonitor


def test_concept_drift_monitor_stub_contract() -> None:
    monitor = ConceptDriftMonitor().fit(reference={"ok": True})
    with pytest.raises(NotImplementedError):
        monitor.evaluate(current={"ok": True})


def test_calibration_monitor_stub_contract() -> None:
    monitor = CalibrationMonitor().fit(reference={"ok": True})
    with pytest.raises(NotImplementedError):
        monitor.evaluate(current={"ok": True})


def test_feature_importance_monitor_contract() -> None:
    monitor = FeatureImportanceDriftMonitor(threshold=0.3).fit([0.2, 0.3, 0.5])
    result = monitor.evaluate([0.22, 0.29, 0.49])
    assert result.metrics["max_importance_shift"] < 0.3
    assert result.alerts == ()
