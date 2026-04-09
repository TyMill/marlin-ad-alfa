from __future__ import annotations

import numpy as np
import pandas as pd

from marlin_ad.monitoring.drift.data_drift import KSTestDriftMonitor
from marlin_ad.monitoring.health.self_tests import SelfTestMonitor
from marlin_ad.monitoring.stability.prediction_stability import PredictionStabilityMonitor


def test_ks_drift_alerts_on_shift() -> None:
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    current = pd.DataFrame({"a": rng.normal(loc=1.0, size=400), "b": rng.normal(size=400)})

    monitor = KSTestDriftMonitor(pvalue_threshold=0.01, min_samples=50).fit(reference)
    result = monitor.evaluate(current)

    assert result.metrics["a.ks_statistic"] > 0.0
    assert result.metrics["summary.columns_tested"] == 2.0
    assert "data_drift:column:a" in result.alerts


def test_ks_drift_stays_quiet_without_shift() -> None:
    rng = np.random.default_rng(10)
    reference = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    current = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})

    monitor = KSTestDriftMonitor(pvalue_threshold=0.001, min_samples=50).fit(reference)
    result = monitor.evaluate(current)

    assert result.metrics["summary.columns_drifted"] == 0.0
    assert result.alerts == ()


def test_prediction_stability_alerts_on_distribution_shift() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(loc=0.0, scale=1.0, size=500)
    current = rng.normal(loc=0.8, scale=1.4, size=500)

    monitor = PredictionStabilityMonitor(
        mean_tolerance=0.2,
        std_ratio_tolerance=0.15,
        psi_tolerance=0.1,
        n_bins=12,
    ).fit(reference)
    result = monitor.evaluate(current)

    assert result.metrics["prediction.mean_shift"] > 0.2
    assert "prediction_stability:mean_shift" in result.alerts
    assert len(result.alerts) >= 1


def test_prediction_stability_no_alert_for_similar_distribution() -> None:
    rng = np.random.default_rng(15)
    reference = rng.normal(loc=0.0, scale=1.0, size=500)
    current = rng.normal(loc=0.05, scale=1.05, size=500)

    monitor = PredictionStabilityMonitor(
        mean_tolerance=0.2,
        std_ratio_tolerance=0.25,
        psi_tolerance=0.2,
    ).fit(reference)
    result = monitor.evaluate(current)

    assert result.metrics["summary.alert_count"] == 0.0
    assert result.alerts == ()


def test_self_test_monitor_detects_non_finite_values() -> None:
    monitor = SelfTestMonitor(min_sample_ratio=0.5, max_nonfinite_fraction=0.0).fit([0.1, 0.2, 0.3])
    result = monitor.evaluate([0.1, np.nan, np.inf])

    assert result.metrics["health.nonfinite_fraction"] > 0.0
    assert "self_test:non_finite_values" in result.alerts
