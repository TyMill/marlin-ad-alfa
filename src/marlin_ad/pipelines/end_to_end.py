from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from marlin_ad.alerting.formatter import Alert, format_alert
from marlin_ad.alerting.rules import AlertRule, evaluate_rules
from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.monitoring.drift.data_drift import KSTestDriftMonitor
from marlin_ad.monitoring.health.self_tests import SelfTestMonitor
from marlin_ad.pipelines.fit import fit_detector, fit_monitor
from marlin_ad.pipelines.score import detection_anomaly_rate, evaluate_monitor, score_detector
from marlin_ad.preprocessing.cleaning import basic_clean
from marlin_ad.preprocessing.time import ensure_datetime
from marlin_ad.preprocessing.windows import rolling_mean
from marlin_ad.types.errors import DataError
from marlin_ad.types.protocols import DetectionResult, Detector, Monitor, MonitorResult


@dataclass(frozen=True)
class PipelineReport:
    """Structured report for a minimal MARLIN-AD orchestration run."""

    anomaly_rate: float
    detection: DetectionResult
    data_monitoring: MonitorResult
    system_monitoring: MonitorResult
    alerts: tuple[str, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "anomaly_rate": self.anomaly_rate,
            "detection": {
                "scores": self.detection.scores,
                "labels": self.detection.labels,
                "metadata": dict(self.detection.metadata),
            },
            "data_monitoring": {
                "metrics": dict(self.data_monitoring.metrics),
                "alerts": list(self.data_monitoring.alerts),
                "metadata": dict(self.data_monitoring.metadata),
            },
            "system_monitoring": {
                "metrics": dict(self.system_monitoring.metrics),
                "alerts": list(self.system_monitoring.alerts),
                "metadata": dict(self.system_monitoring.metadata),
            },
            # Backward-compatible alias used by current CLI/tests.
            "drift": {
                "metrics": dict(self.data_monitoring.metrics),
                "alerts": list(self.data_monitoring.alerts),
                "metadata": dict(self.data_monitoring.metadata),
            },
            "alerts": list(self.alerts),
            "metadata": dict(self.metadata),
        }


def _select_numeric_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    excluded = set(exclude or [])
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def run_end_to_end_csv(
    path: str,
    *,
    detector: Detector | None = None,
    data_monitor: Monitor | None = None,
    system_monitor: Monitor | None = None,
    alert_rules: Iterable[AlertRule] | None = None,
    timestamp_col: str = "timestamp",
    reference_fraction: float = 0.5,
) -> Mapping[str, Any]:
    """Run a minimal two-stage MARLIN-AD pipeline on a CSV file.

    Stage 1: detect anomalies in maritime input data.
    Stage 2: assess whether the AI system itself behaves abnormally.
    """

    df = basic_clean(pd.read_csv(path))
    if timestamp_col in df.columns:
        df = ensure_datetime(df, col=timestamp_col)

    numeric_cols = _select_numeric_columns(df, exclude=[timestamp_col])
    if not numeric_cols:
        raise DataError("No numeric columns found for detection.")

    if numeric_cols:
        # Lightweight smoothing for noisy streams without changing column semantics.
        df = rolling_mean(df, col=numeric_cols[0], window=5)

    split_idx = int(len(df) * reference_fraction)
    if split_idx <= 0 or split_idx >= len(df):
        raise DataError("reference_fraction must split the dataset into reference/current parts.")

    reference_df = df.iloc[:split_idx]
    current_df = df.iloc[split_idx:]

    x_reference = reference_df[numeric_cols].to_numpy()
    x_current = current_df[numeric_cols].to_numpy()

    resolved_detector = detector or IsolationForestDetector()
    resolved_data_monitor = data_monitor or KSTestDriftMonitor()
    resolved_system_monitor = system_monitor or SelfTestMonitor()

    fit_detector(resolved_detector, x_reference)
    detection_result = score_detector(resolved_detector, x_current)

    fit_monitor(resolved_data_monitor, reference_df[numeric_cols])
    data_monitoring_result = evaluate_monitor(resolved_data_monitor, current_df[numeric_cols])

    reference_scores = score_detector(resolved_detector, x_reference).scores
    fit_monitor(resolved_system_monitor, reference_scores)
    system_monitoring_result = evaluate_monitor(resolved_system_monitor, detection_result.scores)

    alerts: list[str] = []
    alerts.extend(data_monitoring_result.alerts)
    alerts.extend(system_monitoring_result.alerts)

    for rule_alert in evaluate_rules(list(alert_rules or []), data_monitoring_result.metrics):
        alerts.append(format_alert(rule_alert))

    monitor_count_alert = Alert(
        name="pipeline:monitor_alert_count",
        severity="info",
        payload={
            "data_alerts": len(data_monitoring_result.alerts),
            "system_alerts": len(system_monitoring_result.alerts),
        },
    )
    alerts.append(format_alert(monitor_count_alert))

    report = PipelineReport(
        anomaly_rate=detection_anomaly_rate(detection_result),
        detection=detection_result,
        data_monitoring=data_monitoring_result,
        system_monitoring=system_monitoring_result,
        alerts=tuple(alerts),
        metadata={
            "rows_reference": len(reference_df),
            "rows_current": len(current_df),
            "n_features": len(numeric_cols),
            "execution_order": (
                "detect_anomalies",
                "assess_system_health",
                "evaluate_alert_rules",
            ),
        },
    )
    return report.to_dict()
