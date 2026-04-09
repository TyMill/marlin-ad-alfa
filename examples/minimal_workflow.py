"""Minimal MARLIN-AD workflow example.

This script demonstrates the two-core-lens workflow:
1) anomaly detection in maritime-like operational data,
2) anomaly monitoring of model-facing behaviour via drift checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marlin_ad.detection import IsolationForestDetector
from marlin_ad.monitoring import KSTestDriftMonitor


def main() -> None:
    rng = np.random.default_rng(42)

    reference = pd.DataFrame(
        {
            "speed_knots": rng.normal(13.0, 1.0, size=300),
            "engine_temp_c": rng.normal(79.0, 1.8, size=300),
            "course_change": rng.normal(0.0, 3.0, size=300),
        }
    )

    current = pd.DataFrame(
        {
            "speed_knots": rng.normal(14.2, 1.2, size=300),
            "engine_temp_c": rng.normal(81.5, 2.3, size=300),
            "course_change": rng.normal(0.0, 3.5, size=300),
        }
    )

    # Inject sparse operational anomalies.
    idx = rng.choice(len(current), size=20, replace=False)
    current.loc[idx, "engine_temp_c"] += rng.normal(15.0, 3.0, size=len(idx))

    detector = IsolationForestDetector(random_state=42).fit(reference.to_numpy())
    detection = detector.score(current.to_numpy())

    monitor = KSTestDriftMonitor(pvalue_threshold=0.01).fit(reference)
    drift = monitor.evaluate(current)

    anomaly_rate = float(detection.labels.mean())

    print("anomaly_rate", round(anomaly_rate, 3))
    print("drift_alert_count", len(drift.alerts))
    print("drift_alerts", drift.alerts)


if __name__ == "__main__":
    main()
