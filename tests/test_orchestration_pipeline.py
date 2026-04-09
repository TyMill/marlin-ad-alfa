from __future__ import annotations

import numpy as np
import pandas as pd

from marlin_ad.pipelines.end_to_end import run_end_to_end_csv


def test_run_end_to_end_csv_returns_two_stage_report(tmp_path) -> None:
    rng = np.random.default_rng(7)
    rows = 120
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "speed": np.r_[rng.normal(10, 0.5, rows // 2), rng.normal(12, 0.5, rows // 2)],
            "heading": rng.normal(180, 5, rows),
        }
    )
    path = tmp_path / "voyage.csv"
    df.to_csv(path, index=False)

    report = run_end_to_end_csv(str(path), reference_fraction=0.5)

    assert "detection" in report
    assert "data_monitoring" in report
    assert "system_monitoring" in report
    assert "alerts" in report
    assert report["metadata"]["execution_order"] == (
        "detect_anomalies",
        "assess_system_health",
        "evaluate_alert_rules",
    )
    assert isinstance(report["anomaly_rate"], float)
    assert 0.0 <= report["anomaly_rate"] <= 1.0
    assert "drift" in report
