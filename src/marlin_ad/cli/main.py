from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from marlin_ad.datasets import get, list_datasets
from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.monitoring.drift.data_drift import KSTestDriftMonitor
from marlin_ad.pipelines.end_to_end import run_end_to_end_csv


@dataclass(frozen=True)
class DemoData:
    """Synthetic reference/current data for the CLI demonstration."""

    reference: pd.DataFrame
    current: pd.DataFrame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marlin-ad",
        description="CLI for MARLIN-AD anomaly detection and model monitoring workflows.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run an end-to-end anomaly + drift monitoring demo.")
    demo.add_argument("--rows", type=int, default=500, help="Rows per split for synthetic demo data.")
    demo.add_argument("--seed", type=int, default=42, help="Random seed for reproducible synthetic data.")
    demo.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path. When supplied, run the production-like CSV pipeline.",
    )

    datasets_cmd = sub.add_parser("datasets", help="Dataset registry utilities.")
    datasets_sub = datasets_cmd.add_subparsers(dest="datasets_cmd", required=True)
    datasets_sub.add_parser("list", help="List built-in datasets and descriptions.")

    return parser


def _make_synthetic_demo_data(rows: int, *, seed: int) -> DemoData:
    if rows < 50:
        raise ValueError("--rows must be >= 50 for a meaningful demo.")

    rng = np.random.default_rng(seed)

    reference = pd.DataFrame(
        {
            "speed_knots": rng.normal(13.0, 1.2, size=rows),
            "course_change": rng.normal(0.0, 3.0, size=rows),
            "engine_temp_c": rng.normal(79.0, 2.0, size=rows),
        }
    )

    current = pd.DataFrame(
        {
            "speed_knots": rng.normal(14.6, 1.4, size=rows),
            "course_change": rng.normal(0.0, 3.5, size=rows),
            "engine_temp_c": rng.normal(82.0, 2.8, size=rows),
        }
    )

    # Add sparse but explicit operational anomalies in the current set.
    anomaly_idx = rng.choice(rows, size=max(3, rows // 20), replace=False)
    current.loc[anomaly_idx, "course_change"] += rng.normal(35.0, 8.0, size=len(anomaly_idx))
    current.loc[anomaly_idx, "engine_temp_c"] += rng.normal(18.0, 4.0, size=len(anomaly_idx))

    return DemoData(reference=reference, current=current)


def _run_demo(args: argparse.Namespace) -> None:
    if args.csv:
        report = run_end_to_end_csv(args.csv)
        print("Mode: CSV pipeline")
        print(f"Anomaly rate: {report['anomaly_rate']:.3f}")
        print(f"Drift alerts: {len(report['drift']['alerts'])}")
        if report["drift"]["alerts"]:
            print("Drift alert details:", ", ".join(report["drift"]["alerts"]))
        return

    data = _make_synthetic_demo_data(args.rows, seed=args.seed)

    detector = IsolationForestDetector(random_state=args.seed).fit(data.reference.to_numpy())
    detection_result = detector.score(data.current.to_numpy())

    drift_monitor = KSTestDriftMonitor(pvalue_threshold=0.01).fit(data.reference)
    drift_result = drift_monitor.evaluate(data.current)

    anomaly_rate = float(detection_result.labels.mean()) if detection_result.labels is not None else 0.0

    print("Mode: synthetic demo")
    print(
        "Synthetic setup: reference=normal maritime behavior, current=distribution shift + injected outliers"
    )
    print(f"Reference rows: {len(data.reference)} | Current rows: {len(data.current)}")
    print(f"Anomaly rate: {anomaly_rate:.3f}")
    print(f"Drift alerts: {len(drift_result.alerts)}")
    if drift_result.alerts:
        print("Drift alert details:", ", ".join(drift_result.alerts))


def _run_datasets_list() -> None:
    for name in list_datasets():
        description = get(name).description
        print(f"{name}: {description}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        _run_demo(args)
        return

    if args.cmd == "datasets" and args.datasets_cmd == "list":
        _run_datasets_list()
        return

    parser.error("Unknown command.")


if __name__ == "__main__":
    main()
