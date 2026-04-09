"""Run lightweight benchmark suites for detectors and monitors."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.detection.classical.lof import LOFDetector
from marlin_ad.detection.classical.robust_stats import RobustZScoreDetector
from marlin_ad.evaluation.benchmarks import BenchmarkCase, run_benchmark
from marlin_ad.evaluation.reports import EvaluationReport, evaluate_detection
from marlin_ad.monitoring.drift.data_drift import KSTestDriftMonitor


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    factory: Callable[[], Any]
    needs_threshold: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_detection_data(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_features: int,
    anomaly_fraction: float,
    reference_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ref = int(n_samples * reference_fraction)
    n_cur = max(1, n_samples - n_ref)
    X_ref = rng.normal(0, 1, size=(n_ref, n_features))
    X_cur = rng.normal(0, 1, size=(n_cur, n_features))
    y_true = np.zeros(n_cur, dtype=int)
    n_anom = int(n_cur * anomaly_fraction)
    if n_anom > 0:
        idx = rng.choice(n_cur, size=n_anom, replace=False)
        X_cur[idx] += rng.normal(6, 1.5, size=(n_anom, n_features))
        y_true[idx] = 1
    return X_ref, X_cur, y_true


def _run_detector_benchmarks(
    detectors: list[DetectorSpec],
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    y_true: np.ndarray,
    threshold_quantile: float,
) -> dict[str, EvaluationReport]:
    cases: list[BenchmarkCase] = []
    results: dict[str, EvaluationReport] = {}

    for spec in detectors:
        detector = spec.factory()
        if spec.needs_threshold:
            detector = detector.fit(X_ref)
            det_result = detector.score(X_cur)
            threshold = float(np.quantile(det_result.scores, threshold_quantile))
            report = evaluate_detection(det_result, y_true, threshold=threshold)
            meta = dict(report.metadata)
            meta["threshold"] = threshold
            results[spec.name] = EvaluationReport(metrics=report.metrics, metadata=meta)
        else:
            cases.append(
                BenchmarkCase(
                    name=spec.name,
                    detector=detector,
                    X_reference=X_ref,
                    X_current=X_cur,
                    y_true=y_true,
                )
            )

    if cases:
        results.update(run_benchmark(cases))

    return results


def _run_monitor_benchmarks(rng: np.random.Generator, n_samples: int) -> dict[str, Any]:
    ref = pd.DataFrame(
        {
            "speed": rng.normal(12, 2, size=n_samples),
            "heading": rng.normal(180, 15, size=n_samples),
        }
    )
    cur = pd.DataFrame(
        {
            "speed": rng.normal(14, 2.5, size=n_samples),
            "heading": rng.normal(210, 20, size=n_samples),
        }
    )
    monitor = KSTestDriftMonitor(pvalue_threshold=0.05, min_samples=25).fit(ref)
    result = monitor.evaluate(cur)
    return {
        "ks_drift": {
            "metrics": dict(result.metrics),
            "alerts": list(result.alerts),
            "metadata": dict(result.metadata),
        }
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark suites for MARLIN-AD.")
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON file for benchmark results.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--samples",
        type=int,
        default=600,
        help="Number of total samples for synthetic data.",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=6,
        help="Number of features for synthetic data.",
    )
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=0.08,
        help="Fraction of anomalies in the current window.",
    )
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.6,
        help="Fraction of samples used for reference data.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Quantile used when detectors need a score threshold.",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=["isolation_forest", "robust_zscore", "lof"],
        help="Detectors to run (isolation_forest, robust_zscore, lof).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_ref, X_cur, y_true = _make_detection_data(
        rng,
        n_samples=args.samples,
        n_features=args.features,
        anomaly_fraction=args.anomaly_fraction,
        reference_fraction=args.reference_fraction,
    )

    registry: dict[str, DetectorSpec] = {
        "isolation_forest": DetectorSpec(
            name="isolation_forest",
            factory=IsolationForestDetector,
        ),
        "robust_zscore": DetectorSpec(
            name="robust_zscore",
            factory=RobustZScoreDetector,
        ),
        "lof": DetectorSpec(
            name="lof",
            factory=LOFDetector,
            needs_threshold=True,
        ),
    }

    selected = [registry[name] for name in args.detectors if name in registry]
    detector_results = _run_detector_benchmarks(
        selected, X_ref, X_cur, y_true, args.threshold_quantile
    )
    monitor_results = _run_monitor_benchmarks(rng, n_samples=min(300, args.samples))

    payload = {
        "config": {
            "samples": args.samples,
            "features": args.features,
            "anomaly_fraction": args.anomaly_fraction,
            "reference_fraction": args.reference_fraction,
            "threshold_quantile": args.threshold_quantile,
            "detectors": [spec.name for spec in selected],
        },
        "detectors": {name: asdict(report) for name, report in detector_results.items()},
        "monitors": monitor_results,
    }

    output_path = _repo_root() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
