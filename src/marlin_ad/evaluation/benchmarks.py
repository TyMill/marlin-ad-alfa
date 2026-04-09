from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from marlin_ad.evaluation.reports import EvaluationReport, evaluate_detection
from marlin_ad.types.protocols import Detector


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    detector: Detector
    X_reference: np.ndarray
    X_current: np.ndarray
    y_true: np.ndarray


def run_benchmark(cases: Iterable[BenchmarkCase]) -> Mapping[str, EvaluationReport]:
    results: dict[str, EvaluationReport] = {}
    for case in cases:
        detector = case.detector.fit(case.X_reference)
        det_result = detector.score(case.X_current)
        report = evaluate_detection(det_result, case.y_true)
        results[case.name] = report
    return results
