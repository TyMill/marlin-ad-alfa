from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from marlin_ad.evaluation.reports import EvaluationReport, evaluate_detection
from marlin_ad.types.protocols import Detector


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    detector: Detector
    X_reference: npt.NDArray[np.float64]
    X_current: npt.NDArray[np.float64]
    y_true: npt.NDArray[np.int_]


def run_benchmark(cases: Iterable[BenchmarkCase]) -> Mapping[str, EvaluationReport]:
    results: dict[str, EvaluationReport] = {}
    for case in cases:
        detector = case.detector.fit(case.X_reference)
        det_result = detector.score(case.X_current)
        report = evaluate_detection(det_result, case.y_true)
        results[case.name] = report
    return results
