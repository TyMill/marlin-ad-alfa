from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from marlin_ad.types.protocols import DetectionResult


@dataclass(frozen=True)
class ExplainabilityReport:
    """Lightweight explainability report for detector outputs."""

    summary: Mapping[str, Any]
    top_features: list[tuple[str, float]]
    metadata: Mapping[str, Any]


def build_explainability_report(
    result: DetectionResult,
    *,
    feature_names: Sequence[str] | None = None,
    top_k: int = 5,
) -> ExplainabilityReport:
    scores = result.scores
    summary = {
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "num_anomalies": int(np.sum(result.labels)) if result.labels is not None else None,
    }
    top_features: list[tuple[str, float]] = []
    if result.explanations and "feature_scores" in result.explanations:
        feature_scores = np.asarray(result.explanations["feature_scores"])
        feature_importance = np.mean(np.abs(feature_scores), axis=0)
        names = list(feature_names) if feature_names else [f"f{i}" for i in range(len(feature_importance))]
        ranked = sorted(zip(names, feature_importance), key=lambda x: x[1], reverse=True)
        top_features = [(name, float(score)) for name, score in ranked[:top_k]]
    return ExplainabilityReport(summary=summary, top_features=top_features, metadata=result.metadata)
