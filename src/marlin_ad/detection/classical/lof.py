from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from marlin_ad.types.protocols import DetectionResult

@dataclass
class LOFDetector:
    n_neighbors: int = 35
    novelty: bool = True

    def __post_init__(self) -> None:
        self._model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=self.novelty)

    def fit(self, X: Any, y: Any | None = None) -> "LOFDetector":
        self._model.fit(X)
        return self

    def score(self, X: Any) -> DetectionResult:
        s = -self._model.decision_function(X)
        meta: Mapping[str, Any] = {"method": "lof", "n_neighbors": self.n_neighbors}
        return DetectionResult(scores=np.asarray(s), labels=None, metadata=meta)
