from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from marlin_ad.detection.base import BaseDetector
from marlin_ad.types.protocols import DetectionResult
from marlin_ad.types.validation import ensure_2d_array


@dataclass
class PCAAutoencoderDetector(BaseDetector):
    """Lightweight autoencoder-style detector using PCA reconstruction error."""

    n_components: int = 2

    def __post_init__(self) -> None:
        self._model = PCA(n_components=self.n_components, random_state=42)

    def fit(self, X: Any, y: Any | None = None) -> "PCAAutoencoderDetector":
        arr = ensure_2d_array(X)
        self._model.fit(arr)
        self._is_fitted = True
        return self

    def score(self, X: Any) -> DetectionResult:
        self._check_is_fitted()
        arr = ensure_2d_array(X)
        recon = self._model.inverse_transform(self._model.transform(arr))
        errors = np.mean((arr - recon) ** 2, axis=1)
        labels = None
        meta = {"method": "pca_autoencoder", "n_components": self.n_components}
        return DetectionResult(scores=errors, labels=labels, metadata=meta)
