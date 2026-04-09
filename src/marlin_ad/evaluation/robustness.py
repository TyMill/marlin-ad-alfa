from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt

from marlin_ad.types.protocols import Detector


def add_gaussian_noise(
    X: npt.NDArray[np.float64],
    *,
    sigma: float = 0.1,
    random_state: int | None = None,
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(random_state)
    return np.asarray(X + rng.normal(scale=sigma, size=X.shape), dtype=np.float64)


def robustness_curve(
    detector: Detector,
    X_reference: npt.NDArray[np.float64],
    X_current: npt.NDArray[np.float64],
    *,
    sigmas: Iterable[float] = (0.0, 0.1, 0.2),
) -> dict[float, float]:
    """Compute anomaly rate under increasing noise levels."""
    detector.fit(X_reference)
    results: dict[float, float] = {}
    for sigma in sigmas:
        noisy = add_gaussian_noise(X_current, sigma=sigma, random_state=42)
        scores = detector.score(noisy)
        rate = float(np.mean(scores.labels)) if scores.labels is not None else 0.0
        results[float(sigma)] = rate
    return results
