from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ChangePointResult:
    indices: npt.NDArray[np.int_]
    scores: npt.NDArray[np.floating]


def cusum_changepoint(
    series: Sequence[float],
    *,
    threshold: float = 5.0,
    drift: float = 0.01,
) -> ChangePointResult:
    """Simple CUSUM change-point detection for univariate series."""
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError("series must be 1D.")
    mean = np.mean(x)
    s_pos = 0.0
    s_neg = 0.0
    scores = np.zeros_like(x)
    change_indices: list[int] = []
    for i, value in enumerate(x):
        s_pos = max(0.0, s_pos + value - mean - drift)
        s_neg = min(0.0, s_neg + value - mean + drift)
        score = max(s_pos, abs(s_neg))
        scores[i] = score
        if score > threshold:
            change_indices.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return ChangePointResult(indices=np.asarray(change_indices, dtype=int), scores=scores)
