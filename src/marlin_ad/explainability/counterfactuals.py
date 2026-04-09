from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CounterfactualResult:
    original: npt.NDArray[np.floating]
    counterfactual: npt.NDArray[np.floating]
    deltas: npt.NDArray[np.floating]


def generate_counterfactual(
    sample: Sequence[float],
    reference_median: Sequence[float],
    *,
    step: float = 0.1,
    max_steps: int = 50,
) -> CounterfactualResult:
    """Generate a simple counterfactual by nudging features toward the reference median."""
    original = np.asarray(sample, dtype=float)
    median = np.asarray(reference_median, dtype=float)
    if original.shape != median.shape:
        raise ValueError("sample and reference_median must have the same shape.")
    current = original.copy()
    for _ in range(max_steps):
        delta = median - current
        if np.allclose(delta, 0.0, atol=1e-6):
            break
        current = current + np.clip(delta, -step, step)
    return CounterfactualResult(
        original=original,
        counterfactual=current,
        deltas=current - original,
    )
