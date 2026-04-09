from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class EnsembleStats:
    mean: npt.NDArray[np.floating]
    std: npt.NDArray[np.floating]


def ensemble_mean_std(predictions: npt.NDArray[np.floating]) -> EnsembleStats:
    """Compute ensemble mean/std for predictions shaped (n_models, n_samples)."""
    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D array (n_models, n_samples).")
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    return EnsembleStats(mean=mean, std=std)
