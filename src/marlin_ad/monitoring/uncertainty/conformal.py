from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ConformalIntervals:
    lower: npt.NDArray[np.floating]
    upper: npt.NDArray[np.floating]


@dataclass
class ConformalRegressor:
    """Basic split conformal regressor for uncertainty intervals."""

    alpha: float = 0.1
    _quantile: float | None = None

    def fit(self, y_true: Sequence[float], y_pred: Sequence[float]) -> "ConformalRegressor":
        residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
        self._quantile = float(np.quantile(residuals, 1 - self.alpha))
        return self

    def interval(self, y_pred: Sequence[float]) -> ConformalIntervals:
        if self._quantile is None:
            raise ValueError("Call fit() before interval().")
        preds = np.asarray(y_pred, dtype=float)
        lower = preds - self._quantile
        upper = preds + self._quantile
        return ConformalIntervals(lower=lower, upper=upper)
