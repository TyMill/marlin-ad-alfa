from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from marlin_ad.types.errors import DataError


def ensure_2d_array(X: Any, *, name: str = "X") -> npt.NDArray[np.floating]:
    """Ensure input is a finite 2D float array."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DataError(f"{name} must be a 2D array. Got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise DataError(f"{name} must contain at least one row.")
    if not np.isfinite(arr).all():
        raise DataError(f"{name} contains NaN or infinite values.")
    return arr


def ensure_1d_array(X: Any, *, name: str = "X") -> npt.NDArray[np.floating]:
    """Ensure input is a finite 1D float array."""
    arr = np.asarray(X, dtype=float).reshape(-1)
    if arr.size == 0:
        raise DataError(f"{name} must contain at least one value.")
    if not np.isfinite(arr).all():
        raise DataError(f"{name} contains NaN or infinite values.")
    return arr


def validate_quantile(quantile: float) -> None:
    if not 0.0 < quantile < 1.0:
        raise DataError(f"quantile must be in (0, 1). Got {quantile}.")
