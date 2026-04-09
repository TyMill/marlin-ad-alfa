"""Utilities for converting anomaly scores into binary anomaly labels.

All helpers assume the MARLIN-AD orientation convention:
**higher score means more anomalous**.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from marlin_ad.types.errors import DataError
from marlin_ad.types.validation import ensure_1d_array, validate_quantile


def score_threshold(scores: npt.ArrayLike, quantile: float = 0.95) -> float:
    """Compute a score threshold at the requested quantile.

    Parameters
    ----------
    scores:
        One-dimensional anomaly scores where higher values indicate stronger anomalies.
    quantile:
        Quantile in ``(0, 1)`` used to define the threshold.

    Returns
    -------
    float
        The scalar threshold value.
    """

    validate_quantile(quantile)
    score_arr = ensure_1d_array(scores, name="scores")
    return float(np.quantile(score_arr, quantile))


def labels_from_threshold(scores: npt.ArrayLike, threshold: float) -> npt.NDArray[np.int_]:
    """Generate binary anomaly labels from a fixed scalar threshold.

    Scores greater than or equal to ``threshold`` are labeled anomalous (1).
    """

    score_arr = ensure_1d_array(scores, name="scores")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise DataError("threshold must be finite.")
    return (score_arr >= threshold_value).astype(int)


def threshold_scores(scores: npt.ArrayLike, quantile: float = 0.95) -> npt.NDArray[np.int_]:
    """Generate binary anomaly labels using a quantile threshold.

    This is equivalent to computing :func:`score_threshold` then applying
    :func:`labels_from_threshold`.
    """

    threshold = score_threshold(scores, quantile)
    return labels_from_threshold(scores, threshold)
