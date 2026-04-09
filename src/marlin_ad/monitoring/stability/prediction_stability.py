from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.errors import DataError
from marlin_ad.types.protocols import MonitorResult
from marlin_ad.types.validation import ensure_1d_array


_EPSILON = 1e-8


def _safe_histogram(
    values: npt.NDArray[np.floating[Any]],
    bin_edges: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    counts, _ = np.histogram(values, bins=bin_edges)
    proportions = counts.astype(float) / max(float(values.size), 1.0)
    return np.clip(proportions, _EPSILON, None)


def _population_stability_index(
    reference: npt.NDArray[np.floating[Any]], current: npt.NDArray[np.floating[Any]]
) -> float:
    ratio = current / reference
    return float(np.sum((current - reference) * np.log(ratio)))


@dataclass
class PredictionStabilityMonitor(BaseMonitor):
    """Lightweight monitor for prediction distribution stability.

    The monitor tracks three deterministic and interpretable signals:
    1) absolute mean shift,
    2) relative standard-deviation shift,
    3) population stability index (PSI).
    """

    mean_tolerance: float = 0.2
    std_ratio_tolerance: float = 0.25
    psi_tolerance: float = 0.2
    n_bins: int = 10
    _reference_mean: float | None = field(init=False, default=None, repr=False)
    _reference_std: float | None = field(init=False, default=None, repr=False)
    _reference_histogram: npt.NDArray[np.floating[Any]] | None = field(
        init=False, default=None, repr=False
    )
    _bin_edges: npt.NDArray[np.floating[Any]] | None = field(init=False, default=None, repr=False)

    def fit(self, reference: Any) -> "PredictionStabilityMonitor":
        if self.mean_tolerance < 0.0:
            raise DataError("mean_tolerance must be >= 0.")
        if self.std_ratio_tolerance < 0.0:
            raise DataError("std_ratio_tolerance must be >= 0.")
        if self.psi_tolerance < 0.0:
            raise DataError("psi_tolerance must be >= 0.")
        if self.n_bins < 2:
            raise DataError("n_bins must be >= 2.")

        reference_values = ensure_1d_array(reference, name="reference_predictions")
        self._reference_mean = float(np.mean(reference_values))
        self._reference_std = float(np.std(reference_values))

        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        candidate_edges = np.quantile(reference_values, quantiles)
        unique_edges = np.unique(candidate_edges)
        if unique_edges.size < 3:
            minimum = float(np.min(reference_values))
            maximum = float(np.max(reference_values))
            if np.isclose(minimum, maximum):
                minimum -= 0.5
                maximum += 0.5
            unique_edges = np.linspace(minimum, maximum, self.n_bins + 1)

        bin_edges = unique_edges.astype(float)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        self._bin_edges = bin_edges
        self._reference_histogram = _safe_histogram(reference_values, bin_edges)
        self._is_fitted = True
        return self

    def evaluate(self, current: Any) -> MonitorResult:
        self._check_is_fitted()
        if (
            self._reference_mean is None
            or self._reference_std is None
            or self._reference_histogram is None
            or self._bin_edges is None
        ):
            raise DataError("Reference state missing; call fit() first.")

        current_values = ensure_1d_array(current, name="current_predictions")
        current_mean = float(np.mean(current_values))
        current_std = float(np.std(current_values))

        mean_shift = abs(current_mean - self._reference_mean)
        standardized_mean_shift = mean_shift / max(self._reference_std, _EPSILON)
        std_ratio = current_std / max(self._reference_std, _EPSILON)

        current_histogram = _safe_histogram(current_values, self._bin_edges)
        psi = _population_stability_index(self._reference_histogram, current_histogram)

        alerts: list[str] = []
        if mean_shift > self.mean_tolerance:
            alerts.append("prediction_stability:mean_shift")
        if abs(std_ratio - 1.0) > self.std_ratio_tolerance:
            alerts.append("prediction_stability:std_ratio_shift")
        if psi > self.psi_tolerance:
            alerts.append("prediction_stability:psi_shift")

        metrics = {
            "prediction.current_mean": current_mean,
            "prediction.current_std": current_std,
            "prediction.mean_shift": mean_shift,
            "prediction.standardized_mean_shift": standardized_mean_shift,
            "prediction.std_ratio": std_ratio,
            "prediction.psi": psi,
            "summary.alert_count": float(len(alerts)),
        }
        metadata: Mapping[str, Any] = {
            "method": "distribution_stability",
            "mean_tolerance": self.mean_tolerance,
            "std_ratio_tolerance": self.std_ratio_tolerance,
            "psi_tolerance": self.psi_tolerance,
            "n_bins": self.n_bins,
            "reference_mean": self._reference_mean,
            "reference_std": self._reference_std,
        }
        return MonitorResult(metrics=metrics, alerts=tuple(alerts), metadata=metadata)
