from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from marlin_ad.monitoring.base import BaseMonitor
from marlin_ad.types.errors import DataError
from marlin_ad.types.protocols import MonitorResult


_EPS = 1e-12


def _ks_2samp(a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]) -> tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n1 = a_sorted.size
    n2 = b_sorted.size
    if n1 == 0 or n2 == 0:
        raise DataError("KS test requires non-empty samples.")

    data_all = np.concatenate([a_sorted, b_sorted])
    cdf1 = np.searchsorted(a_sorted, data_all, side="right") / n1
    cdf2 = np.searchsorted(b_sorted, data_all, side="right") / n2
    ks_statistic = float(np.max(np.abs(cdf1 - cdf2)))

    effective_n = math.sqrt((n1 * n2) / (n1 + n2))
    p_value = _kolmogorov_pvalue(ks_statistic, effective_n)
    return ks_statistic, p_value


def _kolmogorov_pvalue(ks_statistic: float, effective_n: float, terms: int = 100) -> float:
    """Asymptotic Kolmogorov distribution approximation."""
    if ks_statistic <= 0.0:
        return 1.0
    if effective_n <= 0.0:
        return 0.0

    lam = (effective_n + 0.12 + (0.11 / max(effective_n, _EPS))) * ks_statistic
    accumulator = 0.0
    for j in range(1, terms + 1):
        term = ((-1.0) ** (j - 1)) * math.exp(-2.0 * (lam**2) * (j**2))
        accumulator += term
        if abs(term) < 1e-10:
            break

    return float(max(0.0, min(1.0, 2.0 * accumulator)))


@dataclass
class KSTestDriftMonitor(BaseMonitor):
    """Column-wise KS-test drift monitor for numeric tabular features.

    This monitor treats the AI system as a risk-bearing component by checking whether
    current feature distributions diverge from a trusted reference distribution.
    """

    pvalue_threshold: float = 0.01
    min_samples: int = 25
    _columns: list[str] = field(init=False, default_factory=list)
    _reference: pd.DataFrame | None = field(init=False, default=None, repr=False)

    def fit(self, reference: pd.DataFrame) -> "KSTestDriftMonitor":
        if not 0.0 < self.pvalue_threshold < 1.0:
            raise DataError("pvalue_threshold must be in (0, 1).")
        if self.min_samples < 2:
            raise DataError("min_samples must be >= 2.")
        if reference.empty:
            raise DataError("Reference data is empty.")

        numeric_columns = [
            column for column in reference.columns if pd.api.types.is_numeric_dtype(reference[column])
        ]
        if not numeric_columns:
            raise DataError("No numeric columns found for KS drift monitoring.")

        self._reference = reference.copy()
        self._columns = sorted(numeric_columns)
        self._is_fitted = True
        return self

    def evaluate(self, current: pd.DataFrame) -> MonitorResult:
        self._check_is_fitted()
        if self._reference is None:
            raise DataError("Reference data missing; fit() must be called first.")

        metrics: dict[str, float] = {}
        alerts: list[str] = []
        tested_columns = 0
        skipped_columns = 0

        for column in self._columns:
            if column not in current.columns:
                raise DataError(f"Current data missing required column '{column}'.")

            reference_values = self._reference[column].dropna().to_numpy(dtype=float)
            current_values = current[column].dropna().to_numpy(dtype=float)

            if reference_values.size < self.min_samples or current_values.size < self.min_samples:
                skipped_columns += 1
                metrics[f"{column}.insufficient_samples"] = 1.0
                continue

            ks_statistic, p_value = _ks_2samp(reference_values, current_values)
            drift_flag = float(p_value < self.pvalue_threshold)

            metrics[f"{column}.ks_statistic"] = ks_statistic
            metrics[f"{column}.p_value"] = p_value
            metrics[f"{column}.drift_flag"] = drift_flag

            tested_columns += 1
            if drift_flag > 0.0:
                alerts.append(f"data_drift:column:{column}")

        drifted_columns = len(alerts)
        metrics["summary.columns_tested"] = float(tested_columns)
        metrics["summary.columns_skipped"] = float(skipped_columns)
        metrics["summary.columns_drifted"] = float(drifted_columns)
        metrics["summary.drift_ratio"] = (
            float(drifted_columns / tested_columns) if tested_columns > 0 else 0.0
        )

        metadata: Mapping[str, Any] = {
            "method": "ks_test",
            "columns": tuple(self._columns),
            "pvalue_threshold": self.pvalue_threshold,
            "min_samples": self.min_samples,
            "requires_numeric": True,
        }
        return MonitorResult(metrics=metrics, alerts=tuple(alerts), metadata=metadata)
