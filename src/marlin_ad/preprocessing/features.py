from __future__ import annotations

import pandas as pd


def add_rate_of_change(
    df: pd.DataFrame,
    col: str,
    *,
    by: str | None = "vessel_id",
    time_col: str = "timestamp",
    per_seconds: float | None = None,
) -> pd.DataFrame:
    """Add a first-difference (or time-normalized) rate-of-change feature."""
    enriched = df.copy()

    value_delta = (
        enriched.groupby(by, sort=False)[col].diff()
        if by is not None and by in enriched.columns
        else enriched[col].diff()
    )

    if per_seconds is None:
        enriched[f"{col}_roc"] = value_delta
        return enriched

    elapsed_seconds = (
        enriched.groupby(by, sort=False)[time_col]
        .diff()
        .dt.total_seconds()
        if by is not None and by in enriched.columns
        else enriched[time_col].diff().dt.total_seconds()
    )
    enriched[f"{col}_roc_per_{int(per_seconds)}s"] = value_delta / (elapsed_seconds / per_seconds)
    return enriched
