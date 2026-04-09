from __future__ import annotations

import pandas as pd


def rolling_mean(
    df: pd.DataFrame,
    col: str,
    *,
    window: int = 10,
    by: str | None = "vessel_id",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add a rolling mean feature to a frame for global or per-vessel windows."""
    enriched = df.copy()
    feature_name = f"{col}_rollmean_{window}"
    if by is None or by not in enriched.columns:
        enriched[feature_name] = enriched[col].rolling(window, min_periods=min_periods).mean()
        return enriched
    enriched[feature_name] = enriched.groupby(by, sort=False)[col].transform(
        lambda series: series.rolling(window, min_periods=min_periods).mean()
    )
    return enriched
