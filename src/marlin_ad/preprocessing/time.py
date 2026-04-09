from __future__ import annotations

import pandas as pd


def ensure_datetime(
    df: pd.DataFrame,
    col: str = "timestamp",
    *,
    utc: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Parse a timestamp-like column into pandas datetime values."""
    normalized = df.copy()
    normalized[col] = pd.to_datetime(normalized[col], errors="coerce", utc=utc)
    if sort:
        sort_columns = [col]
        if "vessel_id" in normalized.columns:
            sort_columns = ["vessel_id", col]
        normalized = normalized.sort_values(sort_columns)
    return normalized.reset_index(drop=True)
