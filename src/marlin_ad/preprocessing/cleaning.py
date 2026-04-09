from __future__ import annotations

import pandas as pd


def basic_clean(df: pd.DataFrame, *, drop_na_timestamp: bool = True) -> pd.DataFrame:
    """Apply lightweight deterministic cleaning for maritime tabular data.

    Steps:
    1) trim string column whitespace,
    2) drop full-row duplicates,
    3) optionally remove rows with missing timestamps.
    """
    cleaned = df.copy()
    text_columns = cleaned.select_dtypes(include=("object", "string")).columns
    for col in text_columns:
        cleaned[col] = cleaned[col].astype("string").str.strip()
    cleaned = cleaned.drop_duplicates()
    if drop_na_timestamp and "timestamp" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["timestamp"])
    return cleaned.reset_index(drop=True)
