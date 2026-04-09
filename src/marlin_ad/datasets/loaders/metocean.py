from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from marlin_ad.datasets.validation import METOCEAN_REQUIRED_COLUMNS, require_columns


def load_metocean(
    path: str | Path,
    *,
    validate_schema: bool = True,
    ensure_utc_timestamp: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load metocean observations from a local CSV file."""
    frame = pd.read_csv(path, **read_csv_kwargs)
    if validate_schema:
        require_columns(frame, METOCEAN_REQUIRED_COLUMNS, dataset_name="metocean")
    if ensure_utc_timestamp and "timestamp" in frame.columns:
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame
