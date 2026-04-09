from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from marlin_ad.datasets.validation import LOGS_REQUIRED_COLUMNS, require_columns


def load_logs(
    path: str | Path,
    *,
    validate_schema: bool = True,
    ensure_utc_timestamp: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load operational vessel logs from a local CSV file."""
    frame = pd.read_csv(path, **read_csv_kwargs)
    if validate_schema:
        require_columns(frame, LOGS_REQUIRED_COLUMNS, dataset_name="logs")
    if ensure_utc_timestamp and "timestamp" in frame.columns:
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame
