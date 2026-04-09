from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from marlin_ad.datasets.validation import ENGINE_SENSORS_REQUIRED_COLUMNS, require_columns


def load_engine_sensors(
    path: str | Path,
    *,
    validate_schema: bool = True,
    ensure_utc_timestamp: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load engine sensor telemetry from a local CSV file."""
    frame = pd.read_csv(path, **read_csv_kwargs)
    if validate_schema:
        require_columns(frame, ENGINE_SENSORS_REQUIRED_COLUMNS, dataset_name="engine_sensors")
    if ensure_utc_timestamp and "timestamp" in frame.columns:
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame
