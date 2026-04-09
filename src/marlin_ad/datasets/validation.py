from __future__ import annotations
from collections.abc import Iterable

import pandas as pd

from marlin_ad.types.errors import DataError


AIS_REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "mmsi", "lat", "lon")
ENGINE_SENSORS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "vessel_id",
    "engine_id",
)
METOCEAN_REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "lat", "lon")
LOGS_REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "vessel_id", "event_type")


def missing_columns(df: pd.DataFrame, cols: Iterable[str]) -> tuple[str, ...]:
    """Return required columns missing from the given frame."""
    requested = tuple(cols)
    return tuple(col for col in requested if col not in df.columns)


def require_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    dataset_name: str | None = None,
) -> None:
    """Raise :class:`~marlin_ad.types.errors.DataError` when required columns are absent."""
    missing = missing_columns(df, cols)
    if missing:
        prefix = f"{dataset_name}: " if dataset_name else ""
        raise DataError(f"{prefix}missing required columns: {list(missing)}")
