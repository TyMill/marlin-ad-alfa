from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marlin_ad.datasets.loaders import load_ais, load_engine_sensors, load_logs, load_metocean
from marlin_ad.datasets.validation import AIS_REQUIRED_COLUMNS, missing_columns, require_columns
from marlin_ad.types.errors import DataError


def test_missing_columns_and_require_columns() -> None:
    frame = pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "mmsi": [123]})
    missing = missing_columns(frame, AIS_REQUIRED_COLUMNS)
    assert missing == ("lat", "lon")

    with pytest.raises(DataError):
        require_columns(frame, AIS_REQUIRED_COLUMNS, dataset_name="ais")


def test_load_ais_parses_timestamp_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "ais.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "mmsi": [111000111],
            "lat": [42.0],
            "lon": [-70.0],
            "sog": [12.3],
        }
    ).to_csv(path, index=False)

    frame = load_ais(path)
    assert str(frame["timestamp"].dtype).startswith("datetime64")


def test_other_domain_loaders_require_minimum_columns(tmp_path: Path) -> None:
    engine_path = tmp_path / "engine.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "vessel_id": ["V001"],
            "engine_id": ["ME1"],
            "rpm": [95.2],
        }
    ).to_csv(engine_path, index=False)
    assert not load_engine_sensors(engine_path).empty

    metocean_path = tmp_path / "metocean.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "lat": [41.2],
            "lon": [-69.1],
            "wind_speed_ms": [8.0],
        }
    ).to_csv(metocean_path, index=False)
    assert not load_metocean(metocean_path).empty

    logs_path = tmp_path / "logs.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "vessel_id": ["V001"],
            "event_type": ["warning"],
            "message": ["High jacket-water temperature"],
        }
    ).to_csv(logs_path, index=False)
    assert not load_logs(logs_path).empty
