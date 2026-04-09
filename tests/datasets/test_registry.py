from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marlin_ad.datasets.registry import DatasetSpec, get, list_datasets, load_dataset, register
from marlin_ad.types.errors import DataError


def test_registry_lists_builtin_datasets() -> None:
    names = list_datasets()
    assert {"ais", "engine_sensors", "metocean", "logs"}.issubset(set(names))


def test_registry_rejects_duplicate_name() -> None:
    def _dummy_loader(path: str) -> pd.DataFrame:
        _ = path
        return pd.DataFrame()

    with pytest.raises(ValueError):
        register(DatasetSpec(name="ais", loader=_dummy_loader))


def test_get_unknown_dataset_raises_data_error() -> None:
    with pytest.raises(DataError):
        get("unknown")


def test_load_dataset_delegates_to_registered_loader(tmp_path: Path) -> None:
    ais_path = tmp_path / "ais.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z"],
            "mmsi": [111000111],
            "lat": [42.0],
            "lon": [-70.0],
        }
    ).to_csv(ais_path, index=False)

    loaded = load_dataset("AIS", ais_path)
    assert list(loaded.columns) == ["timestamp", "mmsi", "lat", "lon"]
