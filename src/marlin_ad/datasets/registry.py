from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from marlin_ad.datasets.loaders.ais import load_ais
from marlin_ad.datasets.loaders.engine_sensors import load_engine_sensors
from marlin_ad.datasets.loaders.logs import load_logs
from marlin_ad.datasets.loaders.metocean import load_metocean
from marlin_ad.types.errors import DataError


DatasetLoader = Callable[..., Any]

@dataclass
class DatasetSpec:
    name: str
    loader: DatasetLoader
    description: str = ""

_REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec, *, replace: bool = False) -> None:
    key = spec.name.strip().lower()
    if not key:
        raise ValueError("Dataset name cannot be empty.")
    if key in _REGISTRY and not replace:
        raise ValueError(f"Dataset '{key}' is already registered.")
    _REGISTRY[key] = DatasetSpec(name=key, loader=spec.loader, description=spec.description)


def get(name: str) -> DatasetSpec:
    key = name.strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise DataError(f"Unknown dataset '{name}'. Available datasets: [{available}]") from exc


def load_dataset(name: str, path: str | Path, **loader_kwargs: Any) -> Any:
    """Load a registered dataset by name and local path."""
    spec = get(name)
    return spec.loader(path, **loader_kwargs)


def list_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())


def register_builtin_datasets() -> None:
    if _REGISTRY:
        return
    register(
        DatasetSpec(
            name="ais",
            loader=load_ais,
            description="Automatic Identification System (AIS) vessel tracking data.",
        )
    )
    register(
        DatasetSpec(
            name="engine_sensors",
            loader=load_engine_sensors,
            description="Ship engine sensor telemetry (CSV/Parquet).",
        )
    )
    register(
        DatasetSpec(
            name="metocean",
            loader=load_metocean,
            description="Metocean observations (weather/ocean conditions).",
        )
    )
    register(
        DatasetSpec(
            name="logs",
            loader=load_logs,
            description="Operational logs and events.",
        )
    )
