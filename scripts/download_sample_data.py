"""Download or generate sample datasets for experiments."""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DownloadTarget:
    name: str
    url: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_targets(values: Iterable[str]) -> list[DownloadTarget]:
    targets: list[DownloadTarget] = []
    for value in values:
        if "=" in value:
            name, url = value.split("=", 1)
        else:
            url = value
            name = Path(urllib.parse.urlparse(url).path).name or "dataset.csv"
        targets.append(DownloadTarget(name=name, url=url))
    return targets


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _generate_ais(rows: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1)
    nav_statuses = ["under_way", "at_anchor", "restricted", "not_under_cmd"]
    records = []
    for idx in range(rows):
        records.append(
            {
                "timestamp": (base + timedelta(minutes=idx)).isoformat(),
                "mmsi": int(rng.integers(200_000_000, 799_999_999)),
                "lat": float(rng.uniform(-70, 70)),
                "lon": float(rng.uniform(-170, 170)),
                "sog": float(rng.uniform(0, 25)),
                "cog": float(rng.uniform(0, 360)),
                "heading": float(rng.uniform(0, 359)),
                "nav_status": rng.choice(nav_statuses),
            }
        )
    _write_csv(
        output / "ais.csv",
        records,
        ["timestamp", "mmsi", "lat", "lon", "sog", "cog", "heading", "nav_status"],
    )


def _generate_engine_sensors(rows: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed + 11)
    base = datetime(2024, 1, 2)
    records = []
    for idx in range(rows):
        records.append(
            {
                "timestamp": (base + timedelta(minutes=idx)).isoformat(),
                "rpm": float(rng.normal(900, 40)),
                "temp_c": float(rng.normal(82, 3)),
                "pressure_bar": float(rng.normal(8.5, 0.4)),
                "fuel_rate": float(rng.normal(220, 20)),
            }
        )
    _write_csv(
        output / "engine_sensors.csv",
        records,
        ["timestamp", "rpm", "temp_c", "pressure_bar", "fuel_rate"],
    )


def _generate_logs(rows: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed + 21)
    base = datetime(2024, 1, 3)
    systems = ["navigation", "propulsion", "communications", "safety"]
    levels = ["INFO", "WARN", "ERROR"]
    records = []
    for idx in range(rows):
        records.append(
            {
                "timestamp": (base + timedelta(minutes=idx)).isoformat(),
                "system": rng.choice(systems),
                "level": rng.choice(levels, p=[0.8, 0.15, 0.05]),
                "message": f"event_{rng.integers(1000, 9999)}",
            }
        )
    _write_csv(
        output / "logs.csv",
        records,
        ["timestamp", "system", "level", "message"],
    )


def _generate_metocean(rows: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed + 31)
    base = datetime(2024, 1, 4)
    records = []
    for idx in range(rows):
        records.append(
            {
                "timestamp": (base + timedelta(hours=idx)).isoformat(),
                "wave_height_m": float(rng.uniform(0.2, 4.0)),
                "wind_speed_ms": float(rng.uniform(0, 20)),
                "wind_dir_deg": float(rng.uniform(0, 360)),
            }
        )
    _write_csv(
        output / "metocean.csv",
        records,
        ["timestamp", "wave_height_m", "wind_speed_ms", "wind_dir_deg"],
    )


def _download_targets(targets: Iterable[DownloadTarget], output: Path, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for target in targets:
        dest = output / target.name
        if dest.exists() and not force:
            print(f"Skipping existing file: {dest}")
            continue
        try:
            print(f"Downloading {target.url} -> {dest}")
            urllib.request.urlretrieve(target.url, dest)  # noqa: S310
        except Exception as exc:  # pragma: no cover - network errors are expected
            raise RuntimeError(f"Failed to download {target.url}: {exc}") from exc


def _generate_defaults(output: Path, rows: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _generate_ais(rows, seed, output)
    _generate_engine_sensors(rows, seed, output)
    _generate_logs(max(50, rows // 5), seed, output)
    _generate_metocean(max(50, rows // 10), seed, output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download or generate sample datasets for MARLIN-AD experiments."
    )
    parser.add_argument(
        "--output-dir",
        default="data/sample",
        help="Directory to store datasets (default: data/sample).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="Number of rows per generated dataset (default: 500).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Optional URL to download (repeatable). Format: name=url or url.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files if they already exist.",
    )
    args = parser.parse_args()

    output = _repo_root() / args.output_dir
    targets = _parse_targets(args.url)

    try:
        if targets:
            _download_targets(targets, output, args.force)
        else:
            _generate_defaults(output, args.rows, args.seed)
            print(f"Generated sample datasets in {output}")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
