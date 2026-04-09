# Maritime Data

MARLIN-AD currently targets typed tabular workflows (`pandas.DataFrame` and CSV). This design emphasizes explicit schema contracts before model fitting/scoring.

## Data modalities in scope

Typical maritime signals include:

- AIS trajectory/state fields,
- machinery and engine telemetry,
- metocean context,
- operational and event logs.

Different modalities are expected to be aligned into coherent feature tables before detector and monitor execution.

## Loader and validation philosophy

The `marlin_ad.datasets` package separates two concerns:

1. **Loading/parsing** (source-specific assumptions), and
2. **Validation** (explicit column/type constraints).

This makes pipeline assumptions auditable and reduces silent schema drift in longitudinal studies.

## AIS loader: minimal contract

```python
from marlin_ad.datasets.loaders import load_ais

df = load_ais("data/ais.csv")
```

Expected minimum columns:

- `timestamp`
- `mmsi`
- `lat`
- `lon`

By default the loader validates required columns and normalizes timestamps to UTC datetimes when present.

## Minimal AIS sample

```csv
timestamp,mmsi,lat,lon,sog,cog,heading,nav_status
2025-01-01T00:00:00Z,111000111,42.3601,-70.9912,12.3,89.1,90,under_way
2025-01-01T00:10:00Z,111000111,42.3748,-70.9120,12.7,88.7,89,under_way
```

## Explicit validation in pipelines

```python
from marlin_ad.datasets.validation import require_columns

require_columns(df, ["timestamp", "mmsi", "lat", "lon"], dataset_name="ais")
```

## Recommended dataset hygiene

Before detection/monitoring:

- enforce timestamp monotonicity within vessel tracks,
- resolve missing and non-finite numeric values explicitly,
- document interpolation, resampling, and windowing strategy,
- freeze reference/current split logic for all reported experiments,
- preserve dataset version/hash metadata in outputs.
