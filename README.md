[![PyPI version](https://img.shields.io/pypi/v/marlin-ad?color=blue)](https://pypi.org/project/marlin-ad/)
[![Downloads](https://static.pepy.tech/badge/marlin-ad)](https://pepy.tech/project/marlin-ad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19496951.svg)](https://doi.org/10.5281/zenodo.19496951)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tymill.github.io/marlin-ad-alfa/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/marlin-ad?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/marlin-ad)
[![CI](https://github.com/TyMill/marlin-ad-alfa/actions/workflows/ci.yml/badge.svg)](https://github.com/TyMill/marlin-ad-alfa/actions/workflows/ci.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/TyMill/marlin-ad-alfa)](https://github.com/TyMill/marlin-ad-alfa/releases)


# MARLIN-AD

**MARLIN-AD** is a typed Python library for two coupled technical tasks:

1. **anomaly detection in maritime operational data**, and
2. **anomaly detection in AI model behaviour**.

The project is built for settings where model outputs influence safety-critical or high-cost decisions. In these settings, the relevant failure surface includes both the physical process (vessel, machinery, environment, operations) and the model layer (drift, instability, calibration, uncertainty, and data-quality degradation).

## Core concept

MARLIN-AD unifies two anomaly lenses in a single workflow and API family:

- **Operational anomalies (data-level):** unusual patterns in AIS tracks, engine/sensor telemetry, metocean context, and operational logs.
- **Model-behaviour anomalies (AI-level):** changes in model-facing distributions and output behaviour that indicate reduced reliability.

A practical deployment generally needs both. A changed operating regime can induce model anomalies, while model degradation can mask or distort operational anomaly signals.

## Current scope (v0.1)

### Detection

- Typed detector interfaces (`fit`, `score`) returning structured `DetectionResult`.
- Baseline detectors:
  - `IsolationForestDetector`
  - `RobustZScoreDetector`
- Explicit score-threshold conversion helpers.

### Monitoring

- Deterministic monitor interfaces (`fit`, `evaluate`) returning structured `MonitorResult`.
- Implemented monitors:
  - `KSTestDriftMonitor`
  - `PredictionStabilityMonitor`
  - `SelfTestMonitor`
  - `FeatureImportanceDriftMonitor` (typed scaffold for richer attribution workflows)
- Contract-first stubs for calibration/concept/uncertainty expansion.

### Evaluation, orchestration, and interfaces

- Evaluation helpers for precision/recall/F1, robustness checks, and report scaffolds.
- Minimal orchestration pipelines and rule-based alert formatting.
- Dataset loading and validation utilities for maritime tabular data.
- Typed plugin and CLI entry-point structure.

## Installation

### Development install (recommended for research workflows)

```bash
pip install -e ".[dev,docs]"
```

### Package install

```bash
pip install marlin-ad
```

## Minimal CLI run

```bash
marlin-ad demo --rows 500 --seed 42
```

The demo performs one deterministic end-to-end pass:

- operational anomaly scoring on synthetic maritime-like telemetry,
- drift monitoring between reference and current windows.

You can also run the same pipeline on a local CSV:

```bash
marlin-ad demo --csv path/to/data.csv
```

List built-in datasets:

```bash
marlin-ad datasets list
```

## Minimal Python workflow

```python
import pandas as pd

from marlin_ad.detection import IsolationForestDetector
from marlin_ad.monitoring import KSTestDriftMonitor

reference = pd.DataFrame(
    {
        "speed_knots": [12.8, 13.2, 12.9, 13.1],
        "engine_temp_c": [79.1, 78.9, 79.3, 79.0],
    }
)
current = pd.DataFrame(
    {
        "speed_knots": [13.0, 17.8, 13.4, 13.1],
        "engine_temp_c": [79.2, 96.5, 80.1, 79.0],
    }
)

# A) Operational anomalies in current maritime data.
detector = IsolationForestDetector(random_state=42).fit(reference.to_numpy())
detection = detector.score(current.to_numpy())

# B) AI-behaviour anomalies via feature-distribution drift.
monitor = KSTestDriftMonitor(pvalue_threshold=0.01).fit(reference)
drift = monitor.evaluate(current)

print("anomaly_rate", float(detection.labels.mean()))
print("drift_alerts", drift.alerts)
```

## Documentation

The documentation in `docs/` is organized around manuscript- and operations-oriented reading paths:

- **Concepts**: dual-anomaly framing and architectural principles.
- **Maritime data**: schema assumptions, validation, and loader contracts.
- **Anomaly detection**: detector API and score semantics.
- **Model monitoring**: drift/stability/health monitoring API and interpretation.
- **Explainability**: optional audit-oriented explanation hooks.
- **Evaluation**: reproducible benchmarking and reporting checklist.
- **Minimal workflow**: compact runnable end-to-end example.
- **Implementation status**: explicit implemented-vs-scaffolded matrix (`docs/implementation-status.md`).

Build docs locally:

```bash
mkdocs build -f docs/mkdocs.yml
```

## Repository layout

- `src/marlin_ad/detection/` — maritime anomaly detectors
- `src/marlin_ad/monitoring/` — model/self-monitoring modules
- `src/marlin_ad/datasets/` — loaders and schema validation
- `src/marlin_ad/evaluation/` — metrics, robustness, reporting scaffolds
- `src/marlin_ad/explainability/` — adapters and report hooks
- `src/marlin_ad/pipelines/` — orchestration helpers
- `examples/` — executable reference examples
- `docs/` — MkDocs documentation

## Reproducibility baseline

Before publishing results derived from MARLIN-AD, run:

```bash
ruff check .
mypy src/marlin_ad
pytest -q
```

Record package version, dependency versions, random seeds, dataset versions, and split definitions.

## Citation

If MARLIN-AD contributes to academic or industrial work, cite using `CITATION.cff`.

## License

Apache-2.0 © 2026 Dr Tymoteusz Miller, kpt. Irmina Durlik.
