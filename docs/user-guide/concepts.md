# Concepts

## Problem framing

MARLIN-AD is organized around one operational statement:

> In safety-critical AI systems, anomalies can emerge in both the observed process and the model interpreting that process.

Accordingly, MARLIN-AD tracks two anomaly classes:

1. **Operational anomalies (data-level):** unusual trajectories, telemetry excursions, log irregularities, or regime shifts in maritime observations.
2. **Model-behaviour anomalies (AI-level):** changes in model-facing distributions or outputs that suggest degraded reliability.

These classes are distinct but causally entangled in practice. A new operating regime can induce model drift; a degraded model can suppress or distort anomaly signals from the process.

## Architectural principle

The repository is intentionally modular:

- `marlin_ad.detection`: operational anomaly detectors,
- `marlin_ad.monitoring`: model-behaviour monitors,
- `marlin_ad.evaluation`: metrics and robustness/reporting scaffolds,
- `marlin_ad.datasets`: typed loading and validation,
- `marlin_ad.explainability`: optional diagnostics,
- `marlin_ad.pipelines` and `marlin_ad.alerting`: orchestration and response plumbing.

This decomposition supports both scientific reproducibility (clear contracts) and operational maintainability (bounded module responsibilities).

## Interface contracts

### Detector contract

- `fit(reference)`
- `score(current) -> DetectionResult`

`DetectionResult` contains score vectors, binary labels, and detector metadata. This allows benchmarking and alerting systems to consume a stable envelope rather than model-specific internals.

### Monitor contract

- `fit(reference)`
- `evaluate(current) -> MonitorResult`

`MonitorResult` contains metrics, alerts, and monitor metadata. This keeps drift/stability/health checks composable inside pipelines and reporting layers.

## Reproducibility conventions

For manuscript-grade usage:

- set explicit random seeds for stochastic components,
- keep fixed reference/current split definitions,
- version feature engineering and preprocessing choices,
- report detector/monitor thresholds with rationale,
- archive software and dataset versions with run metadata.
