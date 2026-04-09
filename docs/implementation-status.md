# Implementation status (v0.1)

This page clarifies what is implemented now versus intentionally scaffolded.

## Implemented and production-usable in v0.1

### Detection
- ✅ `IsolationForestDetector` (`marlin_ad.detection.classical.isolation_forest`)
- ✅ `RobustZScoreDetector` (`marlin_ad.detection.classical.robust_stats`)
- ✅ score-threshold helpers (`marlin_ad.detection.scoring`)

### Monitoring
- ✅ `KSTestDriftMonitor` (`marlin_ad.monitoring.drift.data_drift`)
- ✅ `PredictionStabilityMonitor` (`marlin_ad.monitoring.stability.prediction_stability`)
- ✅ `SelfTestMonitor` (`marlin_ad.monitoring.health.self_tests`)
- ✅ `FeatureImportanceDriftMonitor` baseline checks (`marlin_ad.monitoring.stability.feature_importance_drift`)

### Data and orchestration
- ✅ Dataset registry and typed validation (`marlin_ad.datasets.registry`, `marlin_ad.datasets.validation`)
- ✅ CSV end-to-end orchestration pipeline (`marlin_ad.pipelines.end_to_end`)
- ✅ Rule-based alert formatting and sinks (`marlin_ad.alerting`)
- ✅ Plugin registry and builtin plugin hooks (`marlin_ad.plugins`)
- ✅ CLI demo and dataset listing commands (`marlin_ad.cli.main`)

### Evaluation
- ✅ Core detection metrics (`marlin_ad.evaluation.metrics`)
- ✅ Robustness checks and report helpers (`marlin_ad.evaluation.robustness`, `marlin_ad.evaluation.reports`)

## Intentionally scaffolded / contract-first (not fully implemented yet)

The following modules intentionally raise `NotImplementedError` for non-trivial logic while keeping typed interfaces stable:

- ⚠️ `ConceptDriftMonitor` (`marlin_ad.monitoring.drift.concept_drift`)
- ⚠️ `CalibrationMonitor` (`marlin_ad.monitoring.drift.calibration`)
- ⚠️ `UncertaintyMonitor` (`marlin_ad.monitoring.uncertainty.monitor`)
- ⚠️ `VAEDetector` (`marlin_ad.detection.deep.vae`) — deep backend intentionally optional
- ⚠️ `send_webhook` (`marlin_ad.alerting.sinks.webhook`) — transport interface reserved for future delivery backends

## Non-goals in v0.1

- No hard dependency on deep-learning frameworks.
- No live serving stack as a mandatory dependency.
- No automatic remote alert transport beyond local/stdout sinks.

## Recommended V2 priorities

1. Implement concept drift with typed detector-agnostic statistics + windowing policies.
2. Implement calibration drift health metrics with confidence interval reporting.
3. Add optional uncertainty backends under extras while preserving typed fallbacks.
4. Add robust webhook delivery (retry/backoff/signature/auth) with integration tests.
