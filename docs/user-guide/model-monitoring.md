# Model Monitoring

Model monitoring in MARLIN-AD targets **AI-behaviour anomalies**: evidence that model context or outputs are diverging from reference conditions.

## Monitor API

- `fit(reference)`
- `evaluate(current) -> MonitorResult(metrics, alerts, metadata)`

This contract makes drift/stability/health checks composable in evaluation and pipeline layers.

## Example usage

```python
from marlin_ad.monitoring import KSTestDriftMonitor, PredictionStabilityMonitor, SelfTestMonitor

# 1) Drift in model-facing input features
ks = KSTestDriftMonitor(pvalue_threshold=0.01, min_samples=50).fit(reference_df)
ks_result = ks.evaluate(current_df)

# 2) Prediction stability checks
stability = PredictionStabilityMonitor(
    mean_tolerance=0.2,
    std_ratio_tolerance=0.25,
    psi_tolerance=0.2,
).fit(reference_predictions)
stability_result = stability.evaluate(current_predictions)

# 3) Stream integrity checks
health = SelfTestMonitor(min_sample_ratio=0.5, max_nonfinite_fraction=0.0).fit(reference_predictions)
health_result = health.evaluate(current_predictions)
```

## Implemented monitors (v0.1)

- **KSTestDriftMonitor**: deterministic, feature-wise KS testing for numeric inputs.
- **PredictionStabilityMonitor**: mean/variance and PSI-style output-shift diagnostics.
- **SelfTestMonitor**: sample volume and non-finite integrity checks.
- **FeatureImportanceDriftMonitor**: typed attribution-shift scaffold.

Additional contract-first monitor modules exist for concept drift, calibration drift, and uncertainty monitoring.

## Interpreting alerts

Monitor alerts should be treated as reliability evidence, not direct proof of model failure.

Recommended interpretation combines:

- detector anomaly rates,
- monitor alert profiles,
- route/operating-mode/environment context,
- escalation policy and human review thresholds.

## Practical monitoring protocol

- fit monitor baselines on a well-characterized reference period,
- keep window sizes and update cadence explicit,
- monitor false-alert burden and retune thresholds conservatively,
- document all threshold and baseline update changes in run metadata.
