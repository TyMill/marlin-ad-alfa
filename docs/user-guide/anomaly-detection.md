# Anomaly Detection

Detectors in MARLIN-AD target **operational anomalies** in maritime data. The API is intentionally compact and typed.

## Detector API

- `fit(reference)`
- `score(current) -> DetectionResult`

`DetectionResult` exposes:

- `scores`: continuous anomaly evidence,
- `labels`: binary anomaly flags,
- `metadata`: method and threshold details.

## Basic usage

```python
from marlin_ad.detection import IsolationForestDetector

# reference/current are numeric arrays, often derived from DataFrames
model = IsolationForestDetector(random_state=42).fit(X_reference)
result = model.score(X_current)

scores = result.scores      # larger => stronger anomaly evidence
labels = result.labels      # binary labels (0/1)
metadata = result.metadata  # detector/threshold summary
```

## Implemented detectors (v0.1)

- **IsolationForestDetector**
  - robust multivariate baseline,
  - useful when feature relationships are heterogeneous.
- **RobustZScoreDetector**
  - median/MAD-style robust scoring,
  - useful for interpretable low-dimensional telemetry.

## Score thresholding

For explicit quantile thresholding:

```python
from marlin_ad.detection import threshold_scores

labels = threshold_scores(scores, quantile=0.95)
```

### Interpretation convention

In MARLIN-AD, higher score values represent stronger anomaly evidence.

## Experimental reporting guidance

For publication-quality detector reporting:

- report score distributions, not only binary rates,
- state thresholding policy (fixed value, quantile, adaptive),
- fix `random_state` for stochastic models,
- preserve preprocessing and feature definitions with run artifacts,
- evaluate sensitivity to split and threshold choices.
