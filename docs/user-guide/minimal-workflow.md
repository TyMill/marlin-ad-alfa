# Minimal Workflow

This page provides one compact workflow that combines:

1. **operational anomaly detection** on maritime-like telemetry, and
2. **model-behaviour monitoring** via feature-distribution drift.

## End-to-end example

```python
import numpy as np
import pandas as pd

from marlin_ad.detection import IsolationForestDetector
from marlin_ad.monitoring import KSTestDriftMonitor

rng = np.random.default_rng(42)

reference = pd.DataFrame(
    {
        "speed_knots": rng.normal(13.0, 1.0, size=300),
        "engine_temp_c": rng.normal(79.0, 1.8, size=300),
        "course_change": rng.normal(0.0, 3.0, size=300),
    }
)

current = pd.DataFrame(
    {
        "speed_knots": rng.normal(14.2, 1.2, size=300),
        "engine_temp_c": rng.normal(81.5, 2.3, size=300),
        "course_change": rng.normal(0.0, 3.5, size=300),
    }
)

# Inject sparse operational anomalies.
idx = rng.choice(len(current), size=20, replace=False)
current.loc[idx, "engine_temp_c"] += rng.normal(15.0, 3.0, size=len(idx))

# A) Operational anomalies in maritime data
detector = IsolationForestDetector(random_state=42).fit(reference.to_numpy())
detection = detector.score(current.to_numpy())
anomaly_rate = float(detection.labels.mean())

# B) AI-behaviour anomalies (drift in model-facing features)
monitor = KSTestDriftMonitor(pvalue_threshold=0.01).fit(reference)
drift = monitor.evaluate(current)

print("anomaly_rate", round(anomaly_rate, 3))
print("drift_alert_count", len(drift.alerts))
print("drift_alerts", drift.alerts)
```

## What to expect

Because the current window is shifted and includes injected temperature excursions:

- `anomaly_rate` should be materially above zero,
- drift alerts should identify one or more shifted features.

Exact values may vary by dependency versions, but qualitative behavior should remain stable with the fixed seed.

## Why this is the canonical minimal workflow

The pattern mirrors real deployments:

- detector output measures *process-level* abnormality,
- monitor output measures *model-context* reliability shift.

Using both outputs together gives a more defensible risk signal than either in isolation.

A runnable script version is available at `examples/minimal_workflow.py`.
