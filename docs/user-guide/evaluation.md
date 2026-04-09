# Evaluation

Evaluation in MARLIN-AD is designed for both benchmarking and deployment validation. The key objective is reproducible evidence across detector and monitor layers.

## Recommended workflow

1. Define and freeze reference/current split strategy.
2. Fit detector(s) on reference data only.
3. Score current data and summarize anomaly statistics.
4. Evaluate monitor(s) on the same split.
5. Aggregate metrics and alerts into a structured report artifact.

## Minimal metric utility

```python
import numpy as np
from marlin_ad.evaluation import precision_recall_f1

metrics = precision_recall_f1(
    y_true=np.array([0, 0, 1, 1]),
    y_pred=np.array([0, 1, 1, 1]),
)
```

Returned keys:

- `precision`
- `recall`
- `f1`

## Robustness-oriented checks

Beyond point metrics, evaluate sensitivity to:

- thresholding policy,
- window size and split boundary,
- feature subset changes,
- injected perturbations and missingness,
- baseline refresh cadence for monitors.

## Reporting checklist (publication-ready)

Record at minimum:

- MARLIN-AD version and dependency versions,
- random seeds and deterministic settings,
- dataset identifiers/hashes and split definition,
- detector and monitor hyperparameters,
- thresholding and post-processing rules,
- alert-routing policy used during evaluation.

`marlin_ad.evaluation` includes benchmark and reporting scaffolds to support these artifacts.
