# MARLIN-AD

MARLIN-AD is a research-oriented, production-minded Python library for:

1. anomaly detection in **maritime operational data**, and
2. anomaly detection in **AI model behaviour**.

The library is designed for workflows where model outputs and operational decisions are tightly coupled. In that setting, trustworthy anomaly intelligence requires monitoring both the process and the model layer.

## Why a dual-anomaly framework?

In maritime deployments, risk conditions can arise from two interacting sources:

- **Data/process anomalies** (e.g., unusual vessel behaviour, sensor excursions, abrupt environmental regime changes), and
- **Model-behaviour anomalies** (e.g., feature drift, prediction instability, quality degradation signals).

Treating these as separate engineering concerns often yields incomplete diagnostics. MARLIN-AD provides typed interfaces to evaluate both in one reproducible pipeline.

## Documentation map

- [Getting Started](getting-started.md): installation, first deterministic run, and verification commands.
- [Concepts](user-guide/concepts.md): design goals, architecture, and core interfaces.
- [Maritime Data](user-guide/maritime-data.md): loader assumptions, schema validation, and dataset hygiene.
- [Anamaly Detection](user-guide/anomaly-detection.md): detector contracts, score semantics, and thresholding.
- [Model Monitoring](user-guide/model-monitoring.md): drift/stability/health checks and operational interpretation.
- [Explainability](user-guide/explainability.md): optional diagnostics and report integration patterns.
- [Evaluation](user-guide/evaluation.md): metric utilities, robustness patterns, and reporting checklist.
- [Minimal Workflow](user-guide/minimal-workflow.md): one compact end-to-end example with expected outputs.

## Quick start

```bash
pip install -e ".[dev,docs]"
marlin-ad demo --rows 500 --seed 42
```

For a Python-first path, see [Minimal Workflow](user-guide/minimal-workflow.md)
