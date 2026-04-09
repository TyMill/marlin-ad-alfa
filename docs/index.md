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

- **Getting Started**: installation, first deterministic run, and verification commands.
- **Concepts**: design goals, architecture, and core interfaces.
- **Maritime Data**: loader assumptions, schema validation, and dataset hygiene.
- **Anomaly Detection**: detector contracts, score semantics, and thresholding.
- **Model Monitoring**: drift/stability/health checks and operational interpretation.
- **Explainability**: optional diagnostics and report integration patterns.
- **Evaluation**: metric utilities, robustness patterns, and reporting checklist.
- **Minimal Workflow**: one compact end-to-end example with expected outputs.

## Quick start

```bash
pip install -e ".[dev,docs]"
marlin-ad demo --rows 500 --seed 42
```

For a Python-first path, see **User Guide → Minimal Workflow**.
