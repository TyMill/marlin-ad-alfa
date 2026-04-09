# Explainability

Explainability in MARLIN-AD is an **optional diagnostics layer** over detector and monitor outputs. It is not intended as a substitute for statistical validation.

## Current support in v0.1

- detector/monitor outputs can carry explanatory metadata fields,
- explainability modules provide report and adapter scaffolds,
- optional SHAP integration is available without imposing hard runtime dependency in base installs.

## Technical role in workflow

A disciplined workflow is:

1. define detector/monitor configuration and thresholds,
2. evaluate quantitative performance and robustness,
3. add explanations to contextualize flagged samples/windows,
4. separate exploratory explanations from causal claims.

This sequence helps avoid over-interpreting post hoc explanations.

## Suggested report structure

For each anomaly episode:

- **Signal summary**: anomaly score distribution and monitor alerts,
- **Feature context**: top contributing features (if available),
- **Operational context**: route, metocean, and system-mode metadata,
- **Actionability note**: whether alert supports intervention, triage, or observation only.

## Extension directions

Planned and scaffolded areas include:

- richer SHAP-based adapters under optional extras,
- stable explainability report schemas for pipeline integration,
- stronger links between feature-attribution drift and model monitoring outputs.
