# Examples

## `minimal_workflow.py`

A compact, executable reference for the core MARLIN-AD pattern:

1. detect anomalies in maritime-like operational data,
2. monitor drift in model-facing feature distributions.

Run from repository root:

```bash
python examples/minimal_workflow.py
```

Expected behavior:

- non-zero anomaly rate due to injected operational excursions,
- one or more drift alerts due to shifted current distributions.

This example is intentionally minimal and deterministic (fixed seed) so it can be reused in tests, tutorials, and manuscript appendices.
