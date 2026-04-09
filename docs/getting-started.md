# Getting Started

## 1) Install development dependencies

From repository root:

```bash
pip install -e ".[dev,docs]"
```

This installs runtime requirements plus linting, type checking, testing, and documentation tooling.

## 2) Run the minimal CLI pipeline

```bash
marlin-ad demo --rows 500 --seed 42
```

The demo generates deterministic synthetic maritime-like data and performs:

1. anomaly detection on current operational data,
2. drift monitoring between reference and current windows.

Output includes anomaly-rate and drift-alert summaries.

## 3) Run on local CSV data

```bash
marlin-ad demo --csv path/to/data.csv
```

CSV expectations:

- at least one numeric column,
- optional `timestamp` column,
- enough rows to form reference/current windows.

## 4) Inspect built-in dataset loaders

```bash
marlin-ad datasets list
```

## 5) Run the Python minimal example

```bash
python examples/minimal_workflow.py
```

This script mirrors the documentation workflow in **User Guide → Minimal Workflow**.

## 6) Build documentation

```bash
mkdocs build -f docs/mkdocs.yml
```

## 7) Verify repository quality gates

```bash
ruff check .
mypy src/marlin_ad
pytest -q
```
