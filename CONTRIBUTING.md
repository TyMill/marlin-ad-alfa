# Contributing

Thanks for considering contributing to MARLIN-AD.

## Development setup
```bash
pip install -e ".[dev,docs]"
pre-commit install
pytest
```
## Pull requests
- Keep changes focused and well-tested.
- Add or update documentation under `docs/`.
- Use type hints and keep public APIs stable.

## Style
- Formatting and linting via `ruff`.
- Typing via `mypy` (strict).
