PHONY: test lint typecheck docs

test:
	pytest -q

lint:
	ruff check .

typecheck:
	mypy src/marlin_ad

docs:
	mkdocs build
