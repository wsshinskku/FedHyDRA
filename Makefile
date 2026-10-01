.PHONY: install test lint smoke

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

smoke:
	fedsoar train --config configs/smoke.yaml

