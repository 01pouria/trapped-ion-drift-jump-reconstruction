.PHONY: install test notebook full robust validate frozen

install:
	python -m pip install -e ".[dev,notebook]"

test:
	pytest

notebook:
	jupyter lab notebooks/paper_results.ipynb

full:
	python scripts/reproduce_full.py

robust:
	python scripts/run_robustness.py

validate:
	python scripts/run_validation.py

frozen:
	python scripts/check_frozen_results.py
