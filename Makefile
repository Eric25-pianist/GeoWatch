.PHONY: install lint type test validate

install:
	python -m pip install -e .[dev]

lint:
	ruff check .

type:
	mypy src tests

test:
	pytest

validate:
	geowatch validate configs/default.yaml
