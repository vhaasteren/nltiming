.PHONY: test fast lint format check

test:
	@pytest tests/

fast:
	@pytest tests/ -m "not slow"

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

check:
	black --check src/ tests/
	ruff check src/ tests/
	pytest
