.PHONY: install test lint typecheck run-scenarios grade-local clean

install:
	uv sync --extra dev --extra google --extra sqlite

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

run-scenarios:
	uv run python -m langgraph_agent_lab.cli run-scenarios --config configs/lab-sqlite.yaml --output outputs/metrics.json

grade-local:
	uv run python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json
