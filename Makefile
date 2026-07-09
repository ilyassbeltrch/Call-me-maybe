.PHONY: install run debug clean lint lint-strict

SRC := src

install:
	@uv sync

run:
	@uv run python -m $(SRC)

debug:
	@uv run python -m pdb -m $(SRC)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	@uv run flake8 src
	@uv run mypy src

lint-strict:
	@uv run flake8 src
	@uv run mypy src --strict