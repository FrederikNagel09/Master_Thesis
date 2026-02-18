check:
	uv run ruff format . 
	uv run ruff check . --fix
	uv run mypy .

.PHONY: start
start:
	@set -e
	@echo "Pulling latest changes..."
	git pull
	@echo "Syncing virtual env..."
	uv sync