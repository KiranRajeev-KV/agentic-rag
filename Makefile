.PHONY: sync format lint test check qdrant-up qdrant-down

sync:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

check: format lint test

qdrant-up:
	docker compose up -d qdrant

qdrant-down:
	docker compose down
