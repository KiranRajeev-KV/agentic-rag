set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

# List available recipes
@default:
    just --list

# Install project dependencies
setup:
    uv sync

# Start Qdrant container
qdrant-up:
    docker compose up -d qdrant

# Stop Qdrant container
qdrant-down:
    docker compose down

# Initialize SQLite schema (and optionally qdrant via CLI flags)
db-init:
    uv run app db init

# Reset local SQLite DB (requires CONFIRM=YES)
db-reset:
    if [[ "${CONFIRM:-}" != "YES" ]]; then
      echo "Refusing reset. Run: CONFIRM=YES just db-reset"
      exit 1
    fi
    uv run app db reset --yes

# Discover candidate corpus papers
@discover limit="20":
    uv run app corpus discover --limit {{limit}}

# Ingest papers into SQLite only
@ingest limit="20":
    uv run app ingest --limit {{limit}}

# Index into Qdrant; optional limit
@index limit="":
    if [[ -z "{{limit}}" ]]; then
      uv run app index
    else
      uv run app index --limit {{limit}}
    fi

# Ask a question with debug summary
@ask q:
    uv run app ask "{{q}}" --debug

# Ask within a named thread with debug summary
@ask-thread thread q:
    uv run app ask "{{q}}" --thread-id "{{thread}}" --debug

# Show recent traces
@trace-list last="10":
    uv run app trace list --last {{last}}

# Run eval for one retrieval variant
@eval variant="parent_child":
    uv run app eval run --variant {{variant}}

# Compare child_only vs parent_child eval summaries
eval-compare:
    uv run app eval compare --baseline child_only --candidate parent_child

# Run tests
test:
    uv run pytest

# Lint code
lint:
    uv run ruff check .

# Format code
format:
    uv run ruff format .

# Run format, lint, and tests
check:
    uv run ruff format .
    uv run ruff check .
    uv run pytest

# Print small-cost demo commands without running expensive API calls
demo-small:
    echo "uv run app db init"
    echo "uv run app ingest --limit 20"
    echo "uv run app index --limit 500"
    echo "uv run app ask \"What do recent papers say about agent memory?\" --thread-id demo --debug"
