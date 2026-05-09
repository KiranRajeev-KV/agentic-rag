# Demo Script

## Setup
```bash
cp .env.example .env
uv sync
docker compose up -d qdrant
uv run app db init
```

## Required local reset for this revision
```bash
uv run app db reset --yes
docker compose down
rm -rf data/qdrant
docker compose up -d qdrant
```

## Ingest + index (small demo)
```bash
uv run app ingest --limit 20
uv run app index --limit 500
```

## Ask with thread + debug summary
```bash
uv run app ask "What do recent papers say about agent memory?" --thread-id demo --debug
uv run app ask "what about it" --thread-id demo --debug
```

## Trace inspection
```bash
uv run app trace list --last 10
uv run app trace show <trace_id>
```

## Eval + ablation
```bash
uv run app eval run --variant child_only
uv run app eval run --variant parent_child
uv run app eval compare --baseline child_only --candidate parent_child
```

## Optional Justfile shortcuts
```bash
just setup
just qdrant-up
just ingest 20
just index 500
just ask-thread demo "What do recent papers say about agent memory?"
just trace-list 10
```
