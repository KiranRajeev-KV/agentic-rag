# Demo Script

## Setup
```bash
cp .env.example .env
uv sync
docker compose up -d qdrant
uv run app db init
```

## Ingest + index (small demo)
```bash
uv run app ingest --limit 20
uv run app index --limit 500
```

## Ask with debug trace summary
```bash
uv run app ask "What do recent papers say about agent memory?" --debug
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
