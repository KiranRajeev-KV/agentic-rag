# Decisions

## Parent-child retrieval with parent scoring
- Decision: Use child chunk vector search plus parent grouping and parent scoring.
- Why: Improves section-level grounding while keeping vector index on smaller child chunks.
- Implementation details: Child chunks are embedded/indexed; parent score uses max child score, capped support bonus, section adjustment, and references penalty.
- Tradeoff: More moving parts than child-only baseline.

## Dense-only BGE-M3 embeddings
- Decision: Use `BAAI/bge-m3` dense mode only.
- Why: Locked assignment scope and local-first setup.
- Implementation details: Embedding uses `BGEM3FlagModel` dense vectors; no sparse or colbert outputs.
- Tradeoff: Misses potential gains from hybrid/sparse retrieval.

## Storage split
- Decision: Qdrant for vectors, SQLite for relational/docstore/memory/evals/traces.
- Why: Keeps vector ops fast and metadata inspectable.
- Implementation details: Qdrant payload is compact metadata; parent text stays in SQLite.
- Tradeoff: Requires coordination across two stores.

## Ingest/index split
- Decision: `app ingest` writes SQLite corpus artifacts; `app index` handles embeddings+Qdrant.
- Why: Prevents accidental heavy model/index work and improves reproducibility.
- Implementation details: Optional `app ingest --index` exists, default is SQLite-only.
- Tradeoff: Two-step workflow for first-time setup.
