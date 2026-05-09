# Decisions

## OpenAI Embeddings (`text-embedding-3-small`, 1536)
- Decision: Use OpenAI embeddings for child chunk indexing and query embedding.
- Why: Aligns with locked implementation context and keeps embedding behavior consistent between indexing and retrieval.
- Implementation details: `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=text-embedding-3-small`, `EMBEDDING_DIMENSIONS=1536`; Qdrant collection vectors are 1536 cosine.
- Tradeoff: Requires network/API key and incurs usage cost.

## OpenAI LLM Control Plane (`gpt-5-nano`)
- Decision: Use `gpt-5-nano` for router, evidence classification, and answer generation through a small OpenAI client abstraction.
- Why: Keeps behavior-first control logic structured while preserving deterministic fallbacks for reliability and tests.
- Implementation details: JSON-structured outputs validated with Pydantic schemas; fallback heuristics activate when API key/call is unavailable.
- Tradeoff: Runtime behavior depends on API availability; fallbacks are less nuanced.

## Parent-Child Retrieval With Parent Scoring
- Decision: Keep child-only baseline and parent-child candidate with parent scoring.
- Why: This is the core retrieval technique under evaluation and ablation.
- Implementation details: Child vectors in Qdrant, grouped by parent, scored with capped support bonus and section adjustments.
- Tradeoff: More orchestration complexity than pure child-only retrieval.

## Ingest/Index Split
- Decision: Preserve explicit split: `app ingest` (SQLite corpus artifacts) and `app index` (embeddings + Qdrant).
- Why: Keeps heavy embedding/index work explicit and reproducible.
- Implementation details: Optional `app ingest --index` remains opt-in convenience.
- Tradeoff: First-time setup is two steps unless convenience flag is used.
