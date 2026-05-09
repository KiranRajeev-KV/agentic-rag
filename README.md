# Agentic RAG Hiring Assignment

## 1. Project summary

Local-first CLI agentic RAG system over recent arXiv `cs.AI` papers, designed for grounded answers with citations, clarification, refusal, and inspectable traces.

## 2. What this system can and cannot answer

Can answer:
- Questions grounded in indexed recent `cs.AI` papers.
- Metadata/search requests via arXiv tool route.
- Follow-ups with project-scoped memory and traceable decisions.

Cannot answer:
- General web knowledge outside indexed corpus.
- Unsupported certainty when evidence is weak.
- Non-corpus domains (weather, finance, etc.).

## 3. Quickstart

```bash
cp .env.example .env
uv sync
docker compose up -d qdrant
uv run app db init
uv run app ingest --limit 20
uv run app index
uv run app ask "What do recent papers say about agent memory?" --debug
uv run app --help
```

## 4. Demo commands

```bash
uv run app ingest --limit 20
uv run app index
uv run app ask "What do recent papers say about agent memory?" --debug
uv run app eval run --variant child_only
uv run app eval run --variant parent_child
uv run app eval compare --baseline child_only --candidate parent_child
uv run app trace list --last 10
```

## 5. Architecture overview

- Ingest: arXiv discovery/filtering -> PDF download -> Docling parse -> HybridChunker parent/child model in SQLite.
- Index: BGE-M3 dense embeddings for child chunks -> Qdrant upsert with compact payload.
- Ask: LangGraph state graph with structured router, retrieval/tool/clarify/refuse actions, evidence gate, citation checks, memory and trace writes.
- Eval: 14 curated behavior-first cases and child-only vs parent-child ablation compare.

## 6. Locked design decisions

See `agentic_rag_implementation_context.md` and `docs/DECISIONS.md` (to be authored).

## 7. Corpus and ingestion

- Corpus scope: recent arXiv `cs.AI` with deterministic relevance filtering.
- Command: `uv run app ingest --limit 20`
- Output: paper metadata, parent sections, child chunks, parse/chunk metadata in SQLite.

## 8. Retrieval strategy

- Baseline: child-only dense retrieval.
- Candidate: parent-child retrieval with parent scoring.
- Parent score: max child score + capped support bonus + section adjustment - references penalty.

## 9. Agent loop and memory

- LangGraph nodes: load state, route, retrieve/tool, evidence check, answer/refuse/clarify, citation validation, memory update.
- Memory: semantic decision memory in SQLite and episodic turn traces.

## 10. Eval methodology

- 14 curated behavior-first cases (direct QA, synthesis, clarify, refuse, tool).
- Per-case 10-point rubric with route/retrieval/evidence/answer/citation/memory/tool/trace components.

## 11. Ablation results

Run:
```bash
uv run app eval run --variant child_only
uv run app eval run --variant parent_child
uv run app eval compare --baseline child_only --candidate parent_child
```

## 12. Observability/traces

- Trace events stored in SQLite trace tables.
- Inspect with:
  - `uv run app trace list --last 10`
  - `uv run app trace show <trace_id>`

## 13. Known limitations

See `docs/LIMITATIONS.md`.

## 14. What I would improve with more time

- Add calibrated thresholds from larger evaluation sets.
- Improve contradiction handling granularity.
- Add richer action-specific scoring and deeper trace visualizations.
