# Agentic RAG Hiring Assignment

## 1. Project summary

Local-first CLI agentic RAG system over recent arXiv `cs.AI` papers, designed for grounded answers with citations, clarification/refusal behavior, inspectable traces, and behavior-first evaluation.

## 2. What this system can and cannot answer

Can answer:
- Questions grounded in indexed recent `cs.AI` papers.
- Metadata/search requests via arXiv tool route.
- Follow-ups inside a shared `--thread-id` using LangGraph SQLite checkpoints + episodic memory.

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
uv run app ask "What do recent papers say about agent memory?" --thread-id demo --debug
uv run app --help
```

## 4. Demo commands

```bash
uv run app ingest --limit 20
uv run app index
uv run app ask "What do recent papers say about agent memory?" --thread-id demo --debug
uv run app eval run --variant child_only
uv run app eval run --variant parent_child
uv run app eval compare --baseline child_only --candidate parent_child
uv run app trace list --last 10
```

## 5. Architecture overview

- Ingest: arXiv discovery/filtering -> PDF download -> Docling parse -> HybridChunker parent/child model in SQLite.
- Index: OpenAI `text-embedding-3-small` child embeddings (1536 dims) -> Qdrant upsert with compact payload.
- Ask: LangGraph state graph with structured router (`gpt-5-nano`), retrieval/tool/clarify/refuse actions, evidence classification, citation checks, memory and trace writes.
- Eval: 14 curated behavior-first cases and child-only vs parent-child ablation compare.

## 6. Locked design decisions

See `agentic_rag_implementation_context.md` and `docs/DECISIONS.md`.

## 7. Corpus and ingestion

- Corpus scope: recent arXiv `cs.AI` with deterministic relevance filtering.
- Command: `uv run app ingest --limit 20`
- Output: paper metadata, parent sections, child chunks, parse/chunk metadata in SQLite.
- Persistence includes discovery metadata: source query, matched filter terms, and filter reason.

## 8. Retrieval strategy

- Baseline: child-only dense retrieval.
- Candidate: parent-child retrieval with parent scoring.
- Parent score: max child score + capped support bonus + section adjustment - references penalty.

## 9. Agent loop and memory

- LangGraph nodes: load state, route, retrieve/tool, evidence check, answer/refuse/clarify, citation validation, memory update.
- Three-layer memory:
  - Conversation memory (LangGraph SQLite checkpoint state scoped by `thread_id`).
  - Semantic decision memory (`semantic_memories`).
  - Episodic trace memory (`episodes`).
- Thread scope: `uv run app ask "..." --thread-id demo`.

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
- Includes router/retrieval/tool/evidence/citation/memory/episode/eval events with compact structured payloads.
- Inspect with:
  - `uv run app trace list --last 10`
  - `uv run app trace show <trace_id>`

## 13. Known limitations

See `docs/LIMITATIONS.md`.

## 14. What I would improve with more time

- Better follow-up resolution across longer threads with stronger summary refresh logic.
- Additional citation repair strategies while preserving strict deterministic safety gate.
- Larger eval set for stronger threshold calibration.
- Richer trace visualization tooling.

## Structured outputs + local reset notes

- Schema-bound LLM calls use OpenAI Responses API strict structured parsing (`responses.parse` with Pydantic models).
- Old local DB/Qdrant layouts are not kept backward-compatible for this change. Reset local data:

```bash
uv run app db reset --yes
docker compose down
rm -rf data/qdrant
docker compose up -d qdrant
```
