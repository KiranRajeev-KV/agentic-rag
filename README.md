# Agentic RAG Hiring Assignment

## 1. Project summary

Local-first CLI agentic RAG system over recent arXiv `cs.AI` papers, designed for grounded answers with citations, clarification, refusal, and inspectable traces.

## 2. What this system can and cannot answer

To be implemented in subsequent milestones.

## 3. Quickstart

```bash
cp .env.example .env
uv sync
docker compose up -d qdrant
uv run app db init
uv run app --help
```

## 4. Demo commands

```bash
uv run app ingest --limit 20
uv run app ask "What do recent papers say about agent memory?" --debug
uv run app trace list --last 10
```

## 5. Architecture overview

To be implemented in subsequent milestones.

## 6. Locked design decisions

See `agentic_rag_implementation_context.md` and `docs/DECISIONS.md` (to be authored).

## 7. Corpus and ingestion

To be implemented in subsequent milestones.

## 8. Retrieval strategy

To be implemented in subsequent milestones.

## 9. Agent loop and memory

To be implemented in subsequent milestones.

## 10. Eval methodology

To be implemented in subsequent milestones.

## 11. Ablation results

To be implemented in subsequent milestones.

## 12. Observability/traces

To be implemented in subsequent milestones.

## 13. Known limitations

To be implemented in subsequent milestones.

## 14. What I would improve with more time

To be implemented in subsequent milestones.
