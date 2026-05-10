# Decisions

## OpenAI Embeddings (`text-embedding-3-small`, 1536)
- Decision: Use OpenAI embeddings for child chunk indexing and query embedding.
- Why: Matches locked context and keeps indexing/retrieval alignment.
- Implementation details: `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=text-embedding-3-small`, `EMBEDDING_DIMENSIONS=1536`; Qdrant vectors are 1536 cosine.
- Known tradeoff: API/network dependency and usage cost.

## OpenAI Structured Outputs via Responses API
- Decision: Use OpenAI Responses API parse helper for all schema-bound control-plane calls.
- Why: Enforces schema-adherent structured outputs instead of JSON mode.
- Implementation details: `client.responses.parse(..., text_format=<PydanticModel>)` for router, evidence, answer, semantic-memory extraction, and LLM citation validation.
- Known tradeoff: Provider/API coupling; deterministic fallbacks required when unavailable.
- Conversation continuity is local-only: no `previous_response_id` and no OpenAI Conversations API.

## LangGraph SQLite Checkpointer Dependency
- Decision: Add `langgraph-checkpoint-sqlite` for durable thread-scoped checkpoints.
- Why: Required for LangGraph-local conversation persistence without changing provider or adding hosted infra.
- Implementation details: `SqliteSaver` stores checkpoints in the app SQLite DB; `app ask` passes `configurable.thread_id`.

## Citation Safety Policy
- Decision: Deterministic validator is hard gate; LLM citation validator is semantic second gate.
- Why: Deterministic checks guarantee mechanical correctness; LLM adds unsupported-claim screening.
- Implementation details: `citation.deterministic_validated` -> optional `citation.llm_validated` -> `citation.validated` outcome.
- Known tradeoff: Extra latency for LLM-enabled validation.

## Three-Layer Memory
- Decision: Conversation memory uses LangGraph SQLite checkpointing; semantic + episodic stay explicit SQLite tables.
- Why: Thread-scoped follow-up state is best handled by LangGraph persistence while keeping durable decisions and reportable episodes queryable in app DB.
- Implementation details:
  - Conversation: checkpointed `AgentState` (`messages`, summary/focus/active IDs) keyed by `thread_id`.
  - Semantic decisions: `semantic_memories`.
  - Episodic turns: `episodes`.
- Known tradeoff: checkpoint tables can grow with thread length; thread ID consistency is required.

## Thread-Scoped Ask Behavior
- Decision: Add `--thread-id` to ask flow (default `default`).
- Why: Follow-up resolution needs stable conversation boundary.
- Implementation details: router context includes conversation summary, recent turns, semantic memories, and recent episodes.
- Known tradeoff: User must reuse thread IDs for best follow-up behavior.

## Ingest/Index Split
- Decision: Preserve explicit split: `app ingest` (SQLite corpus artifacts) and `app index` (embeddings + Qdrant).
- Why: Keeps expensive embedding/index operations explicit and controllable.
- Implementation details: optional `app ingest --index` remains opt-in.
- Known tradeoff: First-time workflow is multi-step.

## Local Reset Requirement For This Revision
- Decision: Use schema source-of-truth reset flow; no backward migration compatibility promised.
- Why: Faster alignment with approved scope and simpler local operation.
- Implementation details: reset SQLite + recreate Qdrant local data before fresh ingest/index.
- Known tradeoff: Existing local runtime data must be regenerated.
