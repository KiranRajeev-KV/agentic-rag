# Design Decisions

This project evolved as an experimentation environment rather than from a single fixed architecture. The decisions below document the current implementation, the problem each choice was intended to address, and the trade-offs that remain.

## 1. Fixed corpus over live discovery

**Decision**

The project uses a fixed `corpus_manifest.json` with pinned arXiv IDs, versions, and PDF checksums instead of live discovery at runtime.

**Why**

Retrieval and evaluation experiments need a stable, repeatable dataset. Pinning paper IDs, versions, and SHA-256 checksums ensures the same documents are used across runs, making ablation comparisons and debugging deterministic. Local ingestion avoids silently changing the experimental dataset.

**Trade-off**

The corpus becomes stale unless deliberately refreshed. It does not represent general web knowledge or newly published papers.

**Implementation**

`corpus_manifest.json` defines 100 papers with `arxiv_id`, `version`, `pdf_sha256`, and metadata. `app ingest` reads the manifest and local PDFs from `data/raw_pdfs/`. A download script fetches only missing or corrupt files and verifies checksums against the manifest.

## 2. Separate ingestion from indexing

**Decision**

`app ingest` parses PDFs and writes document artifacts to SQLite. `app index` performs embedding and Qdrant upserts. They are distinct commands.

**Why**

Embedding calls introduce network latency, API cost, and provider coupling. Parsing and chunking experiments should not automatically force re-embedding. Keeping indexing explicit lets developers control when and how vectors are generated.

**Trade-off**

First-time setup requires multiple steps. Users can accidentally have SQLite and Qdrant out of sync if the workflow is misused.

**Implementation**

`app ingest --limit N` runs Docling parsing, HybridChunker parent/child chunking, and writes paper metadata, parent sections, and child chunks to SQLite. `app index` reads child chunks from SQLite, computes OpenAI `text-embedding-3-small` vectors (1536 dims), and upserts to Qdrant. An optional `app ingest --index` combines both for convenience.

## 3. Retrieve children, expand to parents

**Decision**

The system embeds and retrieves smaller child chunks but retains larger parent sections for context assembly.

**Why**

Small chunks are precise retrieval units that match specific query terms. However, isolated chunks often lose surrounding section context needed for generation. Expanding to parent sections after retrieving relevant children restores broader context.

**Trade-off**

Parent expansion consumes more context window. Parent selection and scoring introduce additional heuristics.

**Implementation**

Docling + HybridChunker produces a parent/child hierarchy stored in SQLite. Qdrant indexes only child chunk vectors (1536 dims). At retrieval time, child hits are grouped by parent. The `parent_child` variant aggregates and scores parents; `child_only` keeps child groups separate as a baseline.

## 4. Heuristic parent aggregation

**Decision**

For the `parent_child` variant, child hits are grouped by parent and scored using: maximum child similarity score, a bounded support bonus for multiple matching children from the same parent, a small section-type adjustment, and a penalty for reference sections.

**Why**

One strong matching child is useful evidence. Repeated supporting child matches can increase confidence. Reference sections often contain high lexical/semantic overlap while providing poor answer context, so they are penalized.

**Trade-off**

Coefficients are hand-authored heuristics, not learned ranking weights. Section-type rules may not generalize to every corpus. This is not a learned reranker.

**Implementation**

`score_parent_groups()` in `src/agentic_rag/retrieval/parent_scoring.py` groups child hits by parent and computes: `max_child_score + min(support_bonus * (supporting_children - 1), cap) + section_adjustment - references_penalty`. Scores are used to rank parents and select the top-k within the dynamic context budget.

## 5. Keep a child-only baseline

**Decision**

The `child_only` retrieval variant groups results without parent aggregation, treating each child hit as its own group. It exists as a comparison baseline alongside `parent_child`.

**Why**

Having both paths in the same system makes it possible to compare retrieval and context strategies using identical infrastructure, evaluation cases, and trace infrastructure.

**Trade-off**

`child_only` can select the same parent multiple times because each child forms its own group, potentially wasting parent/context budget.

**Implementation**

`RetrievalService` branches on the variant parameter. `child_only` skips parent scoring and uses child groups directly for context assembly. `parent_child` runs the parent aggregation and scoring pipeline. Both share the same vector retrieval, lexical fallback, budget, and evidence gate logic.

## 6. Conditional lexical fallback

**Decision**

Primary retrieval is vector similarity over child chunks in Qdrant. If fewer than 8 child hits are returned, the system attempts SQLite lexical retrieval and merges unseen lexical results with heuristic scores.

**Why**

Vector retrieval can return too few usable results for certain queries. Lexical matching can recover exact terminology, identifiers, or paper-specific language that dense embeddings miss.

**Trade-off**

This is a fallback merge with heuristic scores, not a proper sparse+dense fusion algorithm. It is not BM25/vector reranking or learned hybrid search. Use the phrase "conditional lexical fallback", not "hybrid retrieval".

**Implementation**

`RetrievalService.run()` checks the child hit count after vector search. If below the threshold, it runs a `LIKE` query against child text in SQLite, merges unseen results with a fixed heuristic score, and proceeds to grouping. The threshold and heuristic score are hardcoded constants.

## 7. Budget context by query shape

**Decision**

Context assembly selects a variable number of parents based on detected query shape: default 3, comparison-like 5, synthesis/survey-like 6, metadata-like 0. Explicit modes can also be requested.

**Why**

A narrow question usually needs less context. Comparison and synthesis questions benefit from more source breadth. Blindly sending a large fixed context wastes model context and can dilute evidence.

**Trade-off**

Query-shape detection is heuristic. Budget values are manually selected and not learned.

**Implementation**

`dynamic_parent_budget()` uses lightweight keyword heuristics over the query and returns the parent budget. `assemble_context_packets()` applies the selected parent groups to context construction.

## 8. Make evidence sufficiency explicit

**Decision**

Retrieval results are passed through an evidence gate that returns SUFFICIENT, AMBIGUOUS, or INSUFFICIENT based on signals including: top child score, top parent score, second parent score, score margin, supporting child count, total support across leading parents, distinct parent count, distinct paper count, and section-type distribution.

**Why**

"Documents were retrieved" is not equivalent to "enough evidence exists to answer." Explicit signals make the decision inspectable and allow the system to route to clarification, refusal, or recovery instead of generating unsupported answers.

**Trade-off**

Thresholds are hand-set heuristics. Current thresholds are not validated on a large benchmark. Confidence bands are coarse (three categories).

**Implementation**

`evaluate_evidence()` in `src/agentic_rag/retrieval/evidence_gate.py` computes the signals from retrieved parent groups and applies threshold logic. The gate output drives the LangGraph `evidence_check` node branching.

## 9. Recover once before giving up

**Decision**

If evidence is AMBIGUOUS or INSUFFICIENT, `RetrievalService` performs one broader retrieval attempt: retrieve more child hits, permit reference sections, rebuild grouping/context/evidence signals.

**Why**

A narrow first pass should not immediately force refusal if a broader search could recover useful evidence.

**Trade-off**

Broader retrieval increases work and context candidates. Only one explicit recovery expansion is implemented. This is retrieval recovery, not autonomous planning.

**Implementation**

`RetrievalService.retrieve()` calls the internal retrieval pipeline twice when the first pass yields ambiguous or insufficient evidence. The second pass uses a higher child limit and includes reference sections. Results are re-grouped, re-scored, and re-evaluated. No additional LangGraph node is involved.

## 10. Route before executing

**Decision**

The LangGraph flow does not send every user request through retrieval. The router can choose RETRIEVE, TOOL, CLARIFY, REFUSE, or ANSWER. Deterministic routing handles explicit arXiv metadata/search intents before general LLM routing.

**Why**

Metadata search, corpus QA, ambiguous requests, and out-of-domain requests are different operations. Explicitly modeling these paths makes behavior inspectable and testable.

**Trade-off**

Routing contains heuristics and confidence thresholds. Fallback routing is intentionally simpler than model-driven routing.

**Implementation**

`route_query` node in `src/agentic_rag/agent/graph.py` runs deterministic keyword checks for arXiv tool intents first. If no deterministic match, it calls an LLM router with a Pydantic output schema. The router output determines the next node.

## 11. Use structured outputs for control-plane decisions

**Decision**

Schema-bound control-plane LLM operations use Pydantic models with OpenAI Responses API structured parsing (`responses.parse`).

**Why**

Control-plane decisions (routing, evidence assessment, answer behavior, semantic memory extraction, LLM citation validation) should have predictable fields rather than relying on arbitrary JSON or text parsing.

**Trade-off**

Ties these calls to provider capabilities. Schema validity does not guarantee semantic correctness.

**Implementation**

Pydantic models define expected outputs for router, evidence gate, answer classifier, semantic memory extractor, and LLM citation validator. `client.responses.parse()` enforces the schema at the API level. Deterministic fallbacks exist for critical paths.

## 12. Validate citations in layers

**Decision**

Citation validation has two layers. First: deterministic checks (cited IDs must be known, must appear in Sources block, context answers require S# citations, tool answers require T# citations, contradiction answers require multiple source citations, internal IDs must not leak). Second: optional LLM-assisted semantic validation for unsupported-claim screening.

**Why**

Deterministic checks enforce citation mechanics reliably. Semantic validation attempts to identify unsupported claims that mechanical validation cannot detect.

**Trade-off**

Semantic validation can be inconsistent. Traces show it can reject an otherwise successful tool result. Fail-closed behavior favors unsupported-answer prevention over answer availability.

**Implementation**

`validate_citations()` runs deterministic citation invariants first. An optional structured LLM validation step follows for unsupported-claim screening. The combined outcome is `citation.validated`. The pipeline runs after `answer` and `tool` nodes, before `memory_update`.

## 13. Separate conversation, semantic, and episodic memory

**Decision**

Three distinct memory concerns: conversation memory (LangGraph SQLite checkpoints keyed by `thread_id`), semantic memory (explicit SQLite records for reusable decisions/context), episodic memory (explicit SQLite episode records for completed turns).

**Why**

These have different lifecycles and purposes. Conversation state supports follow-up turns. Semantic memory accumulates reusable extracted decisions. Episodic records provide an inspectable turn history. They should not be represented as one generic "memory" blob.

**Trade-off**

More persistence mechanisms mean more state to understand. Checkpoint growth needs management. Thread consistency matters for follow-up quality.

**Implementation**

LangGraph `SqliteSaver` stores checkpoints in the app SQLite DB. `semantic_memories` stores explicit reusable semantic records such as kind/key/value, confidence, source turn, and lifecycle metadata. The current implementation does not embed these records or perform vector-similarity retrieval over semantic memory. `episodes` table stores turn-level summaries including route, retrieved IDs, tools, final action, and short answer summary.

## 14. Split vector search from relational state

**Decision**

Qdrant handles child vector similarity search. SQLite holds corpus/document metadata, parent/child relational structure, checkpoints, semantic memory, episodic records, traces, evaluation state/results, and lexical fallback source.

**Why**

Vector similarity search and relational/inspectable application state have different storage requirements. Qdrant is optimized for ANN search. SQLite keeps everything else locally queryable and inspectable with standard tooling.

**Trade-off**

Two stores introduce synchronization and operational complexity.

**Implementation**

`app index` writes child vectors to Qdrant. All other state uses the app SQLite database (`data/app.sqlite`). No cross-store transactions exist; consistency relies on explicit ingestion/indexing workflows.

## 15. Persist intermediate traces

**Decision**

The application writes structured trace events for turn lifecycle, memory reads/writes, routing, retrieval, context assembly, evidence, contradiction handling, tools, citation validation, episode/checkpoint persistence, and evaluation.

**Why**

Agent failures are difficult to understand from only the final answer. Intermediate decisions need to be inspectable to debug routing, retrieval, evidence, and citation behavior.

**Trade-off**

Local trace storage grows over time. The project currently has CLI inspection rather than a rich trace UI.

**Implementation**

`TraceWriter` writes structured events and specialized trace records to SQLite. `app trace list` and `app trace show` provide CLI access. Events include compact structured data for each decision point in the graph.