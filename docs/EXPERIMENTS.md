# Engineering Evolution and Experiments

This repository was not developed from a single fixed design. It began as a short RAG/agent build and then became a place to try different retrieval, model-control, memory, grounding, and reproducibility approaches. The history below is reconstructed from Git commits and the resulting implementation; it should be read as engineering evolution rather than as a sequence of controlled scientific experiments.

> Not every change below was an A/B experiment. Where the repository contains measured comparison data, it is identified explicitly. Otherwise, the sections document approaches that were implemented, replaced, retained, or refined.

## Timeline

| Phase | Direction |
| --- | --- |
| 1. Local retrieval foundation | SQLite + Docling + Qdrant + local BGE-M3 dense embeddings |
| 2. Explicit agent graph | LangGraph routing, retrieval variants, evidence gating, tracing, evaluation |
| 3. Model-driven control | OpenAI embeddings, structured LLM routing/evidence/answers, stricter citation rules |
| 4. Memory and grounding | Structured outputs, conversation/semantic/episodic state, semantic citation gate |
| 5. Checkpoint migration | Replace custom conversation persistence with LangGraph SQLite checkpoints; add contradiction handling |
| 6. Reproducible corpus | Replace live discovery with fixed manifest + hash-verified local PDFs |
| 7. Retrieval hardening | Intent normalization, conditional lexical fallback, evidence recovery, recorded retrieval ablation |

## 1. Local retrieval foundation

**What was tried**

The initial implementation established the local-first storage and retrieval foundation:

- SQLite for relational document state
- Docling for PDF parsing
- Parent/child chunk representation
- Qdrant for child vector storage
- Explicit index lifecycle (ingest then index) with idempotent skipping and force reindex
- Local BGE-M3 dense embedding inference (1024 dimensions) with CPU/CUDA device selection

**What changed**

This phase established the core infrastructure. BGE-M3 sparse vectors and ColBERT vectors were explicitly not used. The separation of ingestion from indexing meant embedding work remained explicit and controllable.

**Current state**

The SQLite/Qdrant split and parent/child chunking remain. The embedding provider was later migrated to OpenAI (see Section 3). The explicit ingest/index workflow remains.

**What remains uncertain**

The original BGE-M3 approach was never benchmarked against the current OpenAI embedding in this repository. The local-first inference path could still be viable for different cost/latency/privacy trade-offs.

## 2. Child-only vs parent-child retrieval

**What was tried**

Both retrieval variants were introduced together in the first substantial agent commit:

- `child_only`: each retrieved child chunk forms its own group; exists as a comparison baseline; can select the same parent multiple times, wasting context budget
- `parent_child`: groups child hits by parent, scores parent groups using max child score + bounded support bonus + section adjustment - references penalty, expands to broader parent context for generation

Both variants shared:
- Dynamic parent/context budgeting (default 3, comparison 5, synthesis 6, metadata 0)
- Context packets with stable source IDs (S1, S2, ...)
- Retrieval evidence signals (top scores, margins, support counts, distinct papers, section distribution)
- Evidence states: SUFFICIENT, AMBIGUOUS, INSUFFICIENT

**What changed**

The evaluation runner was added immediately so the two variants could be compared through common behavioral cases.

**Current state**

Both variants remain implemented. `child_only` is retained as the comparison baseline. The recorded ablation uses `parent_child` as the default candidate.

**What remains uncertain**

The repository does not contain evidence that either variant is universally superior. The recorded ablation showed identical aggregate retrieval metrics and a score difference driven by tool/citation behavior, not retrieval quality.

## 3. Local BGE-M3 to OpenAI embeddings

| Stage | Embedding approach | Vector size | Status |
| --- | --- | ---: | --- |
| Early | BGE-M3 dense-only local inference | 1024 | Replaced |
| Current | OpenAI `text-embedding-3-small` | 1536 | Active |

There is no controlled retrieval-quality benchmark in this repository comparing these embedding models, so this migration should not be interpreted as evidence that one model was empirically superior here.

The Qdrant collection configuration was updated for the new dimension, and explicit vector-dimension mismatch checking was added.

## 4. Deterministic control to structured LLM control

**What was tried**

Early agent logic contained predominantly deterministic/heuristic routing and control behavior.

**What changed**

Schema-bound OpenAI LLM operations were introduced for:
- Routing (with Pydantic output schema)
- Evidence decisions
- Answer generation
- Semantic memory-write decisions

Fallback behavior was retained when model-backed control was unavailable or failed. Tools were expanded to distinguish arXiv search from arXiv ID lookup. Citation handling became stricter (allowed source IDs only, rejection of internal ID leaks). Semantic memory writes were made more selective. Ingestion became more idempotent with unchanged-paper skipping and force mode.

**Current state**

The system uses a mixed control model:
- Deterministic rules where behavior is easy to specify (high-confidence arXiv tool intents)
- Structured LLM decisions where semantic interpretation is required
- Safe fallback/refusal paths on failure

**What remains uncertain**

The boundary between deterministic and LLM-driven control is a design choice, not an experimentally optimized threshold.

## 5. Citation validation in two layers

**What was tried**

The citation system evolved through three stages:

1. Early: deterministic mechanical ID validation
2. Stricter deterministic invariants: allowed source/tool IDs only, internal chunk/parent/paper ID rejection, source-block presence requirements
3. Optional LLM-assisted semantic citation validation as a second gate

**What changed**

The semantic layer runs after deterministic checks pass. It can catch unsupported claims that mechanical validation cannot detect.

**Current state**

Two-layer validation is active by default. The deterministic gate is a hard requirement; the semantic gate is optional but enabled in the recorded runs.

**What remains uncertain**

The recorded `e15_tool_search` trace shows that the semantic validator can fail closed after a successful tool call, suppressing an otherwise valid answer. This nondeterministic availability downside is a known trade-off of the fail-closed design.

For the metric caveats and trace-backed failure analyses, see [`ABLATION_REPORT.md`](ABLATION_REPORT.md).

## 6. Memory: custom conversation state to LangGraph checkpoints

**What was tried**

Initially the project maintained explicit custom conversation repositories and tables for:
- Threads
- Turns
- Summaries/focus state

**What changed**

Conversation state was migrated to LangGraph SQLite checkpointing keyed by `thread_id`. The old manually maintained conversation repositories/tables were removed.

Semantic memory (selected reusable decisions) and episodic records (completed turn histories) remained separate persistence concerns.

Terminal CLARIFY and REFUSE paths were changed to also pass through memory-update/checkpoint persistence.

**Current state**

Conversation memory uses LangGraph SQLite checkpoints. Semantic memory and episodic records use explicit SQLite tables.

**What remains uncertain**

This was an architectural replacement, not a measured performance comparison. The framework-native approach simplifies the codebase but couples conversation continuity to the graph execution framework.

## 7. Live discovery to a fixed corpus manifest

**What was tried**

Before this change, the project had a live corpus-discovery workflow:
- arXiv recent-paper discovery
- Date windows
- Query/category filtering
- Discovery/filter metadata

**What changed**

The runtime corpus became:
- Fixed `corpus_manifest.json`
- Fixed paper/version metadata
- Locally downloaded PDFs
- SHA-256 verification
- Explicit download step
- Explicit ingest step

The old discovery/filter modules were then removed. Section-heading extraction was corrected and indexing diagnostics/empty-chunk handling were improved as related engineering refinements.

**Current state**

Fixed manifest with hash-verified local PDFs. The download script checks existing local PDFs against expected SHA-256, skips correct files, downloads missing/invalid files, and verifies after download.

**What remains uncertain**

The fixed corpus makes evaluation runs easier to reproduce, but does not permanently guarantee reproducibility—remote PDFs may become unavailable, parsing libraries can change output, and model APIs evolve independently.

## 8. Retrieval fallback and recovery

Two distinct mechanisms were added at different times. They should not be merged conceptually.

### A. Conditional lexical fallback

**What was tried**

Primary retrieval is Qdrant child-vector similarity search. When fewer than 8 child hits are returned:

- Query terms are extracted
- SQLite child text is searched using simple LIKE matching
- Unseen lexical matches are appended with heuristic scores

**What changed**

This fallback was added after the initial vector-only first-pass retrieval.

**Current state**

Active as a conditional fallback triggered by sparse vector results.

**What remains uncertain**

This is not BM25, sparse-vector retrieval, Reciprocal Rank Fusion, learned reranking, or full dense+sparse hybrid retrieval. It is a heuristic fallback merge.

### B. Evidence-triggered retrieval recovery

**What was tried**

After initial retrieval/context assembly, if evidence is AMBIGUOUS or INSUFFICIENT:

- One broader vector retrieval is attempted
- Retrieval expands to at least 60 child candidates
- Reference sections are permitted
- Groups/context/evidence are rebuilt
- Evidence signals are recalculated

**What changed**

This recovery was added after the single-pass retrieval design.

**Current state**

Active inside `RetrievalService`. It is not an autonomous planner and not another LangGraph agent node.

**What remains uncertain**

Only one explicit recovery expansion is implemented. Evidence thresholds remain hand-set heuristics not validated on a large benchmark.

## 9. What the recorded ablation actually showed

| Metric | child_only | parent_child |
| --- | ---: | ---: |
| Normalized score | 95.62 | 96.25 |
| Route accuracy | 0.9375 | 0.9375 |
| Final-action accuracy | 0.9375 | 0.9375 |
| parent MRR | 0.7500 | 0.7500 |
| context size approximation | 722.75 | 959.50 |

- +0.62 was caused by `e15_tool_search` downstream tool/citation behavior
- Not by retrieval-quality difference
- Relevance labels are absent in the current cases
- Current evidence is insufficient to declare a retrieval winner
- `parent_child` used more assembled context
- `child_only` exposed duplicate-parent budget waste in one recorded trace

For the metric caveats and trace-backed failure analyses, see [`EVAL_REPORT.md`](EVAL_REPORT.md) and [`ABLATION_REPORT.md`](ABLATION_REPORT.md).

## 10. Current system after these iterations

| Concern | Current approach | Historical alternative tried |
| --- | --- | --- |
| Embedding | OpenAI `text-embedding-3-small` | Local BGE-M3 dense |
| Corpus selection | Fixed manifest | Live arXiv discovery/filtering |
| Conversation state | LangGraph SQLite checkpoints | Custom conversation persistence |
| Retrieval structure | `parent_child` default + `child_only` baseline | Both variants introduced together and retained |
| Routing | Deterministic tool intents + structured LLM/fallback routing | Primarily heuristic/deterministic routing |
| Citation checking | Deterministic + optional semantic LLM validation | Deterministic-only validation |
| Sparse recovery | Conditional SQLite lexical fallback | Vector-only first-pass retrieval |
| Evidence recovery | One broader evidence-triggered retrieval pass | Single retrieval pass |

The repository therefore reflects several replaced approaches rather than a single architecture that appeared fully formed.