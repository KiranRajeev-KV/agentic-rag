# Testing Strategy

The test suite focuses primarily on deterministic behavior around retrieval, evidence, routing, persistence, indexing, tools, and control-plane policies. External model, vector-database, and arXiv interactions are usually replaced with fakes or mocked clients so the tests can exercise project logic without depending on live services.

## Running the checks

```bash
uv run pytest
uv run ruff check .
```

These commands are the intended local checks. They were not rerun as part of this documentation update, so this document does not claim a current passing test count.

## Coverage map

| Area | Representative tests | What they verify |
|---|---|---|
| Retrieval and evidence | `test_retrieval_logic.py` | parent scoring, evidence gating, context construction, query-shaped context budgets, retrieval variants (`child_only`, `parent_child`) |
| Agent policies | `test_agent_components.py`, `test_agent_nodes_policy.py` | routing/refusal/tool behavior, citation invariants, fail-closed citation validation, memory-write policy, contradiction routing |
| Memory and checkpoints | `test_memory_layers.py`, `test_graph_checkpointing.py` | episodic persistence, conversation-context use, checkpoint state loading, thread-id propagation |
| Ingestion | `test_ingest_pipeline.py` | manifest workflow, missing PDFs, unchanged-parse skipping, force behavior, partial progress |
| Indexing | `test_indexing.py`, `test_embeddings.py`, `test_qdrant_config.py` | pending selection, idempotency, force reindexing, embedding request shape/config, vector dimensions |
| LLM client | `test_llm_client.py` | structured parse request wiring and missing-structured-output failure |
| arXiv tools | `test_arxiv_tools.py` | cache behavior, schema bounds, structured error handling |
| CLI | `test_cli.py` | command wiring, argument propagation, summaries, debug output |
| Evaluation plumbing | `test_evals.py` | variant execution/comparison and evaluator mechanics |
| Convenience commands | `test_justfile.py` | expected recipe presence |

Do not add a coverage percentage.

## Testing style

- Temporary SQLite databases and temporary directories isolate persistence-heavy tests.
- Fake retrievers/repositories make retrieval heuristics deterministic.
- Mocked OpenAI clients verify request/schema wiring without live model calls.
- Fake Qdrant clients verify indexing/upsert behavior without requiring a live vector database.
- Fake arXiv clients verify caching/error semantics without live API dependency.
- CLI tests use Typer `CliRunner`.
- Policy tests target explicit failure/refusal/citation behavior rather than only successful answers.

## What the suite is particularly useful for

- preventing accidental changes to parent scoring and evidence-gate behavior
- preserving citation-safety invariants
- verifying indexing idempotency
- verifying thread/checkpoint configuration
- checking tool-routing and error-handling behavior
- checking that CLI changes still propagate expected arguments/state
- regression testing evaluator plumbing

## What the current suite does not establish

- It does not prove retrieval relevance quality over a labelled benchmark.
- It does not validate production-scale Qdrant behavior or concurrency.
- It does not continuously test live OpenAI model behavior.
- It does not continuously test live arXiv availability.
- It does not provide browser/UI testing because the project is CLI-first.
- It does not provide load, soak, latency-SLO, or cost testing.
- It does not currently provide CI-backed verification on every commit.
- It does not prove long-horizon memory/checkpoint correctness across extended conversations.
- It does not provide a code-coverage percentage in this repository.

## Recently cleaned maintenance items

Two stale test/developer-workflow inconsistencies identified during repository review were cleaned up:

- The mocked CLI evaluator fixture now uses the current 16-case evaluation-set size.
- The retired live-discovery `just discover` recipe and its test expectation were removed after the project moved to the fixed manifest workflow.

For evaluation-specific limitations, see `EVAL_REPORT.md`. For runtime configuration and reproducibility, see `CONFIGURATION.md`.
