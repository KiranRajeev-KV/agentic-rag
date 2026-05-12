# Trace Reference

The trace system records intermediate agent decisions into local SQLite tables. This document describes the current event model and the common flow shapes visible through the CLI.

## Trace identity

- `trace_id` — unique per invocation (format `tr_<id>`)
- `thread_id` — conversation boundary for checkpoint persistence
- `turn_id` — unique per graph invocation (format `turn_<id>`)
- `run_mode` — execution mode (e.g., `cli`, `eval`)
- `started_at` / `completed_at` — lifecycle timestamps

A thread can contain multiple turns, while each invocation gets its own trace/turn IDs.

## Event format

Each generic event contains:

- `timestamp`
- `level`
- `event name`
- optional graph node
- JSON payload

Events are displayed in insertion order by `app trace show`.

## Common event families

| Family | Events |
| --- | --- |
| Lifecycle | `turn.started`, `turn.failed`, `turn.completed` |
| Memory/state | `memory.checkpoint_read`, `memory.checkpoint_enabled`, `memory.semantic_read`, `memory.episodic_read`, `memory.write_decision`, `memory.write`, `conversation.updated`, `episode.written`, `checkpoint.persisted` |
| Routing | `router.started`, `router.completed` |
| Retrieval/context | `retrieval.started`, `retrieval.failed`, `retrieval.children_found`, `retrieval.parents_scored`, `context.assembled` |
| Evidence | `evidence.checked` |
| Contradictions | `contradiction.detected`, `contradiction.checked`, `contradiction.tool_lookup_requested`, `contradiction.handled` |
| Tools | `tool.started`, `tool.called`, `tool.completed` |
| Generation/citations | `answer.generated`, `citation.deterministic_validated`, `citation.llm_validated`, `citation.validated` |
| LLM | `llm.call` |
| Evaluation | `eval.case_started`, `eval.case_scored`, `eval.completed` |

## Typical corpus-answer flow

```text
turn.started
memory.checkpoint_read
memory.checkpoint_enabled
memory.semantic_read
memory.episodic_read
router.started
router.completed
retrieval.started
retrieval.children_found
retrieval.parents_scored
context.assembled
evidence.checked
answer.generated
citation.deterministic_validated
citation.llm_validated
citation.validated
memory.write_decision
[optional memory.write]
conversation.updated
episode.written
checkpoint.persisted
turn.completed
```

This is representative, not guaranteed. Evidence conflict, retrieval failure, clarification, refusal, fallback routing, and citation failure change the event sequence.

## Typical tool flow

```text
turn.started
memory.checkpoint_read
memory.checkpoint_enabled
memory.semantic_read
memory.episodic_read
router.started
router.completed
tool.started
tool.called
tool.completed
citation.deterministic_validated
citation.llm_validated
citation.validated
memory.write_decision
[optional memory.write]
conversation.updated
episode.written
checkpoint.persisted
turn.completed
```

Again this is representative.

## Contradiction path

Contradiction handling may emit:

```text
contradiction.detected
contradiction.checked
contradiction.tool_lookup_requested   # only when metadata lookup is selected
contradiction.handled
```

Do not imply all four always occur.

## Failure visibility

Current failure surfaces:

- `retrieval.failed` captures retrieval exceptions before the graph degrades toward refusal.
- failed `llm.call` events can contain model/schema, latency and error.
- failed citation validation records deterministic or LLM validation outcome before fail-closed refusal.
- `turn.failed` captures otherwise unhandled graph invocation exceptions.

Do NOT call these retries. They are failure capture plus safe degradation.

## Structured trace tables

| Table | Purpose | Current CLI exposure |
| --- | --- | --- |
| `traces` | Run identity and lifecycle | `trace list`, `trace show` |
| `trace_events` | Ordered generic event stream | `trace show` |
| `retrieval_traces` | Structured retrieval/evidence signals | Not joined by current `trace show` |
| `tool_traces` | Structured tool execution record | Not joined by current `trace show` |
| `evidence_traces` | Structured evidence outcome | Not joined by current `trace show` |
| `answer_traces` | Successful validated-answer record | Not joined by current `trace show` |

The `answer_traces` table is not a complete log of every final action; clarification/refusal paths are represented in the generic event/episode/state flow but are not comprehensively persisted through `answer_traces`.

## Debug summary

The existing `app ask --debug` fields grouped by concern:

Identity:

- trace/thread/turn

Memory:

- checkpoint status
- message/summary information
- conversation/semantic/episodic read counts
- active focus

Routing:

- action
- mode
- confidence
- public reason

Retrieval/evidence:

- retrieved child count
- selected parents
- parent scores
- evidence status/confidence
- conflict label
- contradiction action
- context packet IDs

Outcome:

- final action

Latency:

- total
- retrieval
- accumulated LLM
- tool

## Known tracing limitations

1. Local SQLite tracing only; no distributed tracing.
2. The current CLI reader does not join specialized trace tables.
3. Structured tool traces currently cover the supported arXiv search/lookup executions; unsupported tool routes do not create a dedicated `tool_traces` row.
4. `answer_traces` does not comprehensively represent clarification/refusal outcomes.
5. No token/cost accounting in traces.
6. No dashboard, metrics backend, sampling, alerting, or retention management.
7. Trace/trajectory quality is not yet evaluated.