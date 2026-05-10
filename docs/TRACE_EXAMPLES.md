# Trace Examples

## Retrieve Flow
- `turn.started`
- `memory.checkpoint_read`
- `memory.checkpoint_enabled`
- `memory.semantic_read`
- `memory.episodic_read`
- `router.started`
- `router.completed`
- `retrieval.started`
- `retrieval.children_found`
- `retrieval.parents_scored`
- `context.assembled`
- `evidence.checked`
- `contradiction.detected` (when applicable)
- `contradiction.checked` (when applicable)
- `contradiction.handled` (when applicable)
- `answer.generated`
- `citation.deterministic_validated`
- `citation.llm_validated` (or `skipped`)
- `citation.validated`
- `memory.write_decision`
- `conversation.updated`
- `episode.written`
- `checkpoint.persisted`
- `turn.completed`

## Tool Flow
- `turn.started`
- `memory.checkpoint_read`
- `memory.checkpoint_enabled`
- `memory.semantic_read`
- `memory.episodic_read`
- `router.started`
- `router.completed` (`TOOL`)
- `tool.started`
- `tool.completed`
- `citation.deterministic_validated`
- `citation.llm_validated` (or `skipped`)
- `citation.validated`
- `memory.write_decision`
- `conversation.updated`
- `episode.written`
- `checkpoint.persisted`
- `turn.completed`

## Eval Flow
- `eval.case_started`
- `eval.case_scored`
- `eval.completed`

## Debug Summary Fields (`app ask --debug`)
- Trace ID
- Thread ID
- Turn ID
- Memory read counts: conversation / semantic / episodic
- Checkpoint enabled + message count + active focus
- Route + routing mode + route confidence + route reason
- Retrieval hit count
- Selected parent IDs + parent scores
- Evidence status + evidence confidence
- Conflict label + contradiction handler action
- Context packet IDs (`S#`)
- Final action
- Latencies: total / retrieval / llm / tool
