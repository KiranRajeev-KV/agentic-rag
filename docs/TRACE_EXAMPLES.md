# Trace Examples

## Typical retrieve flow
- `turn.started`
- `memory.read`
- `router.completed`
- `retrieval.parents_scored`
- `evidence.checked`
- `citation.validated`
- `memory.write`
- `turn.completed`

## Typical tool flow
- `turn.started`
- `memory.read`
- `router.completed` (`TOOL`)
- `tool.called`
- `citation.validated`
- `turn.completed`
