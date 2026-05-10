# Ablation Report (Child-Only vs Parent-Child)

This report reflects the latest executed comparison:

```bash
uv run app eval compare --baseline child_only --candidate parent_child
```

Source summaries (these are the source of truth for all numbers below):
- `runs/reports/eval_summary_child_only.json` (eval_run_id=`eval_ff7bce7adf`)
- `runs/reports/eval_summary_parent_child.json` (eval_run_id=`eval_4288cb3545`)

## Delta Metrics (Candidate - Baseline)

Overall:
- normalized score delta: `96.25 - 95.62 = +0.62`

Retrieval metrics delta:
- `paper_hit@k`: `+0.0` (1.0 -> 1.0)
- `parent_hit@k`: `+0.0` (1.0 -> 1.0)
- `parent_mrr`: `+0.0000` (0.7500 -> 0.7500)
- `context_token_count`: `+236.75` (722.75 -> 959.50)

Interpretation:
- In this run, parent-child’s win is explained by better behavior on a tool-routing case (`e15_tool_search`) while performing similarly elsewhere. Parent scoring quality (`parent_mrr`) is identical in aggregate for this run, but parent-child uses more context on average.

## Trace-Backed Failure Analyses

### 1) Tool Route: LLM Citation Validator Can Override Successful Tool Output (e15)

Case:
- `e15_tool_search` ("Search arXiv for recent papers on agent memory orchestration.")

Observed:
- `child_only` ends `REFUSE` (trace `tr_13dad02e10dd`)
- `parent_child` ends `ANSWER_FROM_TOOL` (trace `tr_5fb2c98684db`)

What happened (from trace events):
- Both runs route deterministically to `TOOL` (`router.completed` reason: "Deterministic arXiv metadata/search intent match.").
- In `child_only`, the tool call succeeds (`tool.completed status=ok papers=5`) but the LLM citation validator returns `verdict=UNSUPPORTED_CLAIM`, and the graph ends in `REFUSE`:
  - `tool.completed` -> `citation.deterministic_validated status=ok` -> `citation.llm_validated status=failed verdict=UNSUPPORTED_CLAIM` -> `turn.completed final_action=REFUSE` (`tr_13dad02e10dd`)
- In `parent_child`, the same tool call succeeds and the citation validator returns `VALID`, preserving `ANSWER_FROM_TOOL` (`tr_5fb2c98684db`).

Why it matters:
- This is a real “fail-closed” behavior: a successful tool result can still produce a refusal if the LLM citation validator is overly strict/flaky.

### 2) Ambiguous Follow-Up: Answer vs Clarify Divergence Despite SUFFICIENT Evidence (e12)

Case:
- `e12_ambiguous_reference` question: "What about it?"

Observed:
- `child_only` ends `CLARIFY` (trace `tr_4a64e975406c`)
- `parent_child` ends `ANSWER_FROM_CONTEXT` (trace `tr_58c495685cc9`)

What happened (from trace events):
- Both variants route to `RETRIEVE` as a follow-up via checkpoint memory (`router.completed` reason: "Follow-up query resolved from thread conversation memory.").
- Both variants have `evidence_status=SUFFICIENT` and `final_evidence_action=ANSWER_FROM_CONTEXT`.
- In `child_only`, the graph ends `CLARIFY` after the answer/evidence steps (`turn.completed final_action=CLARIFY`) (`tr_4a64e975406c`).
- In `parent_child`, the model generates an answer with citations and passes citation validation (`answer.generated` then `citation.validated status=ok`) and ends `ANSWER_FROM_CONTEXT` (`tr_58c495685cc9`).

Why it matters:
- This shows the “underspecified follow-up” edge: even with sufficient retrieved evidence, the system may decide the user intent is unclear (clarify) versus taking a best-effort grounded answer path.

### 3) Child-Only Retrieval Can Waste Parent Budget via Duplicate Parent IDs (e12)

Observed (same case `e12_ambiguous_reference`, child-only):
- `retrieval.parents_scored` includes duplicate parent IDs in `selected_parents`:
  - `["parent_08b6ece59f36b96b", "parent_9242c728b6972841", "parent_08b6ece59f36b96b"]`
  - Trace event: `retrieval.parents_scored` in `tr_4a64e975406c`

Why it matters:
- `child_only` uses per-child grouping without deduping parent IDs before slicing to the parent budget, which can reduce evidence diversity in the assembled context.
