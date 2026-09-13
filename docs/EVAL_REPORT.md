# Eval Report

> This report documents an existing recorded run.

## Scope

- 16 behavior-oriented cases.
- Categories represented: content QA, synthesis/comparison, follow-ups, ambiguity, refusal, and arXiv tools.
- Not a large benchmark.
- Current cases lack expected paper/parent relevance labels.

## Recorded Results

| Metric | child_only | parent_child |
| --- | ---: | ---: |
| Cases | 16 | 16 |
| Normalized score | 95.62 | 96.25 |
| Raw score | 153.00 | 154.00 |
| Route accuracy | 0.9375 | 0.9375 |
| Final-action accuracy | 0.9375 | 0.9375 |
| Hard-fail refusal | false | false |
| paper_hit@k | 1.0 | 1.0 |
| parent_hit@k | 1.0 | 1.0 |
| parent_mrr | 0.7500 | 0.7500 |
| context_token_count | 722.75 | 959.50 |

## Metric Interpretation

### Behavioral score

The overall score is a custom 10-point-per-case engineering rubric composed of:
- route correctness (2 points)
- final action correctness (2 points)
- behavioral/evidence outcome (2 points)
- citation behavior (2 points)
- retrieval/tool-related checks (remaining points)

**Important**: `memory_score` is currently hardcoded to `0.0` and `trace_score` is currently hardcoded to `0.0`. Therefore the rubric does not currently evaluate those dimensions despite the fields existing.

### Retrieval metrics

The evaluator reports `paper_hit@k`, `parent_hit@k`, and `parent_mrr`. However, the current YAML cases do not specify expected paper IDs or expected parent IDs (the model fields exist but their defaults are empty lists).

- `paper_hit@k` returns 1.0 automatically when no expected paper IDs are supplied.
- `parent_hit@k` returns 1.0 automatically when no expected parent IDs are supplied.
- For `parent_mrr`: when expected parent IDs exist, reciprocal rank is calculated normally; when expected parent IDs are absent, the implementation returns 1.0 if any parent was selected and 0 otherwise.

Therefore the current hit@k and MRR values MUST NOT be described as strong relevance-quality measurements for this eval set. They are instrumentation produced by an evaluator that is ready to consume relevance labels, but the current cases do not contain those labels.

### Context size

`context_token_count` is implemented by splitting context strings on whitespace and counting resulting words. It is NOT calculated using an LLM tokenizer. It is currently a whitespace-based context-size approximation.

## Interpretation

- `parent_child` normalized score was 0.62 higher.
- Retrieval metric values (paper_hit@k, parent_hit@k, parent_mrr) were unchanged.
- Context approximation increased by 236.75.
- The score difference was caused by `e15_tool_search`, where the LLM citation validator behaved differently after the same successful tool route.
- Current evidence does not establish parent-child retrieval superiority.

## Evaluation Isolation Caveat

- Thread ID uses `eval_<variant>_<case_id>`.
- Persistent LangGraph checkpoints are keyed by thread ID.
- Repeated eval runs can potentially inherit prior checkpoint state because the thread ID does not include an `eval_run_id`.
- Unique run-scoped thread IDs would make evaluation isolation stronger.
- The existing `e12` traces make this caveat particularly relevant, but the trace alone is not enough to prove that contamination caused the recorded behavior.

## Next Evaluation Improvements

1. Add expected paper and parent relevance labels.
2. Add Recall@K, Precision@K, MRR and NDCG over labelled cases.
3. Use a tokenizer for context/token metrics.
4. Add groundedness/faithfulness checks separate from citation formatting.
5. Implement memory-quality scoring.
6. Implement trajectory/trace-quality scoring.
7. Isolate checkpoint threads by eval run.
8. Run repeated trials for model-dependent steps.
