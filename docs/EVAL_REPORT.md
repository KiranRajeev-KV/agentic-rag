# Eval Report

This report reflects the latest executed eval runs:

```bash
uv run app eval run --variant child_only
uv run app eval run --variant parent_child
```

Raw summaries (these are the source of truth for all numbers below):
- `runs/reports/eval_summary_child_only.json` (eval_run_id=`eval_ff7bce7adf`)
- `runs/reports/eval_summary_parent_child.json` (eval_run_id=`eval_4288cb3545`)

## Summary

| Variant | Cases | Normalized | Raw | Route Acc | Final Action Acc | Hard-Fail Refusal |
|---|---:|---:|---:|---:|---:|---|
| `child_only` | 16 | 95.62 | 153.00 | 0.9375 | 0.9375 | false |
| `parent_child` | 16 | 96.25 | 154.00 | 0.9375 | 0.9375 | false |

## Retrieval Metrics

| Variant | paper_hit@k | parent_hit@k | parent_mrr | context_token_count |
|---|---:|---:|---:|---:|
| `child_only` | 1.0 | 1.0 | 0.7500 | 722.75 |
| `parent_child` | 1.0 | 1.0 | 0.7500 | 959.50 |

## Child-Only vs Parent-Child Notes

- Both variants were run against the same 16 corpus-grounded eval cases in `src/agentic_rag/evals/cases.yaml`.
- `parent_child` scored higher overall in the latest run (+0.62 normalized; see `docs/ABLATION_REPORT.md`).
- `parent_child` used more context on average (+236.75 tokens per case in `context_token_count`).
