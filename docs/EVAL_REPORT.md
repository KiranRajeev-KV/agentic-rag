# Eval Report

This report is generated after running:

```bash
uv run app eval run --variant child_only
uv run app eval run --variant parent_child
```

Current implementation writes JSON summaries to:
- `runs/reports/eval_summary_child_only.json`
- `runs/reports/eval_summary_parent_child.json`

Populate this document with:
- normalized scores,
- route/final-action accuracy,
- hard-fail refusal check status,
- retrieval metrics (`paper_hit@k`, `parent_hit@k`, `parent_mrr`, `context_token_count`).
