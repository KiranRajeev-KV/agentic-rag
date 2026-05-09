# Ablation Report

Generate comparison with:

```bash
uv run app eval compare --baseline child_only --candidate parent_child
```

The command writes:
- `runs/reports/ablation_report.md`

This report should include:
- baseline vs candidate normalized score delta,
- retrieval metrics delta (`paper_hit@k`, `parent_hit@k`, `parent_mrr`, `context_token_count`),
- at least 2-3 failure analyses from eval traces.
