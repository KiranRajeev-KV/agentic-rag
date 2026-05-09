# Limitations

- OpenAI API usage requires an API key and introduces network/API cost dependency.
- Docling parsing quality may vary for unusual PDF layouts.
- Corpus is intentionally limited to filtered recent `cs.AI` papers.
- No general web search is used for paper-content answers.
- Evidence gate thresholds are calibrated on 14 curated cases, not a large benchmark.
- Contradiction handling surfaces conflicts but does not adjudicate scientific truth.
- No reranking, local embedding model, or hybrid sparse retrieval in v1.
- Conversation memory is thread-scoped; follow-up quality depends on consistent `--thread-id` use.
- This revision expects local runtime reset (SQLite/Qdrant) rather than backward-compatible DB migration.
