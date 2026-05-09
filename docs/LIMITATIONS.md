# Limitations

- OpenAI API usage requires an API key and introduces network dependency plus usage cost.
- Docling parsing quality may vary for unusual PDF layouts.
- Corpus is intentionally limited to filtered recent `cs.AI` papers.
- No general web search is used for paper-content answers.
- Evidence thresholds and contradiction handling are calibrated on a small curated eval set (14 cases), not a large benchmark.
- Contradiction handling surfaces conflicts but does not adjudicate scientific truth.
- No reranking, local embedding model, or hybrid sparse retrieval in v1.
