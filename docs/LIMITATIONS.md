# Limitations

- BGE-M3 local inference may be slow on CPU.
- Docling parsing quality may vary for unusual PDF layouts.
- Corpus is intentionally limited to filtered recent `cs.AI` papers.
- No general web search is used for paper-content answers.
- Evidence thresholds are calibrated on a small curated eval set (14 cases).
- Contradiction handling surfaces conflicts but does not adjudicate scientific truth.
- No reranking or hybrid sparse retrieval in v1.
