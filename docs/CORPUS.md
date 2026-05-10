# Corpus

## Purpose
This repository uses a fixed, local corpus for reproducible ingestion, retrieval, and evaluation behavior.

The canonical corpus is:
- `corpus_manifest.json`
- PDFs in `data/raw_pdfs/`

## Corpus Definition
- Size: **100 papers**
- Source family: recent arXiv papers in assignment scope (`cs.AI`-anchored)
- Corpus lock date: **2026-05-10**
- File integrity: each paper has pinned `pdf_sha256`

## Why This Corpus
1. Reproducibility
- Exact arXiv IDs and versions are pinned.
- SHA-256 checksums ensure byte-level consistency of PDFs.

2. Local-first execution
- Runtime ingestion depends on local files, not live discovery.
- The locked manifest is deterministic across runs.

3. Assignment fit
- The set is focused on agentic/LLM retrieval-memory-planning-evaluation themes required by the project’s QA and eval tasks.

4. Parse-ready for this stack
- The selected set is validated against Docling + HybridChunker behavior used in this repository.

## Types Of Papers In This Corpus
### Topic families
- Memory-centric LLM/agent systems
- Retrieval and RAG system methods
- Agent tool-use and workflow/orchestration
- Planning and reasoning methods for agents
- Evaluation/benchmarking and faithfulness-oriented papers

### Category profile
The corpus keeps a `cs.AI` majority while allowing adjacent categories when relevant.

Primary-category distribution in the current lock:
- `cs.AI`: 68
- `cs.LG`: 10
- `cs.CL`: 8
- `cs.CR`: 3
- `stat.ML`: 3
- `quant-ph`: 1
- `cs.SE`: 1
- `cs.RO`: 1
- `cs.MA`: 1
- `cs.CY`: 1
- `cs.NI`: 1
- `eess.AS`: 1
- `astro-ph.IM`: 1

## Runtime Contract
- `app ingest` reads `corpus_manifest.json` and ingests local PDFs from `data/raw_pdfs/`.
- If files are missing/corrupt, run:

```bash
uv run python scripts/download_corpus_pdfs.py
```

The downloader fetches only missing/corrupt files and verifies SHA against the manifest.