# Configuration and Reproducibility

This repository is designed around a local, inspectable execution environment: SQLite stores application state, Qdrant stores child vectors, and a fixed corpus manifest defines the document set. External model calls are still required by the current embedding and LLM implementations, so local-first should not be read as fully offline.

## Requirements

|Requirement|Purpose|
|---|---|
|Python 3.11+|Runtime|
|`uv`|Dependency installation from the committed lockfile|
|Docker + Docker Compose|Local Qdrant|
|Network access|Corpus download and configured API calls|
|OpenAI API key|Indexing/query embeddings and configured LLM operations|

The committed `uv.lock` is the dependency lock used by `uv sync`. This improves repeatability but does not guarantee future compatibility with every operating system, external API, or model endpoint.

## Clean setup

```bash
cp .env.example .env
uv sync
docker compose up -d qdrant
uv run app db init
uv run python scripts/download_corpus_pdfs.py
uv run app ingest --limit 100
uv run app index
uv run app ask "What do recent papers say about agent memory?" --thread-id demo --debug
```

Set `OPENAI_API_KEY` in `.env` before running `app index`. The current query retrieval path also embeds the query through the configured OpenAI embedding implementation.

## Corpus reproducibility

The corpus manifest contains fixed paper metadata, version information, and SHA-256 checksums for PDFs. The download script `scripts/download_corpus_pdfs.py` reads the manifest, checks existing local PDFs against their expected SHA-256, skips files with correct checksums, downloads missing or invalid files, and verifies the SHA after download. A SHA mismatch is reported as failure. This manifest-plus-hash approach makes the corpus substantially more reproducible than rediscovering papers live on every run.

However, remote files and services can still change: PDF URLs may break, Docling parsing output can shift across library versions, and model/API behavior is an external dependency. The manifest guarantees a fixed corpus only as long as the underlying artifacts remain accessible and unchanged.

## Configuration reference

|Variable|Default|Purpose|
|---|---|---|
|APP_ENV|local|application environment label|
|APP_DB_PATH|./data/app.sqlite|SQLite application database|
|APP_RUNS_DIR|./runs|local run-artifact directory|
|APP_PDF_DIR|./data/raw_pdfs|local corpus PDFs|
|APP_CORPUS_MANIFEST|./corpus_manifest.json|fixed corpus manifest|
|APP_LOG_JSONL|./runs/logs/app.jsonl|JSONL application/ingestion log path|
|QDRANT_URL|http://localhost:6333|Qdrant HTTP endpoint|
|QDRANT_COLLECTION|arxiv_child_chunks|Qdrant child-vector collection|
|OPENAI_API_KEY|empty|credential for current OpenAI embeddings and LLM calls|
|EMBEDDING_PROVIDER|openai|embedding provider selector|
|EMBEDDING_MODEL|text-embedding-3-small|embedding model used for child chunks and query embeddings|
|EMBEDDING_DIMENSIONS|1536|embedding vector dimension; must remain consistent with Qdrant/indexed vectors|
|LLM_PROVIDER|openai|LLM provider selector|
|ANSWER_MODEL|gpt-5-nano|answer-generation structured LLM model|
|ROUTER_MODEL|gpt-5-nano|routing/intent structured LLM model|
|EVIDENCE_MODEL|gpt-5-nano|evidence/contradiction/citation-related structured LLM operations|
|ARXIV_USER_AGENT|agentic-rag/0.1|user-agent sent to arXiv requests|

For provider rows (`EMBEDDING_PROVIDER`, `LLM_PROVIDER`) the current implementation supports OpenAI; do not imply arbitrary providers work.

## Data layout

|Path / service|Role|
|---|---|
|`corpus_manifest.json`|Fixed corpus definition|
|`data/raw_pdfs/`|Downloaded source PDFs|
|`data/app.sqlite`|Application relational state|
|`data/qdrant/`|Local Qdrant persistence|
|`runs/`|Run/log/report artifacts|
|Qdrant `arxiv_child_chunks`|Child embedding vectors|

Do not claim every generated run artifact is committed.

## External dependencies by operation

|Operation|Qdrant|OpenAI|arXiv/network|
|---|---|---|---|
|`db init`|No|No|No|
|`download_corpus_pdfs.py`|No|No|Yes|
|`app ingest`|No|No|No if PDFs are already present|
|`app index`|Yes|Yes|No|
|`app ask`|Yes for corpus retrieval|Yes for the current query-embedding path; configured LLM calls also use it|Only when the selected route invokes the arXiv tool|
|`app trace list/show`|No|No|No|

## Resetting local state

Normal startup should not require reset. SQLite reset and Qdrant reset are separate operations. The `uv run app db reset --yes` command resets the SQLite application DB; it does NOT remove Qdrant local storage. A complete local data reset requires the following destructive sequence:

```bash
uv run app db reset --yes
docker compose down
rm -rf data/qdrant
docker compose up -d qdrant
```

This is destructive and removes local vector-index persistence.

## Reproducibility boundaries

- The corpus is fixed locally, but remote PDFs may become unavailable in the future.
- Parsing output can change across Docling/library versions.
- Model/API behavior can change independently of this repository.
- OpenAI model names/endpoints are external dependencies.
- `uv.lock` pins Python package resolution, not Docker images beyond what Compose explicitly pins or remote model behavior.
- Reusing an old SQLite/Qdrant state can affect experiments; evaluation checkpoint isolation has additional limitations documented in the evaluation report.
- No automated CI reproducibility check is currently provided.

See `CORPUS.md` for corpus details and `DEMO_SCRIPT.md` for the existing manual demo flow.