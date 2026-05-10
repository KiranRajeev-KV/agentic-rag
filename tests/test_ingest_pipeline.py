import json
from pathlib import Path
from types import SimpleNamespace

from agentic_rag.config import get_settings
from agentic_rag.ingest.pipeline import IngestPipeline
from agentic_rag.storage.bootstrap import initialize_storage
from agentic_rag.storage.repositories import PaperRecord, PaperRepository
from agentic_rag.storage.sqlite import SQLiteStore


def _write_manifest(path: Path, arxiv_id: str = "2501.20001") -> None:
    payload = {
        "corpus_name": "x",
        "snapshot_date": "2026-05-10",
        "date_from": "2026-02-09",
        "date_to": "2026-05-10",
        "source": "arxiv_api",
        "source_query": "cat:cs.AI",
        "selection_policy_version": "v1",
        "selection_strategy": "strategy_b_stratified",
        "rate_limit_policy": "seq",
        "papers": [
            {
                "arxiv_id": arxiv_id,
                "arxiv_version": "v1",
                "title": "Test paper",
                "authors": ["A"],
                "abstract": "retrieval and memory",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "abs_url": "https://arxiv.org/abs/2501.20001",
                "pdf_url": "https://arxiv.org/pdf/2501.20001.pdf",
                "doi": None,
                "comment": None,
                "pdf_sha256": "",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ingest_manifest_missing_pdf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("APP_CORPUS_MANIFEST", str(tmp_path / "corpus_manifest.json"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)
    _write_manifest(settings.app_corpus_manifest)

    pipeline = IngestPipeline(settings=settings)
    summary = pipeline.run(limit=1)

    assert summary.discovered == 1
    assert summary.download_failed == 1
    assert summary.parsed == 0
    assert any("download_corpus_pdfs.py" in e for e in summary.errors)


def test_ingest_pipeline_skips_unchanged_parsed_paper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("APP_CORPUS_MANIFEST", str(tmp_path / "corpus_manifest.json"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    arxiv_id = "2501.20001"
    _write_manifest(settings.app_corpus_manifest, arxiv_id=arxiv_id)

    pdf_path = settings.app_pdf_dir / f"{arxiv_id}v1.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = b"pdf-content"
    pdf_path.write_bytes(pdf_bytes)

    import hashlib

    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
    payload = json.loads(settings.app_corpus_manifest.read_text(encoding="utf-8"))
    payload["papers"][0]["pdf_sha256"] = pdf_sha
    settings.app_corpus_manifest.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("agentic_rag.ingest.pipeline._parser_version", lambda: "1.2.3")
    store = SQLiteStore(settings.app_db_path)
    paper_repo = PaperRepository(store)
    paper_repo.upsert(
        PaperRecord(
            paper_id="paper_2501.20001",
            arxiv_id=arxiv_id,
            arxiv_version="v1",
            title="Test paper",
            authors=["A"],
            abstract="retrieval and memory",
            primary_category="cs.AI",
            categories=["cs.AI"],
            source_query="corpus_manifest",
            pdf_sha256=pdf_sha,
            parse_status="parsed",
            parser_name="docling",
            parser_version="1.2.3",
            chunker_name="docling_hybrid_chunker",
            chunker_config_hash="default",
        )
    )

    def _unexpected_parse(pdf_path: Path):  # noqa: ARG001
        raise AssertionError("parse_pdf should not be called for unchanged parsed paper")

    pipeline = IngestPipeline(settings=settings)
    pipeline.parser = SimpleNamespace(parser_name="docling", parse_pdf=_unexpected_parse)
    summary = pipeline.run(limit=1)
    assert summary.discovered == 1
    assert summary.skipped == 1
    assert summary.downloaded == 0
    assert summary.parsed == 0


def test_ingest_pipeline_force_bypasses_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("APP_CORPUS_MANIFEST", str(tmp_path / "corpus_manifest.json"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    arxiv_id = "2501.30001"
    _write_manifest(settings.app_corpus_manifest, arxiv_id=arxiv_id)

    pdf_path = settings.app_pdf_dir / f"{arxiv_id}v1.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = b"cached-pdf"
    pdf_path.write_bytes(pdf_bytes)

    import hashlib

    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
    payload = json.loads(settings.app_corpus_manifest.read_text(encoding="utf-8"))
    payload["papers"][0]["pdf_sha256"] = pdf_sha
    settings.app_corpus_manifest.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("agentic_rag.ingest.pipeline._parser_version", lambda: "1.2.3")
    store = SQLiteStore(settings.app_db_path)
    paper_repo = PaperRepository(store)
    paper_repo.upsert(
        PaperRecord(
            paper_id="paper_2501.30001",
            arxiv_id=arxiv_id,
            arxiv_version="v1",
            title="Force reparse",
            authors=["A"],
            abstract="retrieval and memory",
            primary_category="cs.AI",
            categories=["cs.AI"],
            source_query="corpus_manifest",
            pdf_sha256=pdf_sha,
            parse_status="parsed",
            parser_name="docling",
            parser_version="1.2.3",
            chunker_name="docling_hybrid_chunker",
            chunker_config_hash="default",
        )
    )

    calls = {"parsed": 0}

    def _parse_pdf(path: Path):  # noqa: ARG001
        calls["parsed"] += 1
        return SimpleNamespace(ok=True, status="success", error=None, conversion=object())

    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.build_parent_child_rows",
        lambda **kwargs: SimpleNamespace(
            parent_rows=[
                {
                    "parent_id": "parent_1",
                    "paper_id": "paper_2501.30001",
                    "section_path": "__root__",
                    "section_heading": "__root__",
                    "section_type": "content",
                    "section_level": 1,
                    "parent_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "token_count": 1,
                    "char_count": 1,
                    "content_types": "[]",
                    "child_chunk_ids": "[]",
                    "text_hash": "x",
                    "parent_text": "x",
                }
            ],
            chunk_rows=[
                {
                    "chunk_id": "chunk_1",
                    "parent_id": "parent_1",
                    "paper_id": "paper_2501.30001",
                    "chunk_index": 0,
                    "section_path": "__root__",
                    "section_type": "content",
                    "content_type": "text",
                    "page_start": 1,
                    "page_end": 1,
                    "token_count": 1,
                    "char_start": None,
                    "char_end": None,
                    "docling_ref_ids": "[]",
                    "embedding_text_hash": "x",
                    "embedding_model": "",
                    "embedding_config_hash": "",
                    "indexed_embedding_text_hash": "",
                    "last_indexed_at": None,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "chunk_text": "x",
                }
            ],
        ),
    )

    pipeline = IngestPipeline(settings=settings)
    pipeline.parser = SimpleNamespace(parser_name="docling", parse_pdf=_parse_pdf)
    summary = pipeline.run(limit=1, force=True)

    assert summary.discovered == 1
    assert calls["parsed"] == 1
    assert summary.parsed == 1


def test_ingest_manifest_partial_progress_with_missing_pdf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("APP_CORPUS_MANIFEST", str(tmp_path / "corpus_manifest.json"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    payload = {
        "corpus_name": "x",
        "snapshot_date": "2026-05-10",
        "date_from": "2026-02-09",
        "date_to": "2026-05-10",
        "source": "arxiv_api",
        "source_query": "cat:cs.AI",
        "selection_policy_version": "v1",
        "selection_strategy": "strategy_b_stratified",
        "rate_limit_policy": "seq",
        "papers": [
            {
                "arxiv_id": "2501.90001",
                "arxiv_version": "v1",
                "title": "Missing PDF",
                "authors": ["A"],
                "abstract": "x",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "pdf_sha256": "",
            },
            {
                "arxiv_id": "2501.90002",
                "arxiv_version": "v1",
                "title": "Present PDF",
                "authors": ["A"],
                "abstract": "x",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "pdf_sha256": "",
            },
        ],
    }
    settings.app_corpus_manifest.write_text(json.dumps(payload), encoding="utf-8")
    present_pdf = settings.app_pdf_dir / "2501.90002v1.pdf"
    present_pdf.parent.mkdir(parents=True, exist_ok=True)
    present_pdf.write_bytes(b"pdf")

    monkeypatch.setattr("agentic_rag.ingest.pipeline._parser_version", lambda: "1.2.3")
    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.build_parent_child_rows",
        lambda **kwargs: SimpleNamespace(
            parent_rows=[
                {
                    "parent_id": "parent_1",
                    "paper_id": "paper_2501.90002",
                    "section_path": "__root__",
                    "section_heading": "__root__",
                    "section_type": "content",
                    "section_level": 1,
                    "parent_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "token_count": 1,
                    "char_count": 1,
                    "content_types": "[]",
                    "child_chunk_ids": "[]",
                    "text_hash": "x",
                    "parent_text": "x",
                }
            ],
            chunk_rows=[
                {
                    "chunk_id": "chunk_1",
                    "parent_id": "parent_1",
                    "paper_id": "paper_2501.90002",
                    "chunk_index": 0,
                    "section_path": "__root__",
                    "section_type": "content",
                    "content_type": "text",
                    "page_start": 1,
                    "page_end": 1,
                    "token_count": 1,
                    "char_start": None,
                    "char_end": None,
                    "docling_ref_ids": "[]",
                    "embedding_text_hash": "x",
                    "embedding_model": "",
                    "embedding_config_hash": "",
                    "indexed_embedding_text_hash": "",
                    "last_indexed_at": None,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "chunk_text": "x",
                }
            ],
        ),
    )
    pipeline = IngestPipeline(settings=settings)

    def _parse_ok(path: Path):  # noqa: ARG001
        return SimpleNamespace(ok=True, status="success", error=None, conversion=object())

    pipeline.parser = SimpleNamespace(
        parser_name="docling",
        parse_pdf=_parse_ok,
    )
    summary = pipeline.run(limit=2)
    assert summary.discovered == 2
    assert summary.download_failed == 1
    assert summary.parsed == 1


def test_ingest_transient_parse_failure_does_not_downgrade_parsed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("APP_CORPUS_MANIFEST", str(tmp_path / "corpus_manifest.json"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)
    arxiv_id = "2501.91001"
    _write_manifest(settings.app_corpus_manifest, arxiv_id=arxiv_id)
    pdf = settings.app_pdf_dir / f"{arxiv_id}v1.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"x")

    monkeypatch.setattr("agentic_rag.ingest.pipeline._parser_version", lambda: "1.2.3")
    store = SQLiteStore(settings.app_db_path)
    repo = PaperRepository(store)
    repo.upsert(
        PaperRecord(
            paper_id=f"paper_{arxiv_id}",
            arxiv_id=arxiv_id,
            arxiv_version="v1",
            title="Already Parsed",
            authors=["A"],
            abstract="x",
            primary_category="cs.AI",
            categories=["cs.AI"],
            source_query="corpus_manifest",
            pdf_sha256="bad-old-sha",
            parse_status="parsed",
            parser_name="docling",
            parser_version="1.2.3",
            chunker_name="docling_hybrid_chunker",
            chunker_config_hash="default",
        )
    )
    # Ensure manifest sha is empty so local file passes validation.
    payload = json.loads(settings.app_corpus_manifest.read_text(encoding="utf-8"))
    payload["papers"][0]["pdf_sha256"] = ""
    settings.app_corpus_manifest.write_text(json.dumps(payload), encoding="utf-8")

    pipeline = IngestPipeline(settings=settings)

    def _parse_fail(path: Path):  # noqa: ARG001
        return SimpleNamespace(ok=False, status="failure", error="boom", conversion=None)

    pipeline.parser = SimpleNamespace(
        parser_name="docling",
        parse_pdf=_parse_fail,
    )
    summary = pipeline.run(limit=1, force=True)
    assert summary.parse_failed == 1
    state = repo.get_ingest_state(f"paper_{arxiv_id}")
    assert state is not None
    assert state["parse_status"] == "parsed"
