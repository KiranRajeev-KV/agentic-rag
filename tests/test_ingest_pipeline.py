import hashlib
from pathlib import Path
from types import SimpleNamespace

from agentic_rag.config import get_settings
from agentic_rag.ingest.pipeline import IngestPipeline
from agentic_rag.storage.bootstrap import initialize_storage
from agentic_rag.storage.repositories import PaperRecord, PaperRepository
from agentic_rag.storage.sqlite import SQLiteStore
from agentic_rag.tools.schemas import ArxivPaperMetadata


def test_ingest_pipeline_parse_failure_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    paper = ArxivPaperMetadata(
        arxiv_id="2501.10001",
        version="v1",
        title="Agentic systems",
        authors=["A"],
        abstract="retrieval and memory",
        categories=["cs.AI"],
        pdf_url="https://example.com/paper.pdf",
    )

    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.discover_relevant_papers",
        lambda **kwargs: (
            [
                SimpleNamespace(
                    metadata=paper,
                    source_query="q",
                    filter_terms=["agentic"],
                    filter_reason="matched_terms=agentic",
                    days_back=90,
                )
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.download_pdf",
        lambda **kwargs: SimpleNamespace(
            ok=True, paper_path=tmp_path / "paper.pdf", pdf_sha256="abc", error=None
        ),
    )

    class _Parser:
        parser_name = "docling"

        def parse_pdf(self, pdf_path: Path) -> SimpleNamespace:  # noqa: ARG002
            return SimpleNamespace(ok=False, status="failure", error="parse error", conversion=None)

    pipeline = IngestPipeline(settings=settings)
    pipeline.parser = _Parser()
    summary = pipeline.run(limit=1, days_back=90)

    assert summary.discovered == 1
    assert summary.downloaded == 1
    assert summary.parse_failed == 1


def test_ingest_pipeline_skips_unchanged_parsed_paper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    paper = ArxivPaperMetadata(
        arxiv_id="2501.20001",
        version="v1",
        title="Stable ingest caching",
        authors=["A"],
        abstract="retrieval and memory",
        categories=["cs.AI"],
        pdf_url="https://example.com/paper.pdf",
    )
    pdf_path = settings.app_pdf_dir / "2501.20001v1.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = b"pdf-content"
    pdf_path.write_bytes(pdf_bytes)
    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()

    monkeypatch.setattr("agentic_rag.ingest.pipeline._parser_version", lambda: "1.2.3")
    store = SQLiteStore(settings.app_db_path)
    paper_repo = PaperRepository(store)
    paper_repo.upsert(
        PaperRecord(
            paper_id="paper_2501.20001",
            arxiv_id=paper.arxiv_id,
            arxiv_version=paper.version,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract or "",
            primary_category=paper.primary_category,
            categories=paper.categories,
            published_at=None,
            updated_at=None,
            pdf_url=paper.pdf_url,
            abs_url=paper.abs_url,
            doi=paper.doi,
            comment=paper.comment,
            source_query="q",
            pdf_sha256=pdf_sha,
            parse_status="parsed",
            parser_name="docling",
            parser_version="1.2.3",
            chunker_name="docling_hybrid_chunker",
            chunker_config_hash="default",
        )
    )

    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.discover_relevant_papers",
        lambda **kwargs: (
            [
                SimpleNamespace(
                    metadata=paper,
                    source_query="q",
                    filter_terms=["agentic"],
                    filter_reason="matched_terms=agentic",
                    days_back=90,
                )
            ],
            [],
        ),
    )

    def _unexpected_download(**kwargs):  # noqa: ANN003, ARG001
        raise AssertionError("download_pdf should not be called for unchanged parsed paper")

    monkeypatch.setattr("agentic_rag.ingest.pipeline.download_pdf", _unexpected_download)

    pipeline = IngestPipeline(settings=settings)
    summary = pipeline.run(limit=1, days_back=90)
    assert summary.discovered == 1
    assert summary.skipped == 1
    assert summary.downloaded == 0
    assert summary.parsed == 0


def test_ingest_pipeline_force_bypasses_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    paper = ArxivPaperMetadata(
        arxiv_id="2501.30001",
        version="v1",
        title="Force reparse",
        authors=["A"],
        abstract="retrieval and memory",
        categories=["cs.AI"],
        pdf_url="https://example.com/paper.pdf",
    )
    pdf_path = settings.app_pdf_dir / "2501.30001v1.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = b"cached-pdf"
    pdf_path.write_bytes(pdf_bytes)
    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()

    monkeypatch.setattr("agentic_rag.ingest.pipeline._parser_version", lambda: "1.2.3")
    store = SQLiteStore(settings.app_db_path)
    paper_repo = PaperRepository(store)
    paper_repo.upsert(
        PaperRecord(
            paper_id="paper_2501.30001",
            arxiv_id=paper.arxiv_id,
            arxiv_version=paper.version,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract or "",
            primary_category=paper.primary_category,
            categories=paper.categories,
            source_query="q",
            pdf_sha256=pdf_sha,
            parse_status="parsed",
            parser_name="docling",
            parser_version="1.2.3",
            chunker_name="docling_hybrid_chunker",
            chunker_config_hash="default",
        )
    )

    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.discover_relevant_papers",
        lambda **kwargs: (
            [
                SimpleNamespace(
                    metadata=paper,
                    source_query="q",
                    filter_terms=["agentic"],
                    filter_reason="matched_terms=agentic",
                    days_back=90,
                )
            ],
            [],
        ),
    )

    calls = {"downloaded": 0, "parsed": 0}
    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.download_pdf",
        lambda **kwargs: (
            calls.__setitem__("downloaded", calls["downloaded"] + 1)
            or SimpleNamespace(
                ok=True, paper_path=tmp_path / "paper.pdf", pdf_sha256="new", error=None
            )
        ),
    )

    class _Parser:
        parser_name = "docling"

        def parse_pdf(self, pdf_path: Path) -> SimpleNamespace:  # noqa: ARG002
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
    pipeline.parser = _Parser()
    summary = pipeline.run(limit=1, days_back=90, force=True)
    assert summary.skipped == 0
    assert summary.downloaded == 1
    assert summary.parsed == 1
    assert calls["downloaded"] == 1
    assert calls["parsed"] == 1


def test_ingest_persists_discovery_filter_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    paper = ArxivPaperMetadata(
        arxiv_id="2501.40001",
        version="v1",
        title="Agentic systems",
        authors=["A"],
        abstract="retrieval and memory",
        categories=["cs.AI"],
        pdf_url="https://example.com/paper.pdf",
    )
    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.discover_relevant_papers",
        lambda **kwargs: (
            [
                SimpleNamespace(
                    metadata=paper,
                    source_query="cs.AI recent papers",
                    filter_terms=["agentic", "memory"],
                    filter_reason="matched_terms=agentic,memory;title_terms=agentic;abstract_terms=memory",
                    days_back=90,
                )
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        "agentic_rag.ingest.pipeline.download_pdf",
        lambda **kwargs: SimpleNamespace(
            ok=False, paper_path=None, pdf_sha256=None, error="mocked download fail"
        ),
    )
    pipeline = IngestPipeline(settings=settings)
    pipeline.run(limit=1, days_back=90)
    rows = SQLiteStore(settings.app_db_path).fetchall(
        """
        SELECT
          source_query,
          discovery_source_query,
          discovery_filter_terms,
          discovery_filter_reason,
          discovery_days_back
        FROM papers
        WHERE paper_id = ?
        """,
        ("paper_2501.40001",),
    )
    assert rows
    row = dict(rows[0])
    assert row["source_query"] == "cs.AI recent papers"
    assert row["discovery_source_query"] == "cs.AI recent papers"
    assert "agentic" in str(row["discovery_filter_terms"])
    assert "matched_terms=" in str(row["discovery_filter_reason"])
    assert int(row["discovery_days_back"]) == 90
