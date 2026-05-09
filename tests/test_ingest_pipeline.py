from pathlib import Path
from types import SimpleNamespace

from agentic_rag.config import get_settings
from agentic_rag.ingest.pipeline import IngestPipeline
from agentic_rag.storage.bootstrap import initialize_storage
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
            [SimpleNamespace(metadata=paper, source_query="q", filter_terms=["agentic"])],
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
