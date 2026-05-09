from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import agentic_rag.cli as cli_module
from agentic_rag.cli import app
from agentic_rag.config import get_settings
from agentic_rag.ingest.pipeline import IngestSummary
from agentic_rag.tools.schemas import ArxivPaperMetadata, ArxivToolOutput, ToolStatus

runner = CliRunner()


def test_help_renders() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Agentic RAG local-first CLI" in result.stdout


def test_ingest_with_mocked_pipeline(monkeypatch) -> None:
    class _FakePipeline:
        def __init__(self, settings) -> None:  # noqa: ANN001
            self.settings = settings

        def run(self, limit: int, days_back: int = 90) -> IngestSummary:
            assert limit == 20
            assert days_back == 90
            return IngestSummary(
                requested_limit=20,
                discovered=3,
                downloaded=3,
                parsed=2,
                parse_failed=1,
                download_failed=0,
                skipped=0,
                errors=[],
            )

    monkeypatch.setattr(cli_module, "IngestPipeline", _FakePipeline)
    monkeypatch.setattr(cli_module, "initialize_storage", lambda settings, init_qdrant: None)

    result = runner.invoke(app, ["ingest", "--limit", "20"])
    assert result.exit_code == 0
    assert "ingest.summary requested=20 discovered=3 downloaded=3 parsed=2" in result.stdout


def test_db_init_and_trace_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()

    init_result = runner.invoke(app, ["db", "init"])
    assert init_result.exit_code == 0
    assert "Initialized storage: sqlite" in init_result.stdout

    trace_result = runner.invoke(app, ["trace", "list", "--last", "5"])
    assert trace_result.exit_code == 0
    assert "No traces found." in trace_result.stdout


def test_corpus_discover_with_mocked_tool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()

    class _FakeToolset:
        def __init__(self, settings) -> None:  # noqa: ANN001
            self.settings = settings

        def arxiv_get_recent(self, payload) -> ArxivToolOutput:  # noqa: ANN001
            del payload
            return ArxivToolOutput(
                status=ToolStatus.ok,
                source="arxiv_api",
                papers=[
                    ArxivPaperMetadata(
                        arxiv_id="2501.00001",
                        version="v1",
                        title="Mock Paper",
                        authors=["A. Author"],
                        abstract="Summary",
                        categories=["cs.AI"],
                    )
                ],
            )

    monkeypatch.setattr(cli_module, "ArxivToolset", _FakeToolset)

    result = runner.invoke(app, ["corpus", "discover", "--limit", "20"])
    assert result.exit_code == 0
    assert "discovered=1 status=ok source=arxiv_api" in result.stdout


def test_index_with_mocked_indexer(monkeypatch) -> None:
    class _FakeIndexer:
        def __init__(self, settings) -> None:  # noqa: ANN001
            self.settings = settings

        def index_unembedded_chunks(self, limit: int, batch_size: int):  # noqa: ANN001
            assert limit == 10
            assert batch_size == 4
            return SimpleNamespace(
                selected_chunks=2,
                indexed_chunks=2,
                model_name="BAAI/bge-m3",
                device="cpu",
            )

    monkeypatch.setattr(cli_module, "ChildChunkIndexer", _FakeIndexer)
    result = runner.invoke(app, ["index", "--limit", "10", "--batch-size", "4"])
    assert result.exit_code == 0
    assert "index.summary selected=2 indexed=2 model=BAAI/bge-m3 device=cpu" in result.stdout
