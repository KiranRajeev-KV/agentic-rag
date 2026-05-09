from pathlib import Path

from typer.testing import CliRunner

import agentic_rag.cli as cli_module
from agentic_rag.cli import app
from agentic_rag.config import get_settings
from agentic_rag.tools.schemas import ArxivPaperMetadata, ArxivToolOutput, ToolStatus

runner = CliRunner()


def test_help_renders() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Agentic RAG local-first CLI" in result.stdout


def test_ingest_stub() -> None:
    result = runner.invoke(app, ["ingest", "--limit", "20"])
    assert result.exit_code == 0
    assert "[stub] ingest requested with --limit=20" in result.stdout


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
