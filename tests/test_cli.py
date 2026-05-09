from pathlib import Path

from typer.testing import CliRunner

from agentic_rag.cli import app
from agentic_rag.config import get_settings

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
