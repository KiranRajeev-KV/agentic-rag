from typer.testing import CliRunner

from agentic_rag.cli import app

runner = CliRunner()


def test_help_renders() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Agentic RAG local-first CLI" in result.stdout


def test_ingest_stub() -> None:
    result = runner.invoke(app, ["ingest", "--limit", "20"])
    assert result.exit_code == 0
    assert "[stub] ingest requested with --limit=20" in result.stdout
