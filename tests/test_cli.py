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


def test_ingest_with_optional_index(monkeypatch) -> None:
    class _FakePipeline:
        def __init__(self, settings) -> None:  # noqa: ANN001
            self.settings = settings

        def run(self, limit: int, days_back: int = 90) -> IngestSummary:
            return IngestSummary(
                requested_limit=limit,
                discovered=1,
                downloaded=1,
                parsed=1,
                parse_failed=0,
                download_failed=0,
                skipped=0,
                errors=[],
            )

    calls = {"indexed": False}

    def _fake_run_index(settings, limit: int, batch_size: int, force: bool) -> None:  # noqa: ANN001
        calls["indexed"] = True
        assert limit == 500
        assert batch_size == 32
        assert force is False

    monkeypatch.setattr(cli_module, "IngestPipeline", _FakePipeline)
    monkeypatch.setattr(cli_module, "initialize_storage", lambda settings, init_qdrant: None)
    monkeypatch.setattr(cli_module, "_run_index", _fake_run_index)

    result = runner.invoke(app, ["ingest", "--limit", "20", "--index"])
    assert result.exit_code == 0
    assert calls["indexed"] is True


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


def test_ask_with_mocked_graph(monkeypatch) -> None:
    class _FakeRunner:
        def __init__(self, settings) -> None:  # noqa: ANN001
            self.settings = settings

        def run(self, query: str):  # noqa: ANN001
            assert query == "What is agent memory?"
            return {
                "trace_id": "tr_test",
                "route_action": "RETRIEVE",
                "retrieved_child_ids": ["c1", "c2"],
                "selected_parent_ids": ["p1"],
                "evidence_status": "SUFFICIENT",
                "context_packets": [{"source_id": "S1"}],
                "final_action": "ANSWER_FROM_CONTEXT",
                "final_answer": "Answer body [S1]\n\nSources:\n[S1] Test source",
            }

    monkeypatch.setattr(cli_module, "AskGraphRunner", _FakeRunner)
    monkeypatch.setattr(cli_module, "initialize_storage", lambda settings, init_qdrant: None)
    result = runner.invoke(app, ["ask", "What is agent memory?", "--debug"])
    assert result.exit_code == 0
    assert "Answer body [S1]" in result.stdout
    assert "Trace: tr_test" in result.stdout


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

        def index_unembedded_chunks(self, limit: int, batch_size: int, force: bool):  # noqa: ANN001
            assert limit == 10
            assert batch_size == 4
            assert force is True
            return SimpleNamespace(
                total_chunks=10,
                pending_chunks=4,
                selected_chunks=2,
                indexed_chunks=2,
                skipped_as_up_to_date=6,
                force=True,
                model_name="BAAI/bge-m3",
                config_hash="abcd1234efgh5678",
                device="cpu",
            )

    monkeypatch.setattr(cli_module, "ChildChunkIndexer", _FakeIndexer)
    result = runner.invoke(app, ["index", "--limit", "10", "--batch-size", "4", "--force"])
    assert result.exit_code == 0
    assert "index.progress total_chunks=10 pending=4 up_to_date=6" in result.stdout
    assert "index.summary selected=2 indexed=2 model=BAAI/bge-m3" in result.stdout


def test_eval_commands_with_mocked_runner(monkeypatch) -> None:
    class _FakeEvalRunner:
        def __init__(self, settings) -> None:  # noqa: ANN001
            self.settings = settings

        def run_variant(self, variant):  # noqa: ANN001
            assert str(variant) in {"child_only", "parent_child"}
            return {
                "variant": str(variant),
                "cases": 14,
                "raw_score": 100.0,
                "normalized_score": 71.4,
                "hard_fail_refusal": False,
            }

        def compare(self, baseline, candidate):  # noqa: ANN001
            return {
                "baseline": str(baseline),
                "candidate": str(candidate),
                "baseline_score": 60.0,
                "candidate_score": 70.0,
                "delta_score": 10.0,
            }

    monkeypatch.setattr(cli_module, "EvalRunner", _FakeEvalRunner)
    monkeypatch.setattr(cli_module, "initialize_storage", lambda settings, init_qdrant: None)

    run_result = runner.invoke(app, ["eval", "run", "--variant", "child_only"])
    assert run_result.exit_code == 0
    assert "eval.summary variant=child_only cases=14" in run_result.stdout

    compare_result = runner.invoke(
        app,
        ["eval", "compare", "--baseline", "child_only", "--candidate", "parent_child"],
    )
    assert compare_result.exit_code == 0
    assert "eval.compare baseline=child_only candidate=parent_child" in compare_result.stdout
