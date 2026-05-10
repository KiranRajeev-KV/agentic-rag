from pathlib import Path

from agentic_rag.agent.graph import AskGraphRunner
from agentic_rag.config import get_settings
from agentic_rag.storage.bootstrap import initialize_storage


def _setup_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)


class _FakeGraph:
    def __init__(self) -> None:
        self.config = None

    def invoke(self, state, config=None):  # noqa: ANN001
        self.config = config
        return {
            **state,
            "final_action": "REFUSE",
            "final_answer": "refused",
        }


def test_ask_runner_invokes_graph_with_thread_id_config(tmp_path: Path, monkeypatch) -> None:
    _setup_env(tmp_path, monkeypatch)
    settings = get_settings()
    fake_graph = _FakeGraph()

    monkeypatch.setattr(AskGraphRunner, "_compile_graph", lambda self: fake_graph)

    runner = AskGraphRunner(settings=settings)
    out = runner.run("test question", thread_id="demo-thread")

    assert out["thread_id"] == "demo-thread"
    assert fake_graph.config == {"configurable": {"thread_id": "demo-thread"}}
