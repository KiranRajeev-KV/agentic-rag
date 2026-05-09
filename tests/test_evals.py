from pathlib import Path

from agentic_rag.evals.runner import EvalRunner
from agentic_rag.retrieval.types import RetrievalVariant


class _FakeGraphRunner:
    def run(self, query: str, thread_id: str = "default", retrieval_variant=None):  # noqa: ANN001, ARG002
        action = "ANSWER_FROM_TOOL" if "search arxiv" in query.lower() else "ANSWER_FROM_CONTEXT"
        route = "TOOL" if action == "ANSWER_FROM_TOOL" else "RETRIEVE"
        answer = (
            "Answer [S1]\n\nSources:\n[S1] test" if action == "ANSWER_FROM_CONTEXT" else "Tool [T1]"
        )
        return {
            "route_action": route,
            "final_action": action,
            "final_answer": answer,
            "selected_parent_ids": ["p1"],
            "context_packets": [
                {"paper_id": "paper1", "section_context": "x", "highlighted_evidence": "y"}
            ],
            "trace_id": "tr_x",
            "tool_name": "arxiv_search" if action == "ANSWER_FROM_TOOL" else "",
        }


def test_eval_runner_variant_and_compare(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))

    from agentic_rag.config import get_settings
    from agentic_rag.storage.bootstrap import initialize_storage

    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)

    runner = EvalRunner(settings=settings, graph_runner=_FakeGraphRunner())
    child = runner.run_variant(RetrievalVariant.child_only)
    parent = runner.run_variant(RetrievalVariant.parent_child)
    comparison = runner.compare(RetrievalVariant.child_only, RetrievalVariant.parent_child)

    assert child["cases"] == 14
    assert parent["cases"] == 14
    assert comparison["baseline"] == "child_only"
