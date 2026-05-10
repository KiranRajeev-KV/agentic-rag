from pathlib import Path

from agentic_rag.agent.memory import MemoryService
from agentic_rag.agent.nodes import AgentDependencies, AgentNodes
from agentic_rag.agent.router import route_query_with_llm
from agentic_rag.config import get_settings
from agentic_rag.storage.bootstrap import initialize_storage
from agentic_rag.storage.repositories import EpisodeRepository, SemanticMemoryRepository
from agentic_rag.storage.sqlite import SQLiteStore


class _FakeTraceWriter:
    def event(self, **kwargs):  # noqa: ANN003
        del kwargs

    def retrieval(self, **kwargs):  # noqa: ANN003
        del kwargs

    def tool(self, **kwargs):  # noqa: ANN003
        del kwargs

    def evidence(self, **kwargs):  # noqa: ANN003
        del kwargs

    def answer(self, **kwargs):  # noqa: ANN003
        del kwargs


class _FakeLLMClient:
    settings = type(
        "S",
        (),
        {
            "router_model": "gpt-5-nano",
            "answer_model": "gpt-5-nano",
            "evidence_model": "gpt-5-nano",
        },
    )()

    def enabled(self) -> bool:
        return False


def _init_settings(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)
    return settings


def _memory_service(settings):
    store = SQLiteStore(settings.app_db_path)
    return MemoryService(
        semantic_repo=SemanticMemoryRepository(store),
        episode_repo=EpisodeRepository(store),
    )


def test_episode_written_and_read(tmp_path: Path, monkeypatch) -> None:
    settings = _init_settings(tmp_path, monkeypatch)
    memory = _memory_service(settings)

    episode_id = memory.write_episode(
        thread_id="demo",
        turn_id="turn_2",
        user_query="What is SIRA?",
        route_action="RETRIEVE",
        route_confidence=0.7,
        retrieved_child_ids=["c1", "c2"],
        selected_parent_ids=["p1"],
        tool_calls=[],
        final_action="ANSWER_FROM_CONTEXT",
        final_answer_summary="SIRA summary",
    )
    assert episode_id.startswith("ep_")
    episodes = memory.read_episodes(thread_id="demo", limit=4)
    assert len(episodes) == 1
    assert episodes[0]["turn_id"] == "turn_2"


def test_followup_query_uses_conversation_context() -> None:
    decision, mode = route_query_with_llm(
        query="what about it",
        llm_client=None,
        router_model="gpt-5-nano",
        memory_context=[],
        conversation_summary="We discussed paper 2605.06641.",
        recent_turns=[{"role": "user", "content": "Explain GlazyBench"}],
        episodic_context=[],
    )
    assert mode == "fallback"
    assert decision.action == "RETRIEVE"


def test_load_state_reads_checkpoint_fields_and_emits_counts(tmp_path: Path, monkeypatch) -> None:
    settings = _init_settings(tmp_path, monkeypatch)
    memory = _memory_service(settings)
    nodes = AgentNodes(
        AgentDependencies(
            retrieval_service=object(),
            memory_service=memory,
            toolset=object(),
            trace_writer=_FakeTraceWriter(),
            llm_client=_FakeLLMClient(),
        )
    )
    out = nodes.load_state(
        {
            "thread_id": "demo",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
            "conversation_summary": "sum",
            "active_focus": "arxiv:2605.06641",
        }
    )
    assert out["conversation_summary"] == "sum"
    assert out["conversation_memory_read_count"] == 2
    assert out["active_focus"] == "arxiv:2605.06641"


def test_schema_does_not_include_manual_conversation_tables() -> None:
    schema = Path("src/agentic_rag/storage/schema.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS conversation_threads" not in schema
    assert "CREATE TABLE IF NOT EXISTS conversation_turns" not in schema
    assert "CREATE TABLE IF NOT EXISTS conversation_summaries" not in schema
