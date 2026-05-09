from pathlib import Path

from agentic_rag.agent.memory import MemoryService
from agentic_rag.agent.router import route_query_with_llm
from agentic_rag.config import get_settings
from agentic_rag.storage.bootstrap import initialize_storage
from agentic_rag.storage.repositories import (
    ConversationRepository,
    EpisodeRepository,
    SemanticMemoryRepository,
)
from agentic_rag.storage.sqlite import SQLiteStore


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
        conversation_repo=ConversationRepository(store),
        episode_repo=EpisodeRepository(store),
    )


def test_conversation_memory_persistence(tmp_path: Path, monkeypatch) -> None:
    settings = _init_settings(tmp_path, monkeypatch)
    memory = _memory_service(settings)

    memory.update_conversation_after_turn(
        thread_id="demo",
        turn_id="turn_1",
        user_query="Tell me about arxiv:2605.06641",
        assistant_answer="GlazyBench details [S1]",
        route_action="RETRIEVE",
        final_action="ANSWER_FROM_CONTEXT",
    )

    conv = memory.read_conversation(thread_id="demo", limit=6)
    assert len(conv["recent_turns"]) == 2
    assert conv["active_focus"].startswith("arxiv:")
    assert conv["summary"]


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
