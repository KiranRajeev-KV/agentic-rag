from types import SimpleNamespace

from agentic_rag.agent.nodes import AgentDependencies, AgentNodes
from agentic_rag.llm.schemas import LLMCitationValidationOutput
from agentic_rag.tools.schemas import ArxivPaperMetadata, ArxivToolOutput, ToolStatus


class _FakeMemoryService:
    def __init__(self) -> None:
        self.writes = []
        self.episodes = []

    def read_recent(self, limit: int = 10):  # noqa: ANN001, ARG002
        return []

    def read_semantic(self, limit: int = 10):  # noqa: ANN001, ARG002
        return []

    def read_episodes(self, thread_id: str, limit: int = 6):  # noqa: ANN001, ARG002
        return []

    def write(self, **kwargs):  # noqa: ANN003
        self.writes.append(kwargs)
        return "mem_1"

    def write_episode(self, **kwargs):  # noqa: ANN003
        self.episodes.append(kwargs)
        return "ep_1"


class _FakeLLMClient:
    settings = SimpleNamespace(
        router_model="gpt-5-nano", answer_model="gpt-5-nano", evidence_model="gpt-5-nano"
    )

    def enabled(self) -> bool:
        return False


def _build_nodes(memory_service: _FakeMemoryService, toolset: object | None = None) -> AgentNodes:
    deps = AgentDependencies(
        retrieval_service=object(),
        memory_service=memory_service,
        toolset=toolset or SimpleNamespace(),
        trace_writer=SimpleNamespace(
            event=lambda **kwargs: None,  # noqa: ARG005
            retrieval=lambda **kwargs: None,  # noqa: ARG005
            tool=lambda **kwargs: None,  # noqa: ARG005
            evidence=lambda **kwargs: None,  # noqa: ARG005
            answer=lambda **kwargs: None,  # noqa: ARG005
        ),
        llm_client=_FakeLLMClient(),
    )
    return AgentNodes(deps=deps)


def _build_nodes_with_llm(memory_service: _FakeMemoryService, llm_client: object) -> AgentNodes:
    deps = AgentDependencies(
        retrieval_service=object(),
        memory_service=memory_service,
        toolset=SimpleNamespace(),
        trace_writer=SimpleNamespace(
            event=lambda **kwargs: None,  # noqa: ARG005
            retrieval=lambda **kwargs: None,  # noqa: ARG005
            tool=lambda **kwargs: None,  # noqa: ARG005
            evidence=lambda **kwargs: None,  # noqa: ARG005
            answer=lambda **kwargs: None,  # noqa: ARG005
        ),
        llm_client=llm_client,
    )
    return AgentNodes(deps=deps)


def test_memory_update_skips_normal_qa() -> None:
    memory_service = _FakeMemoryService()
    nodes = _build_nodes(memory_service)
    state = {
        "turn_id": "turn_1",
        "raw_user_query": "What do papers say about memory?",
        "final_action": "ANSWER_FROM_CONTEXT",
        "final_answer": "Answer [S1]",
    }
    nodes.memory_update(state)
    assert memory_service.writes == []


def test_memory_update_persists_checkpoint_for_refuse_turn() -> None:
    memory_service = _FakeMemoryService()
    nodes = _build_nodes(memory_service)
    state = {
        "turn_id": "turn_1",
        "thread_id": "demo",
        "raw_user_query": "weather tomorrow?",
        "final_action": "REFUSE",
        "final_answer": "I can only answer from indexed corpus.",
        "route_action": "REFUSE",
        "messages": [{"role": "user", "content": "weather tomorrow?"}],
    }
    out = nodes.memory_update(state)
    assert out["episode_id"] == "ep_1"
    assert out["messages"][0]["role"] == "assistant"
    assert memory_service.episodes


def test_answer_fallback_strips_internal_ids() -> None:
    memory_service = _FakeMemoryService()
    nodes = _build_nodes(memory_service)
    out = nodes.answer(
        {
            "raw_user_query": "question",
            "context_packets": [
                {
                    "source_id": "S1",
                    "title": "Paper",
                    "arxiv_id": "2501.00001",
                    "section_path": "1 Intro",
                    "page_start": 1,
                    "page_end": 2,
                    "highlighted_evidence": "chunk_abc says x",
                    "section_context": "parent_123 context",
                }
            ],
        }
    )
    assert "chunk_" not in out["final_answer"]
    assert "parent_" not in out["final_answer"]


def test_tool_node_supports_lookup_by_id_route() -> None:
    class _Toolset:
        def arxiv_lookup_by_id(self, payload):  # noqa: ANN001
            assert payload.arxiv_ids == ["2501.00001"]
            return ArxivToolOutput(
                status=ToolStatus.ok,
                papers=[
                    ArxivPaperMetadata(
                        arxiv_id="2501.00001",
                        version="v1",
                        title="Lookup Paper",
                        authors=["A"],
                        categories=["cs.AI"],
                    )
                ],
            )

        def arxiv_search(self, payload):  # noqa: ANN001
            del payload
            raise AssertionError("search should not be called")

    nodes = _build_nodes(_FakeMemoryService(), toolset=_Toolset())
    state = {
        "raw_user_query": "lookup 2501.00001",
        "rewritten_query": "lookup 2501.00001",
        "tool_name": "arxiv_lookup_by_id",
        "tool_args": {"arxiv_ids": ["2501.00001"]},
    }
    out = nodes.tool(state)
    assert out["final_action"] == "ANSWER_FROM_TOOL"
    assert "Unsupported tool route" not in out["final_answer"]


def test_tool_citation_validate_uses_sources_ids_for_tool_answers() -> None:
    nodes = _build_nodes(_FakeMemoryService())
    state = {
        "final_action": "ANSWER_FROM_TOOL",
        "final_answer": "Here are results [T1]\n[T2] Lookup Paper arXiv:2501.00001",
        "sources_block": (
            "Sources:\n[T1] arXiv API arxiv_lookup_by_id ids: 2501.00001\n"
            "[T2] Lookup Paper, arXiv:2501.00001"
        ),
        "citations": ["T1", "T2"],
        "tool_result": {"papers": [{"arxiv_id": "2501.00001"}]},
        "context_packets": [],
        "evidence_status": "SUFFICIENT",
    }
    out = nodes.citation_validate(state)
    assert out.get("final_answer", "").startswith("Here are results [T1]")


def test_tool_lookup_by_id_keeps_full_abstract() -> None:
    full_abstract = "A" * 420 + " END_MARKER"

    class _Toolset:
        def arxiv_lookup_by_id(self, payload):  # noqa: ANN001
            del payload
            return ArxivToolOutput(
                status=ToolStatus.ok,
                papers=[
                    ArxivPaperMetadata(
                        arxiv_id="2501.00001",
                        version="v1",
                        title="Lookup Paper",
                        authors=["A", "B", "C", "D"],
                        abstract=full_abstract,
                        categories=["cs.AI"],
                    )
                ],
            )

    nodes = _build_nodes(_FakeMemoryService(), toolset=_Toolset())
    out = nodes.tool(
        {
            "raw_user_query": "lookup 2501.00001",
            "rewritten_query": "lookup 2501.00001",
            "tool_name": "arxiv_lookup_by_id",
            "tool_args": {"arxiv_ids": ["2501.00001"]},
        }
    )
    answer = out["final_answer"]
    assert "END_MARKER" in answer
    assert "Abstract: " in answer
    assert "et al." in answer


def test_tool_search_truncates_abstract() -> None:
    long_abstract = "B" * 500 + " TAIL_MARKER"

    class _Toolset:
        def arxiv_search(self, payload):  # noqa: ANN001
            del payload
            return ArxivToolOutput(
                status=ToolStatus.ok,
                papers=[
                    ArxivPaperMetadata(
                        arxiv_id="2501.00002",
                        version="v1",
                        title="Search Paper",
                        authors=["A"],
                        abstract=long_abstract,
                        categories=["cs.AI"],
                    )
                ],
            )

    nodes = _build_nodes(_FakeMemoryService(), toolset=_Toolset())
    out = nodes.tool(
        {
            "raw_user_query": "search arxiv transformers",
            "rewritten_query": "search arxiv transformers",
            "tool_name": "arxiv_search",
            "tool_args": {"query": "transformers"},
        }
    )
    answer = out["final_answer"]
    assert "Abstract: " in answer
    assert "TAIL_MARKER" not in answer
    assert "..." in answer


def test_citation_validate_refuses_when_llm_semantic_validation_fails() -> None:
    class _LLMClient:
        settings = SimpleNamespace(
            router_model="gpt-5-nano",
            answer_model="gpt-5-nano",
            evidence_model="gpt-5-nano",
        )

        def enabled(self) -> bool:
            return True

        def complete_json(self, **kwargs):  # noqa: ANN003
            schema = kwargs["schema"]
            if schema is LLMCitationValidationOutput:
                return LLMCitationValidationOutput(
                    valid=False,
                    verdict="UNSUPPORTED_CLAIM",
                    unsupported_claims=["claim 1"],
                    missing_citation_spans=[],
                    unknown_citation_ids=[],
                    internal_id_leaks=[],
                    repair_instruction="remove unsupported claim",
                )
            raise AssertionError("Unexpected schema")

    nodes = _build_nodes_with_llm(_FakeMemoryService(), _LLMClient())
    state = {
        "final_action": "ANSWER_FROM_CONTEXT",
        "final_answer": "Claim [S1]",
        "sources_block": "Sources:\\n[S1] Source",
        "citations": ["S1"],
        "context_packets": [{"source_id": "S1"}],
        "evidence_status": "SUFFICIENT",
        "raw_user_query": "q",
        "rewritten_query": "q",
    }
    out = nodes.citation_validate(state)
    assert out["final_action"] == "REFUSE"


def test_load_state_emits_memory_trace_events() -> None:
    events = []

    class _TraceWriter:
        def event(self, **kwargs):  # noqa: ANN003
            events.append(kwargs)

        def retrieval(self, **kwargs):  # noqa: ANN003
            del kwargs

        def tool(self, **kwargs):  # noqa: ANN003
            del kwargs

        def evidence(self, **kwargs):  # noqa: ANN003
            del kwargs

        def answer(self, **kwargs):  # noqa: ANN003
            del kwargs

    deps = AgentDependencies(
        retrieval_service=object(),
        memory_service=_FakeMemoryService(),
        toolset=SimpleNamespace(),
        trace_writer=_TraceWriter(),
        llm_client=_FakeLLMClient(),
    )
    nodes = AgentNodes(deps=deps)
    out = nodes.load_state({"trace_id": "tr_1", "thread_id": "demo"})
    assert "conversation_summary" in out
    event_names = [event["event"] for event in events]
    assert "memory.checkpoint_read" in event_names
    assert "memory.semantic_read" in event_names
    assert "memory.episodic_read" in event_names


def test_evidence_branch_routes_contradiction_handler() -> None:
    nodes = _build_nodes(_FakeMemoryService())
    branch = nodes.evidence_branch({"final_action": "CONTRADICTION_HANDLER"})
    assert branch == "contradiction_handler"
