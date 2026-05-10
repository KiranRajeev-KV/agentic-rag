from agentic_rag.agent.citations import validate_citations
from agentic_rag.agent.router import route_query, route_query_with_llm


def test_router_tool_and_refuse_paths() -> None:
    tool = route_query("search arxiv for memory agents")
    assert tool.action == "TOOL"

    refuse = route_query("what is the weather tomorrow")
    assert refuse.action == "REFUSE"


def test_citation_validator() -> None:
    ok, _ = validate_citations(
        answer="Claim [S1]",
        sources_block="Sources:\n[S1] source",
        allowed_source_ids={"S1"},
        allowed_tool_ids=set(),
        final_action="ANSWER_FROM_CONTEXT",
    )
    assert ok

    not_ok, _ = validate_citations(
        answer="Claim [S2]",
        sources_block="Sources:\n[S1] source",
        allowed_source_ids={"S1"},
        allowed_tool_ids=set(),
        final_action="ANSWER_FROM_CONTEXT",
    )
    assert not not_ok


def test_router_fallback_mode_when_llm_missing() -> None:
    decision, mode = route_query_with_llm(
        query="search arxiv for memory agents",
        llm_client=None,
        router_model="gpt-5-nano",
        memory_context=[],
    )
    # With no LLM client, routing should still deterministically catch explicit tool intents.
    assert mode in {"deterministic", "fallback"}
    assert decision.action == "TOOL"


def test_citation_validator_rejects_internal_ids() -> None:
    ok, _ = validate_citations(
        answer="Claim from chunk_123 [S1]",
        sources_block="Sources:\n[S1] source",
        allowed_source_ids={"S1"},
        allowed_tool_ids=set(),
        final_action="ANSWER_FROM_CONTEXT",
    )
    assert not ok
