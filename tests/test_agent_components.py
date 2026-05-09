from agentic_rag.agent.citations import validate_citations
from agentic_rag.agent.router import route_query


def test_router_tool_and_refuse_paths() -> None:
    tool = route_query("search arxiv for memory agents")
    assert tool.action == "TOOL"

    refuse = route_query("what is the weather tomorrow")
    assert refuse.action == "REFUSE"


def test_citation_validator() -> None:
    ok, _ = validate_citations(
        answer="Claim [S1]",
        allowed_source_ids={"S1"},
        allowed_tool_ids=set(),
        require_source_citation=True,
    )
    assert ok

    not_ok, _ = validate_citations(
        answer="Claim [S2]",
        allowed_source_ids={"S1"},
        allowed_tool_ids=set(),
        require_source_citation=True,
    )
    assert not not_ok
