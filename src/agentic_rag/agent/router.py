from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.retrieval.types import RetrievalVariant

from .state import RouteAction


class RouterDecision(BaseModel):
    action: RouteAction
    route_confidence: float = Field(ge=0.0, le=1.0)
    route_reason_public: str
    rewritten_query: str
    retrieval_filters: dict[str, str] = Field(default_factory=dict)
    tool_name: str | None = None
    tool_args: dict[str, str] = Field(default_factory=dict)
    clarifying_question: str | None = None
    refusal_reason: str | None = None
    expected_next_node: str
    retrieval_variant: RetrievalVariant = RetrievalVariant.parent_child


def route_query(query: str) -> RouterDecision:
    lowered = query.strip().lower()
    if not lowered:
        return RouterDecision(
            action="CLARIFY",
            route_confidence=0.9,
            route_reason_public="The query is empty.",
            rewritten_query=query,
            clarifying_question="What question should I answer from the indexed arXiv corpus?",
            expected_next_node="clarify",
        )

    if _needs_tool(lowered):
        return RouterDecision(
            action="TOOL",
            route_confidence=0.75,
            route_reason_public="The query asks for arXiv metadata/search info.",
            rewritten_query=query,
            tool_name="arxiv_search",
            tool_args={"query": query},
            expected_next_node="tool",
        )

    if _out_of_domain(lowered):
        return RouterDecision(
            action="REFUSE",
            route_confidence=0.8,
            route_reason_public="The query is outside the indexed arXiv corpus scope.",
            rewritten_query=query,
            refusal_reason="I can only answer from the indexed recent arXiv cs.AI corpus.",
            expected_next_node="refuse",
        )

    if _ambiguous(lowered):
        return RouterDecision(
            action="CLARIFY",
            route_confidence=0.65,
            route_reason_public="The query is missing a concrete target or criterion.",
            rewritten_query=query,
            clarifying_question=(
                "Could you specify the paper, method, or comparison criterion you want?"
            ),
            expected_next_node="clarify",
        )

    variant = (
        RetrievalVariant.child_only
        if _requests_exact_quote(lowered)
        else RetrievalVariant.parent_child
    )
    return RouterDecision(
        action="RETRIEVE",
        route_confidence=0.7,
        route_reason_public="The query appears answerable from indexed paper content.",
        rewritten_query=query,
        expected_next_node="retrieve",
        retrieval_variant=variant,
    )


def _needs_tool(query: str) -> bool:
    markers = ("arxiv id", "paper id", "latest paper", "find papers", "search arxiv", "title by id")
    return any(marker in query for marker in markers)


def _out_of_domain(query: str) -> bool:
    blocked = ("weather", "stock price", "sports score", "restaurant near me")
    return any(marker in query for marker in blocked)


def _ambiguous(query: str) -> bool:
    return query in {"which is better", "best method", "what about it"}


def _requests_exact_quote(query: str) -> bool:
    return any(marker in query for marker in ("exact quote", "quote", "verbatim"))
