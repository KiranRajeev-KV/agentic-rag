from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.llm.client import OpenAILLMClient
from agentic_rag.llm.prompts import ROUTER_SYSTEM_PROMPT, router_user_prompt
from agentic_rag.llm.schemas import LLMRouterOutput
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
    return _fallback_route_query(query)


def route_query_with_llm(
    *,
    query: str,
    llm_client: OpenAILLMClient | None,
    router_model: str,
    memory_context: list[dict[str, str]],
    conversation_summary: str = "",
    recent_turns: list[dict[str, str]] | None = None,
    episodic_context: list[dict[str, str]] | None = None,
) -> tuple[RouterDecision, str]:
    if llm_client is None or not llm_client.enabled():
        return (
            _fallback_route_query_with_context(
                query=query,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns or [],
                episodic_context=episodic_context or [],
            ),
            "fallback",
        )
    try:
        llm_output = llm_client.complete_json(
            model=router_model,
            schema=LLMRouterOutput,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_prompt=router_user_prompt(
                query=query,
                memory_context=memory_context,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns or [],
                episodic_context=episodic_context or [],
            ),
        )
        return (
            RouterDecision(
                action=llm_output.action,
                route_confidence=llm_output.route_confidence,
                route_reason_public=llm_output.route_reason_public,
                rewritten_query=llm_output.rewritten_query,
                retrieval_filters={k: str(v) for k, v in llm_output.retrieval_filters.items()},
                tool_name=llm_output.tool_name,
                tool_args={k: str(v) for k, v in llm_output.tool_args.items()},
                clarifying_question=llm_output.clarifying_question,
                refusal_reason=llm_output.refusal_reason,
                expected_next_node=llm_output.expected_next_node,
                retrieval_variant=RetrievalVariant(llm_output.retrieval_variant),
            ),
            "llm",
        )
    except Exception:  # noqa: BLE001
        return (
            _fallback_route_query_with_context(
                query=query,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns or [],
                episodic_context=episodic_context or [],
            ),
            "fallback",
        )


def _fallback_route_query_with_context(
    *,
    query: str,
    conversation_summary: str,
    recent_turns: list[dict[str, str]],
    episodic_context: list[dict[str, str]],
) -> RouterDecision:
    lowered = query.strip().lower()
    follow_up_markers = ("continue", "what about it", "that paper", "previous point", "and that?")
    if any(marker in lowered for marker in follow_up_markers):
        if conversation_summary.strip() or recent_turns or episodic_context:
            rewritten = (
                f"{query}\n\nConversation summary: {conversation_summary[:240]}"
                if conversation_summary
                else query
            )
            return RouterDecision(
                action="RETRIEVE",
                route_confidence=0.72,
                route_reason_public="Follow-up query resolved from thread conversation memory.",
                rewritten_query=rewritten,
                expected_next_node="retrieve",
                retrieval_variant=RetrievalVariant.parent_child,
            )
    return _fallback_route_query(query)


def _fallback_route_query(query: str) -> RouterDecision:
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
