from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.llm.client import OpenAILLMClient
from agentic_rag.llm.prompts import (
    INTENT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    intent_user_prompt,
    router_user_prompt,
)
from agentic_rag.llm.schemas import LLMIntentOutput, LLMRouterOutput
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


INTENT_CONFIDENCE_THRESHOLD = 0.65
RETRIEVE_CONFIDENCE_THRESHOLD = 0.55
TOOL_CONFIDENCE_THRESHOLD = 0.55


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
    lowered = query.strip().lower()
    deterministic_tool = _tool_route(lowered)
    if deterministic_tool:
        tool_args: dict[str, str] = {"query": query.strip()}
        if deterministic_tool == "arxiv_lookup_by_id":
            tool_args = {"arxiv_ids": ",".join(_extract_arxiv_ids(query))}
        decision = RouterDecision(
            action="TOOL",
            route_confidence=0.9,
            route_reason_public="Deterministic arXiv metadata/search intent match.",
            rewritten_query=query.strip(),
            tool_name=deterministic_tool,
            tool_args=tool_args,
            expected_next_node="tool",
        )
        return (_apply_confidence_thresholds(decision), "deterministic")
    if llm_client is None or not llm_client.enabled():
        decision = _fallback_route_query_with_context(
                query=query,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns or [],
                episodic_context=episodic_context or [],
        )
        return (_apply_confidence_thresholds(decision), "fallback")
    intent_decision = _intent_route_with_llm(
        query=query,
        llm_client=llm_client,
        router_model=router_model,
        conversation_summary=conversation_summary,
        recent_turns=recent_turns or [],
    )
    if intent_decision is not None:
        return (_apply_confidence_thresholds(intent_decision), "llm_intent")
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
        decision = RouterDecision(
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
        )
        return (_apply_confidence_thresholds(decision), "llm")
    except Exception:  # noqa: BLE001
        decision = _fallback_route_query_with_context(
                query=query,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns or [],
                episodic_context=episodic_context or [],
        )
        return (_apply_confidence_thresholds(decision), "fallback")


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
    rewritten = query.strip()
    lowered = rewritten.strip().lower()
    if not lowered:
        return RouterDecision(
            action="CLARIFY",
            route_confidence=0.9,
            route_reason_public="The query is empty.",
            rewritten_query=query,
            clarifying_question="What question should I answer from the indexed arXiv corpus?",
            expected_next_node="clarify",
        )

    tool_name = _tool_route(lowered)
    if tool_name:
        tool_args: dict[str, str] = {"query": rewritten}
        if tool_name == "arxiv_lookup_by_id":
            tool_args = {"arxiv_ids": ",".join(_extract_arxiv_ids(rewritten))}
        return RouterDecision(
            action="TOOL",
            route_confidence=0.8,
            route_reason_public="The query asks for corpus/arXiv metadata.",
            rewritten_query=rewritten,
            tool_name=tool_name,
            tool_args=tool_args,
            expected_next_node="tool",
        )

    if _out_of_domain(lowered):
        return RouterDecision(
            action="REFUSE",
            route_confidence=0.8,
            route_reason_public="The query is outside the indexed arXiv corpus scope.",
            rewritten_query=rewritten,
            refusal_reason="I can only answer from the indexed recent arXiv cs.AI corpus.",
            expected_next_node="refuse",
        )

    if _ambiguous(lowered):
        return RouterDecision(
            action="CLARIFY",
            route_confidence=0.65,
            route_reason_public="The query is missing a concrete target or criterion.",
            rewritten_query=rewritten,
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
        rewritten_query=rewritten,
        expected_next_node="retrieve",
        retrieval_variant=variant,
        retrieval_filters=_infer_retrieval_filters(lowered),
    )


def _tool_route(query: str) -> str | None:
    if _has_explicit_arxiv_id(query):
        return "arxiv_lookup_by_id"
    arxiv_markers = ("latest paper", "find papers", "search arxiv", "search papers")
    if any(marker in query for marker in arxiv_markers):
        return "arxiv_search"
    return None


def _out_of_domain(query: str) -> bool:
    blocked = ("weather", "stock price", "sports score", "restaurant near me")
    return any(marker in query for marker in blocked)


def _ambiguous(query: str) -> bool:
    q = query.strip().lower()
    if q in {"which is better", "best method", "what about it"}:
        return True
    if q.startswith(("which is better", "what about")):
        return True
    # Underspecified comparative/superlative queries lacking explicit subject/criterion.
    if any(token in q for token in ("better", "best", "most effective", "most reliable")):
        has_target = any(
            token in q
            for token in (
                "agent",
                "rag",
                "retrieval",
                "memory",
                "planning",
                "reasoning",
                "safety",
                "benchmark",
                "paper",
                "method",
                "approach",
            )
        )
        if not has_target:
            return True
    # Very short follow-up deictic references are usually ambiguous without richer context.
    deictic = ("it", "that", "this", "those", "them")
    words = [w for w in q.replace("?", "").split() if w]
    if len(words) <= 4 and any(w in deictic for w in words):
        return True
    return False


def _requests_exact_quote(query: str) -> bool:
    return any(marker in query for marker in ("exact quote", "quote", "verbatim"))


def _infer_retrieval_filters(query: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    if "cs.ai" in query:
        filters["primary_category"] = "cs.AI"
    return filters


def _has_explicit_arxiv_id(query: str) -> bool:
    return bool(_extract_arxiv_ids(query))


def _extract_arxiv_ids(query: str) -> list[str]:
    import re

    ids = re.findall(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", query)
    return list(dict.fromkeys(ids))


def _intent_route_with_llm(
    *,
    query: str,
    llm_client: OpenAILLMClient,
    router_model: str,
    conversation_summary: str,
    recent_turns: list[dict[str, str]],
) -> RouterDecision | None:
    try:
        intent = llm_client.complete_json(
            model=router_model,
            schema=LLMIntentOutput,
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_prompt=intent_user_prompt(
                query=query,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns,
            ),
        )
    except Exception:  # noqa: BLE001
        return None
    if intent.confidence < INTENT_CONFIDENCE_THRESHOLD:
        return None
    lowered = query.strip().lower()
    if intent.intent == "ambiguous":
        return RouterDecision(
            action="CLARIFY",
            route_confidence=intent.confidence,
            route_reason_public=intent.reason or "The query is underspecified.",
            rewritten_query=query.strip(),
            clarifying_question="Could you specify the exact target or criterion you want?",
            expected_next_node="clarify",
        )
    if intent.intent == "refusal":
        return RouterDecision(
            action="REFUSE",
            route_confidence=intent.confidence,
            route_reason_public=intent.reason or "The query appears out of scope.",
            rewritten_query=query.strip(),
            refusal_reason="I can only answer from the indexed recent arXiv cs.AI corpus.",
            expected_next_node="refuse",
        )
    if intent.intent == "tool_arxiv":
        tool_name = _tool_route(lowered) or "arxiv_search"
        tool_args: dict[str, str] = {"query": query.strip()}
        if tool_name == "arxiv_lookup_by_id":
            tool_args = {"arxiv_ids": ",".join(_extract_arxiv_ids(query))}
        return RouterDecision(
            action="TOOL",
            route_confidence=intent.confidence,
            route_reason_public=intent.reason or "The query asks for arXiv metadata/search info.",
            rewritten_query=query.strip(),
            tool_name=tool_name,
            tool_args=tool_args,
            expected_next_node="tool",
        )
    if intent.intent in {"content_qa", "comparison", "follow_up", "general"}:
        return RouterDecision(
            action="RETRIEVE",
            route_confidence=intent.confidence,
            route_reason_public=(
                intent.reason or "The query appears answerable from indexed paper content."
            ),
            rewritten_query=query.strip(),
            retrieval_filters=_infer_retrieval_filters(lowered),
            expected_next_node="retrieve",
            retrieval_variant=(
                RetrievalVariant.child_only
                if _requests_exact_quote(lowered)
                else RetrievalVariant.parent_child
            ),
        )
    return None


def _apply_confidence_thresholds(decision: RouterDecision) -> RouterDecision:
    if decision.action == "RETRIEVE" and decision.route_confidence < RETRIEVE_CONFIDENCE_THRESHOLD:
        return RouterDecision(
            action="CLARIFY",
            route_confidence=decision.route_confidence,
            route_reason_public="Low routing confidence; clarification requested before retrieval.",
            rewritten_query=decision.rewritten_query,
            clarifying_question=(
                "Could you restate your question with a concrete target "
                "or comparison criterion?"
            ),
            expected_next_node="clarify",
        )
    if decision.action == "TOOL" and decision.route_confidence < TOOL_CONFIDENCE_THRESHOLD:
        return RouterDecision(
            action="CLARIFY",
            route_confidence=decision.route_confidence,
            route_reason_public="Low routing confidence; clarification requested before tool call.",
            rewritten_query=decision.rewritten_query,
            clarifying_question=(
                "Do you want arXiv metadata lookup by ID "
                "or a broader arXiv search query?"
            ),
            expected_next_node="clarify",
        )
    return decision
