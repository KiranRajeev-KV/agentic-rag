from __future__ import annotations

from typing import Any, Literal, TypedDict

from agentic_rag.retrieval.types import EvidenceStatus, RetrievalVariant

RouteAction = Literal[
    "CLARIFY",
    "RETRIEVE",
    "TOOL",
    "REFUSE",
    "ANSWER_FROM_MEMORY",
    "ANSWER_FROM_CONTEXT",
]


class AgentState(TypedDict, total=False):
    thread_id: str
    turn_id: str
    trace_id: str
    raw_user_query: str
    normalized_query: str

    route_action: RouteAction
    route_confidence: float
    route_reason_public: str
    rewritten_query: str
    retrieval_variant: RetrievalVariant
    forced_retrieval_variant: RetrievalVariant
    retrieval_filters: dict[str, Any]
    tool_name: str
    tool_args: dict[str, Any]
    clarifying_question: str
    refusal_reason: str
    expected_next_node: str

    retrieved_child_ids: list[str]
    selected_parent_ids: list[str]
    parent_scores: dict[str, float]
    evidence_child_ids: list[str]
    context_packets: list[dict[str, Any]]

    evidence_status: EvidenceStatus
    evidence_confidence: str
    evidence_signals: dict[str, Any]

    final_action: str
    final_answer: str
    citations: list[str]
    sources_block: str

    memory_context: list[dict[str, Any]]
    memory_writes_pending: list[dict[str, Any]]
    tool_result: dict[str, Any]
