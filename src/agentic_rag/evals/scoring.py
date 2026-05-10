from __future__ import annotations

from agentic_rag.agent.state import AgentState

from .models import EvalCase, EvalCaseResult


def score_case(case: EvalCase, state: AgentState) -> EvalCaseResult:
    route = str(state.get("route_action", ""))
    final_action = str(state.get("final_action", ""))
    answer = str(state.get("final_answer", ""))
    citations = _extract_citation_ids(answer)
    trace_id = str(state.get("trace_id") or "") or None
    thread_id = str(state.get("thread_id") or "") or None

    route_score = 2.0 if route == case.expected_route else 0.0
    answer_score = 2.0 if final_action == case.expected_final_action else 0.0
    evidence_score = 2.0 if _behavior_ok(case, final_action) else 0.0
    citation_score = 2.0 if _citation_ok(case, citations) else 0.0
    retrieval_score = _retrieval_score(case=case, state=state)
    tool_score = _tool_score(case=case, state=state)
    memory_score = 0.0
    trace_score = 0.0

    total = (
        route_score
        + answer_score
        + evidence_score
        + citation_score
        + retrieval_score
        + tool_score
        + memory_score
        + trace_score
    )
    notes = ""
    if case.should_refuse and final_action != "REFUSE":
        notes = "expected_refusal_not_met"
    elif case.should_clarify and final_action != "CLARIFY":
        notes = "expected_clarify_not_met"

    return EvalCaseResult(
        case_id=case.id,
        intent=case.intent,
        trace_id=trace_id,
        thread_id=thread_id,
        score=total,
        route_score=route_score,
        retrieval_score=retrieval_score,
        evidence_score=evidence_score,
        answer_score=answer_score,
        citation_score=citation_score,
        memory_score=memory_score,
        tool_score=tool_score,
        trace_score=trace_score,
        expected_route=case.expected_route,
        actual_route=route,
        expected_final_action=case.expected_final_action,
        actual_final_action=final_action,
        notes=notes,
    )


def _extract_citation_ids(answer: str) -> set[str]:
    ids: set[str] = set()
    token = ""
    open_bracket = False
    for char in answer:
        if char == "[":
            token = ""
            open_bracket = True
            continue
        if char == "]" and open_bracket:
            open_bracket = False
            if token.startswith(("S", "T")):
                ids.add(token)
            continue
        if open_bracket:
            token += char
    return ids


def _behavior_ok(case: EvalCase, final_action: str) -> bool:
    if case.should_refuse:
        return final_action == "REFUSE"
    if case.should_clarify:
        return final_action == "CLARIFY"
    return final_action in {"ANSWER_FROM_CONTEXT", "ANSWER_FROM_TOOL"}


def _citation_ok(case: EvalCase, citations: set[str]) -> bool:
    if case.should_cite_sources:
        return any(item.startswith("S") for item in citations)
    return True


def _tool_score(case: EvalCase, state: AgentState) -> float:
    if not case.expected_tool:
        return 1.0
    tool_name = str(state.get("tool_name", ""))
    final_action = str(state.get("final_action", ""))
    if tool_name == case.expected_tool and final_action == "ANSWER_FROM_TOOL":
        return 1.0
    return 0.0


def _retrieval_score(case: EvalCase, state: AgentState) -> float:
    route = str(state.get("route_action", ""))
    if case.expected_route == "TOOL":
        return 1.0 if route == "TOOL" else 0.0
    if case.expected_route in {"CLARIFY", "REFUSE"}:
        return 1.0
    return 1.0 if len(state.get("selected_parent_ids", [])) > 0 else 0.0
