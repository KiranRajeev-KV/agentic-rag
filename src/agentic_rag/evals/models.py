from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    conversation_history: list[str] = Field(default_factory=list)
    expected_route: str
    expected_final_action: str
    expected_tool: str | None = None
    expected_paper_ids: list[str] = Field(default_factory=list)
    expected_parent_ids: list[str] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)
    should_cite_sources: bool = True
    should_refuse: bool = False
    should_clarify: bool = False
    notes: str = ""


class EvalCaseResult(BaseModel):
    case_id: str
    score: float
    route_score: float
    retrieval_score: float
    evidence_score: float
    answer_score: float
    citation_score: float
    memory_score: float
    tool_score: float
    trace_score: float
    expected_route: str
    actual_route: str
    expected_final_action: str
    actual_final_action: str
    notes: str = ""
