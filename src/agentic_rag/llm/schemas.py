from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMRouterOutput(BaseModel):
    action: Literal[
        "CLARIFY",
        "RETRIEVE",
        "TOOL",
        "REFUSE",
        "ANSWER_FROM_MEMORY",
        "ANSWER_FROM_CONTEXT",
    ]
    route_confidence: float = Field(ge=0.0, le=1.0)
    route_reason_public: str
    rewritten_query: str
    retrieval_filters: dict[str, str] = Field(default_factory=dict)
    tool_name: str | None = None
    tool_args: dict[str, str] = Field(default_factory=dict)
    clarifying_question: str | None = None
    refusal_reason: str | None = None
    expected_next_node: str
    retrieval_variant: Literal["child_only", "parent_child"] = "parent_child"


class LLMEvidenceOutput(BaseModel):
    evidence_status: Literal["SUFFICIENT", "AMBIGUOUS", "INSUFFICIENT", "CONTRADICTORY"]
    conflict_label: Literal[
        "NONE", "DIFFERENCE", "TENSION", "DIRECT_CONTRADICTION", "METADATA_CONFLICT"
    ] = "NONE"
    recommended_action: Literal["ANSWER_FROM_CONTEXT", "CLARIFY", "REFUSE"] = "REFUSE"
    confidence_band: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    missing_info: str = ""
    contradiction_notes: str = ""


class LLMAnswerOutput(BaseModel):
    final_action: Literal["ANSWER_FROM_CONTEXT", "CLARIFY", "REFUSE"]
    answer_text: str = ""
    clarifying_question: str = ""
    refusal_reason: str = ""
    cited_source_ids: list[str] = Field(default_factory=list)
    cited_tool_ids: list[str] = Field(default_factory=list)


class LLMMemoryWriteOutput(BaseModel):
    should_write: bool = False
    kind: Literal["decision", "constraint", "preference", "entity", "open_question"] = "decision"
    key: str = ""
    value: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LLMCitationValidationOutput(BaseModel):
    valid: bool
    verdict: Literal[
        "VALID",
        "MISSING_CITATION",
        "UNKNOWN_CITATION",
        "UNSUPPORTED_CLAIM",
        "SOURCE_BLOCK_MISMATCH",
        "INTERNAL_ID_LEAK",
        "TOOL_CITATION_ERROR",
    ]
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_citation_spans: list[str] = Field(default_factory=list)
    unknown_citation_ids: list[str] = Field(default_factory=list)
    internal_id_leaks: list[str] = Field(default_factory=list)
    repair_instruction: str = ""


class LLMContradictionOutput(BaseModel):
    conflict_label: Literal[
        "NONE", "DIFFERENCE", "TENSION", "DIRECT_CONTRADICTION", "METADATA_CONFLICT"
    ]
    recommended_action: Literal["ANSWER_WITH_CONFLICT", "CLARIFY", "REFUSE", "TOOL_LOOKUP"]
    summary: str
    side_a_source_ids: list[str] = Field(default_factory=list)
    side_b_source_ids: list[str] = Field(default_factory=list)
    metadata_ids_to_check: list[str] = Field(default_factory=list)
    missing_info: str = ""
    confidence_band: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
