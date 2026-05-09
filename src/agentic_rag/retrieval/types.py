from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RetrievalVariant(StrEnum):
    child_only = "child_only"
    parent_child = "parent_child"


class EvidenceStatus(StrEnum):
    sufficient = "SUFFICIENT"
    ambiguous = "AMBIGUOUS"
    insufficient = "INSUFFICIENT"
    contradictory = "CONTRADICTORY"


@dataclass(frozen=True)
class ChildHit:
    chunk_id: str
    parent_id: str
    paper_id: str
    section_path: str
    section_type: str
    content_type: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentGroup:
    parent_id: str
    paper_id: str
    section_path: str
    section_type: str
    parent_score: float
    max_child_score: float
    support_count: int
    evidence_child_ids: list[str]
    child_hits: list[ChildHit]


@dataclass(frozen=True)
class RetrievalSignals:
    top_child_score: float
    top_parent_score: float
    second_parent_score: float
    score_margin: float
    supporting_child_count: int
    distinct_parent_count: int
    distinct_paper_count: int
    section_type_distribution: dict[str, int]
    evidence_status: EvidenceStatus
    confidence_band: str
    thresholds_used: dict[str, float]


@dataclass(frozen=True)
class ContextPacket:
    source_id: str
    paper_id: str
    title: str
    arxiv_id: str
    section_path: str
    page_start: int | None
    page_end: int | None
    content_type: str
    parent_score: float
    evidence_child_ids: list[str]
    highlighted_evidence: str
    section_context: str


@dataclass(frozen=True)
class RetrievalResult:
    variant: RetrievalVariant
    selected_parent_ids: list[str]
    retrieved_child_ids: list[str]
    evidence_child_ids: list[str]
    parent_scores: dict[str, float]
    context_packets: list[ContextPacket]
    signals: RetrievalSignals
