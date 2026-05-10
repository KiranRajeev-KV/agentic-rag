from __future__ import annotations

from collections import Counter

from .types import ChildHit, EvidenceStatus, ParentGroup, RetrievalSignals


def evaluate_evidence(
    parent_groups: list[ParentGroup],
    child_hits: list[ChildHit],
    thresholds: dict[str, float] | None = None,
) -> RetrievalSignals:
    params = {
        "min_parent_score": 0.42,
        "min_margin": 0.01,
        "min_supporting_children": 1.0,
        "min_total_supporting_children": 3.0,
        "min_distinct_papers": 1.0,
    }
    if thresholds:
        params.update(thresholds)

    top_child_score = child_hits[0].score if child_hits else 0.0
    top_parent_score = parent_groups[0].parent_score if parent_groups else 0.0
    second_parent_score = parent_groups[1].parent_score if len(parent_groups) > 1 else 0.0
    score_margin = max(top_parent_score - second_parent_score, 0.0)
    supporting_child_count = len(parent_groups[0].evidence_child_ids) if parent_groups else 0
    total_supporting_children = _total_supporting_children(parent_groups)
    distinct_parent_count = len(parent_groups)
    distinct_paper_count = len({group.paper_id for group in parent_groups})
    section_type_distribution = dict(Counter(group.section_type for group in parent_groups))

    evidence_status = _status_from_signals(
        top_parent_score=top_parent_score,
        score_margin=score_margin,
        supporting_child_count=supporting_child_count,
        total_supporting_children=total_supporting_children,
        distinct_paper_count=distinct_paper_count,
        thresholds=params,
    )
    confidence_band = _confidence_band(
        evidence_status=evidence_status,
        top_parent_score=top_parent_score,
        score_margin=score_margin,
    )

    return RetrievalSignals(
        top_child_score=top_child_score,
        top_parent_score=top_parent_score,
        second_parent_score=second_parent_score,
        score_margin=score_margin,
        supporting_child_count=supporting_child_count,
        total_supporting_children=total_supporting_children,
        distinct_parent_count=distinct_parent_count,
        distinct_paper_count=distinct_paper_count,
        section_type_distribution=section_type_distribution,
        evidence_status=evidence_status,
        confidence_band=confidence_band,
        thresholds_used=params,
    )


def _status_from_signals(
    top_parent_score: float,
    score_margin: float,
    supporting_child_count: int,
    total_supporting_children: int,
    distinct_paper_count: int,
    thresholds: dict[str, float],
) -> EvidenceStatus:
    if supporting_child_count == 0:
        return EvidenceStatus.insufficient
    if top_parent_score < thresholds["min_parent_score"]:
        return EvidenceStatus.insufficient
    if supporting_child_count < thresholds["min_supporting_children"]:
        return EvidenceStatus.ambiguous
    if (
        score_margin < thresholds["min_margin"]
        and total_supporting_children < thresholds["min_total_supporting_children"]
    ):
        return EvidenceStatus.ambiguous
    if distinct_paper_count < thresholds["min_distinct_papers"]:
        return EvidenceStatus.ambiguous
    return EvidenceStatus.sufficient


def _confidence_band(
    evidence_status: EvidenceStatus,
    top_parent_score: float,
    score_margin: float,
) -> str:
    if (
        evidence_status == EvidenceStatus.sufficient
        and top_parent_score >= 0.6
        and score_margin >= 0.05
    ):
        return "HIGH"
    if evidence_status in {EvidenceStatus.sufficient, EvidenceStatus.ambiguous}:
        return "MEDIUM"
    return "LOW"


def _total_supporting_children(parent_groups: list[ParentGroup]) -> int:
    if not parent_groups:
        return 0
    return sum(len(group.evidence_child_ids) for group in parent_groups[:3])
