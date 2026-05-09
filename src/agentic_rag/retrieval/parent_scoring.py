from __future__ import annotations

from collections import defaultdict

from .types import ChildHit, ParentGroup


def score_parent_groups(child_hits: list[ChildHit], top_m: int = 6) -> list[ParentGroup]:
    grouped: dict[str, list[ChildHit]] = defaultdict(list)
    for hit in child_hits:
        grouped[hit.parent_id].append(hit)

    parent_groups: list[ParentGroup] = []
    for parent_id, hits in grouped.items():
        sorted_hits = sorted(hits, key=lambda item: item.score, reverse=True)
        max_child_score = sorted_hits[0].score
        support_count = len(sorted_hits)
        support_bonus = 0.03 * min(max(support_count - 1, 0), 3)
        section_type = sorted_hits[0].section_type
        section_adjustment = _section_type_adjustment(section_type)
        references_penalty = 0.2 if section_type == "references" else 0.0
        parent_score = max_child_score + support_bonus + section_adjustment - references_penalty

        parent_groups.append(
            ParentGroup(
                parent_id=parent_id,
                paper_id=sorted_hits[0].paper_id,
                section_path=sorted_hits[0].section_path,
                section_type=section_type,
                parent_score=parent_score,
                max_child_score=max_child_score,
                support_count=support_count,
                evidence_child_ids=[hit.chunk_id for hit in sorted_hits[:4]],
                child_hits=sorted_hits,
            )
        )

    parent_groups.sort(key=lambda item: item.parent_score, reverse=True)
    return parent_groups[:top_m]


def _section_type_adjustment(section_type: str) -> float:
    normalized = section_type.lower()
    if normalized == "references":
        return -0.05
    if normalized in {"conclusion", "limitations"}:
        return -0.01
    if normalized in {"results", "method", "methods"}:
        return 0.02
    return 0.0
