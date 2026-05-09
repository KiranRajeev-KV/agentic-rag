from __future__ import annotations

from collections import defaultdict

from agentic_rag.storage.repositories import ChunkRepository, PaperRepository, ParentRepository

from .types import ContextPacket, ParentGroup


def assemble_context_packets(
    parent_groups: list[ParentGroup],
    paper_repo: PaperRepository,
    parent_repo: ParentRepository,
    chunk_repo: ChunkRepository,
    max_parents: int,
) -> list[ContextPacket]:
    selected = parent_groups[:max_parents]
    parent_rows = parent_repo.get_by_ids([group.parent_id for group in selected])
    paper_rows = paper_repo.get_by_ids([group.paper_id for group in selected])

    chunk_ids: list[str] = []
    for group in selected:
        chunk_ids.extend(group.evidence_child_ids)
    chunk_rows = chunk_repo.get_by_ids(chunk_ids)

    packets: list[ContextPacket] = []
    for idx, group in enumerate(selected, start=1):
        parent = parent_rows.get(group.parent_id, {})
        paper = paper_rows.get(group.paper_id, {})
        highlighted = _highlighted_text(group.evidence_child_ids, chunk_rows)
        section_context = _trimmed_parent_text(parent.get("parent_text", ""), highlighted)

        packets.append(
            ContextPacket(
                source_id=f"S{idx}",
                paper_id=group.paper_id,
                title=str(paper.get("title", "")),
                arxiv_id=str(paper.get("arxiv_id", "")),
                section_path=str(parent.get("section_path", group.section_path)),
                page_start=parent.get("page_start"),
                page_end=parent.get("page_end"),
                content_type=_resolve_content_type(group),
                parent_score=group.parent_score,
                evidence_child_ids=group.evidence_child_ids,
                highlighted_evidence=highlighted,
                section_context=section_context,
            )
        )
    return packets


def source_block(packet: ContextPacket) -> str:
    page_span = (
        f"{packet.page_start}-{packet.page_end}"
        if packet.page_start is not None and packet.page_end is not None
        else "unknown"
    )
    return (
        f'<SOURCE id="{packet.source_id}">\n'
        f"paper_id: {packet.paper_id}\n"
        f"title: {packet.title}\n"
        f"arxiv_id: {packet.arxiv_id}\n"
        f"section_path: {packet.section_path}\n"
        f"page_span: {page_span}\n"
        f"content_type: {packet.content_type}\n"
        f"parent_score: {packet.parent_score:.4f}\n"
        f"evidence_child_ids: {packet.evidence_child_ids}\n\n"
        "HIGHLIGHTED_EVIDENCE:\n"
        f"{packet.highlighted_evidence}\n\n"
        "SECTION_CONTEXT:\n"
        f"{packet.section_context}\n"
        "</SOURCE>"
    )


def dynamic_parent_budget(query: str, mode: str = "auto") -> int:
    if mode != "auto":
        mode_map = {
            "specific": 2,
            "single_paper": 3,
            "comparison": 5,
            "synthesis": 6,
            "metadata": 0,
        }
        return mode_map.get(mode, 4)

    lowered = query.lower()
    if any(term in lowered for term in ("compare", "versus", "vs", "difference", "tradeoff")):
        return 5
    if any(term in lowered for term in ("survey", "overall", "across papers", "synthesis")):
        return 6
    if any(term in lowered for term in ("title", "author", "published", "arxiv id")):
        return 0
    return 3


def _highlighted_text(chunk_ids: list[str], chunk_rows: dict[str, dict[str, object]]) -> str:
    lines: list[str] = []
    for chunk_id in chunk_ids:
        row = chunk_rows.get(chunk_id)
        if not row:
            continue
        text = str(row.get("chunk_text", "")).strip()
        if not text:
            continue
        lines.append(f"[{chunk_id}] {text[:700]}")
    return "\n".join(lines)


def _trimmed_parent_text(parent_text: str, highlighted: str) -> str:
    if not parent_text:
        return highlighted[:1200]
    if len(parent_text) <= 2400:
        return parent_text
    if highlighted and highlighted in parent_text:
        start = max(parent_text.find(highlighted[:60]) - 300, 0)
        end = min(start + 2200, len(parent_text))
        return parent_text[start:end]
    return parent_text[:2200]


def _resolve_content_type(group: ParentGroup) -> str:
    counts: dict[str, int] = defaultdict(int)
    for hit in group.child_hits:
        counts[hit.content_type] += 1
    if not counts:
        return "text"
    return max(counts, key=counts.get)
