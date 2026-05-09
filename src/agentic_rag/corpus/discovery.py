from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.corpus.filters import matched_filter_details
from agentic_rag.tools.schemas import ArxivGetRecentInput, ArxivPaperMetadata, ToolStatus


@dataclass(frozen=True)
class DiscoveredPaper:
    metadata: ArxivPaperMetadata
    source_query: str
    filter_terms: list[str]
    filter_reason: str
    days_back: int


def discover_relevant_papers(
    toolset: object,
    limit: int,
    days_back: int = 90,
    query_filter: str | None = None,
) -> tuple[list[DiscoveredPaper], list[str]]:
    payload = ArxivGetRecentInput(
        category="cs.AI",
        days_back=days_back,
        max_results=max(limit * 4, limit),
        query_filter=query_filter,
    )
    output = toolset.arxiv_get_recent(payload)
    if output.status == ToolStatus.error:
        return [], output.errors

    by_id: dict[str, DiscoveredPaper] = {}
    source_query = query_filter or "cs.AI recent papers"
    for paper in output.papers:
        abstract = paper.abstract or ""
        details = matched_filter_details(title=paper.title, abstract=abstract)
        if not details["terms"]:
            continue
        normalized_id = paper.arxiv_id.strip().lower()
        if normalized_id not in by_id:
            reason = (
                f"matched_terms={','.join(details['terms'])};"
                f"title_terms={','.join(details['title_terms'])};"
                f"abstract_terms={','.join(details['abstract_terms'])};"
                f"query_filter={query_filter or ''}"
            )
            by_id[normalized_id] = DiscoveredPaper(
                metadata=paper,
                source_query=source_query,
                filter_terms=details["terms"],
                filter_reason=reason,
                days_back=days_back,
            )

    return list(by_id.values())[:limit], []
