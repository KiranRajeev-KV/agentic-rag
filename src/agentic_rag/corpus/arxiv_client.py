from __future__ import annotations

from datetime import date

import arxiv

from agentic_rag.tools.schemas import ArxivPaperMetadata, ArxivSortBy, ArxivSortOrder

_SORT_BY_MAP: dict[ArxivSortBy, arxiv.SortCriterion] = {
    ArxivSortBy.relevance: arxiv.SortCriterion.Relevance,
    ArxivSortBy.submitted_date: arxiv.SortCriterion.SubmittedDate,
    ArxivSortBy.last_updated_date: arxiv.SortCriterion.LastUpdatedDate,
}

_SORT_ORDER_MAP: dict[ArxivSortOrder, arxiv.SortOrder] = {
    ArxivSortOrder.ascending: arxiv.SortOrder.Ascending,
    ArxivSortOrder.descending: arxiv.SortOrder.Descending,
}


class ArxivApiClient:
    def __init__(self, user_agent: str, page_size: int = 100, delay_seconds: float = 3.0) -> None:
        self.user_agent = user_agent
        self.client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds, num_retries=3)
        self.client._session.headers.update({"User-Agent": user_agent})  # noqa: SLF001

    def lookup_by_ids(self, arxiv_ids: list[str]) -> list[ArxivPaperMetadata]:
        query = arxiv.Search(id_list=arxiv_ids, max_results=len(arxiv_ids))
        return [self._to_metadata(result) for result in self.client.results(query)]

    def search(
        self,
        query: str,
        categories: list[str],
        max_results: int,
        sort_by: ArxivSortBy,
        sort_order: ArxivSortOrder,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ArxivPaperMetadata]:
        full_query = _build_query(
            query=query, categories=categories, date_from=date_from, date_to=date_to
        )
        search_query = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=_SORT_BY_MAP[sort_by],
            sort_order=_SORT_ORDER_MAP[sort_order],
        )
        return [self._to_metadata(result) for result in self.client.results(search_query)]

    @staticmethod
    def _to_metadata(result: arxiv.Result) -> ArxivPaperMetadata:
        short_id = result.get_short_id()
        arxiv_id, version = _split_short_id(short_id)
        return ArxivPaperMetadata(
            arxiv_id=arxiv_id,
            version=version,
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            categories=result.categories,
            primary_category=result.primary_category,
            published_at=result.published,
            updated_at=result.updated,
            abs_url=result.entry_id,
            pdf_url=result.pdf_url,
            doi=result.doi,
            comment=result.comment,
        )


def _split_short_id(short_id: str) -> tuple[str, str | None]:
    parts = short_id.rsplit("v", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], f"v{parts[1]}"
    return short_id, None


def _build_query(
    query: str,
    categories: list[str],
    date_from: date | None,
    date_to: date | None,
) -> str:
    cleaned = _normalize_query_text(query)
    query_term = ""
    if cleaned and cleaned != "*":
        query_term = f"all:{cleaned}"
    cat_clause = " OR ".join(f"cat:{category}" for category in categories) if categories else ""
    date_clause = _date_range_clause(date_from=date_from, date_to=date_to)

    clauses: list[str] = []
    if query_term:
        clauses.append(query_term)
    if cat_clause:
        clauses.append(f"({cat_clause})")
    if date_clause:
        clauses.append(date_clause)
    if not clauses:
        return "cat:cs.AI"
    return " AND ".join(clauses)


def _normalize_query_text(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    lowered = cleaned.lower()
    noise_prefixes = (
        "search arxiv for ",
        "search for ",
        "find papers on ",
        "find papers about ",
    )
    for prefix in noise_prefixes:
        if lowered.startswith(prefix):
            return cleaned[len(prefix) :].strip()
    return cleaned


def _date_range_clause(date_from: date | None, date_to: date | None) -> str:
    if date_from is None and date_to is None:
        return ""
    start = (date_from or date.min).strftime("%Y%m%d0000")
    end = (date_to or date.max).strftime("%Y%m%d2359")
    return f"submittedDate:[{start} TO {end}]"
