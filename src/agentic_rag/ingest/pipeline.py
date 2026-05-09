from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.corpus.discovery import discover_relevant_papers
from agentic_rag.corpus.download import download_pdf
from agentic_rag.ingest.chunking import build_parent_child_rows
from agentic_rag.ingest.parser import DoclingParser
from agentic_rag.logging import write_jsonl_event
from agentic_rag.storage.repositories import (
    ChunkRepository,
    PaperRecord,
    PaperRepository,
    ParentRepository,
)
from agentic_rag.storage.sqlite import SQLiteStore
from agentic_rag.tools.arxiv_tools import ArxivToolset


@dataclass(frozen=True)
class IngestSummary:
    requested_limit: int
    discovered: int
    downloaded: int
    parsed: int
    parse_failed: int
    download_failed: int
    skipped: int
    errors: list[str]


class IngestPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tools = ArxivToolset(settings=settings)
        self.parser = DoclingParser()
        self.store = SQLiteStore(settings.app_db_path)
        self.paper_repo = PaperRepository(self.store)
        self.parent_repo = ParentRepository(self.store)
        self.chunk_repo = ChunkRepository(self.store)

    def run(
        self, limit: int, days_back: int = 90, query_filter: str | None = None
    ) -> IngestSummary:
        discovered, discovery_errors = discover_relevant_papers(
            toolset=self.tools,
            limit=limit,
            days_back=days_back,
            query_filter=query_filter,
        )

        downloaded = 0
        parsed = 0
        parse_failed = 0
        download_failed = 0
        skipped = 0
        errors = list(discovery_errors)

        for item in discovered:
            paper = item.metadata
            paper_id = _paper_id(paper.arxiv_id)

            self.paper_repo.upsert(
                PaperRecord(
                    paper_id=paper_id,
                    arxiv_id=paper.arxiv_id,
                    arxiv_version=paper.version,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract or "",
                    primary_category=paper.primary_category,
                    categories=paper.categories,
                    published_at=_iso(paper.published_at),
                    updated_at=_iso(paper.updated_at),
                    pdf_url=paper.pdf_url,
                    abs_url=paper.abs_url,
                    doi=paper.doi,
                    comment=paper.comment,
                    source_query=item.source_query,
                    parse_status="discovered",
                )
            )

            download = download_pdf(
                paper=paper,
                pdf_dir=self.settings.app_pdf_dir,
                user_agent=self.settings.arxiv_user_agent,
            )
            if not download.ok or not download.paper_path:
                download_failed += 1
                self.paper_repo.upsert(
                    PaperRecord(
                        paper_id=paper_id,
                        arxiv_id=paper.arxiv_id,
                        arxiv_version=paper.version,
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract or "",
                        primary_category=paper.primary_category,
                        categories=paper.categories,
                        published_at=_iso(paper.published_at),
                        updated_at=_iso(paper.updated_at),
                        pdf_url=paper.pdf_url,
                        abs_url=paper.abs_url,
                        doi=paper.doi,
                        comment=paper.comment,
                        source_query=item.source_query,
                        parse_status="download_failed",
                    )
                )
                if download.error:
                    errors.append(f"{paper.arxiv_id}: {download.error}")
                continue

            downloaded += 1
            parse_result = self.parser.parse_pdf(download.paper_path)
            if not parse_result.ok or parse_result.conversion is None:
                parse_failed += 1
                write_jsonl_event(
                    self.settings.app_log_jsonl,
                    {
                        "level": "warning",
                        "event": "parse.failed",
                        "payload": {
                            "paper_id": paper_id,
                            "arxiv_id": paper.arxiv_id,
                            "status": parse_result.status,
                            "error": parse_result.error,
                        },
                    },
                )
                self.paper_repo.upsert(
                    PaperRecord(
                        paper_id=paper_id,
                        arxiv_id=paper.arxiv_id,
                        arxiv_version=paper.version,
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract or "",
                        primary_category=paper.primary_category,
                        categories=paper.categories,
                        published_at=_iso(paper.published_at),
                        updated_at=_iso(paper.updated_at),
                        pdf_url=paper.pdf_url,
                        abs_url=paper.abs_url,
                        doi=paper.doi,
                        comment=paper.comment,
                        source_query=item.source_query,
                        pdf_sha256=download.pdf_sha256,
                        parse_status="parse_failed",
                        parser_name=self.parser.parser_name,
                    )
                )
                if parse_result.error:
                    errors.append(f"{paper.arxiv_id}: {parse_result.error}")
                continue

            chunked = build_parent_child_rows(paper_id=paper_id, conversion=parse_result.conversion)
            if not chunked.parent_rows or not chunked.chunk_rows:
                skipped += 1
                self.paper_repo.upsert(
                    PaperRecord(
                        paper_id=paper_id,
                        arxiv_id=paper.arxiv_id,
                        arxiv_version=paper.version,
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract or "",
                        primary_category=paper.primary_category,
                        categories=paper.categories,
                        published_at=_iso(paper.published_at),
                        updated_at=_iso(paper.updated_at),
                        pdf_url=paper.pdf_url,
                        abs_url=paper.abs_url,
                        doi=paper.doi,
                        comment=paper.comment,
                        source_query=item.source_query,
                        pdf_sha256=download.pdf_sha256,
                        parse_status="parse_empty",
                        parser_name=self.parser.parser_name,
                        chunker_name="docling_hybrid_chunker",
                    )
                )
                continue

            self.parent_repo.insert_many(chunked.parent_rows)
            self.chunk_repo.insert_many(chunked.chunk_rows)
            self.paper_repo.upsert(
                PaperRecord(
                    paper_id=paper_id,
                    arxiv_id=paper.arxiv_id,
                    arxiv_version=paper.version,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract or "",
                    primary_category=paper.primary_category,
                    categories=paper.categories,
                    published_at=_iso(paper.published_at),
                    updated_at=_iso(paper.updated_at),
                    pdf_url=paper.pdf_url,
                    abs_url=paper.abs_url,
                    doi=paper.doi,
                    comment=paper.comment,
                    source_query=item.source_query,
                    pdf_sha256=download.pdf_sha256,
                    parse_status="parsed",
                    parser_name=self.parser.parser_name,
                    chunker_name="docling_hybrid_chunker",
                    chunker_config_hash="default",
                )
            )
            parsed += 1

        return IngestSummary(
            requested_limit=limit,
            discovered=len(discovered),
            downloaded=downloaded,
            parsed=parsed,
            parse_failed=parse_failed,
            download_failed=download_failed,
            skipped=skipped,
            errors=errors,
        )


def _paper_id(arxiv_id: str) -> str:
    normalized = arxiv_id.strip().lower()
    return f"paper_{normalized.replace('/', '_')}"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
