from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.corpus.discovery import discover_relevant_papers
from agentic_rag.corpus.download import download_pdf
from agentic_rag.ingest.chunking import (
    build_parent_child_rows,
    current_chunker_config_hash,
    current_chunker_name,
)
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


@dataclass(frozen=True)
class IngestConfig:
    parser_name: str
    parser_version: str
    chunker_name: str
    chunker_config_hash: str


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
        self,
        limit: int,
        days_back: int = 90,
        query_filter: str | None = None,
        force: bool = False,
    ) -> IngestSummary:
        ingest_config = _current_ingest_config(parser_name=self.parser.parser_name)
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
        total = len(discovered)

        for idx, item in enumerate(discovered, start=1):
            paper = item.metadata
            paper_id = _paper_id(paper.arxiv_id)
            existing_state = self.paper_repo.get_ingest_state(paper_id)
            existing_status = existing_state["parse_status"] if existing_state else None
            _emit_ingest_progress(
                settings=self.settings,
                event="paper.start",
                idx=idx,
                total=total,
                paper_id=paper_id,
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                status=existing_status or "new",
            )
            skip_reason = self._skip_reason(
                paper=paper,
                existing_state=existing_state,
                ingest_config=ingest_config,
                force=force,
            )
            if skip_reason is not None:
                skipped += 1
                _emit_ingest_progress(
                    settings=self.settings,
                    event="paper.skipped",
                    idx=idx,
                    total=total,
                    paper_id=paper_id,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    status="skipped",
                    reason=skip_reason,
                )
                continue

            download = download_pdf(
                paper=paper,
                pdf_dir=self.settings.app_pdf_dir,
                user_agent=self.settings.arxiv_user_agent,
            )
            if not download.ok or not download.paper_path:
                download_failed += 1
                _emit_ingest_warning(
                    settings=self.settings,
                    event="download.failed",
                    idx=idx,
                    total=total,
                    paper_id=paper_id,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    status="download_failed",
                    error=download.error,
                )
                if existing_status != "parsed":
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
                _emit_ingest_warning(
                    settings=self.settings,
                    event="parse.failed",
                    idx=idx,
                    total=total,
                    paper_id=paper_id,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    status=parse_result.status,
                    error=parse_result.error,
                )
                if existing_status != "parsed":
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
                            parser_name=ingest_config.parser_name,
                            parser_version=ingest_config.parser_version,
                            chunker_name=ingest_config.chunker_name,
                            chunker_config_hash=ingest_config.chunker_config_hash,
                        )
                    )
                if parse_result.error:
                    errors.append(f"{paper.arxiv_id}: {parse_result.error}")
                continue

            chunked = build_parent_child_rows(paper_id=paper_id, conversion=parse_result.conversion)
            if not chunked.parent_rows or not chunked.chunk_rows:
                skipped += 1
                _emit_ingest_warning(
                    settings=self.settings,
                    event="parse.empty",
                    idx=idx,
                    total=total,
                    paper_id=paper_id,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    status="parse_empty",
                    error="No parent/child rows produced by chunker",
                )
                if existing_status != "parsed":
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
                            parser_name=ingest_config.parser_name,
                            parser_version=ingest_config.parser_version,
                            chunker_name=ingest_config.chunker_name,
                            chunker_config_hash=ingest_config.chunker_config_hash,
                        )
                    )
                continue

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
                    parser_name=ingest_config.parser_name,
                    parser_version=ingest_config.parser_version,
                    chunker_name=ingest_config.chunker_name,
                    chunker_config_hash=ingest_config.chunker_config_hash,
                )
            )
            self.chunk_repo.delete_for_paper(paper_id)
            self.parent_repo.delete_for_paper(paper_id)
            self.parent_repo.insert_many(chunked.parent_rows)
            self.chunk_repo.insert_many(chunked.chunk_rows)
            _emit_ingest_progress(
                settings=self.settings,
                event="paper.parsed",
                idx=idx,
                total=total,
                paper_id=paper_id,
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                status="parsed",
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

    def _skip_reason(
        self,
        *,
        paper: Any,
        existing_state: dict[str, Any] | None,
        ingest_config: IngestConfig,
        force: bool,
    ) -> str | None:
        if force or not existing_state:
            return None
        if str(existing_state.get("parse_status") or "") != "parsed":
            return None
        if str(existing_state.get("arxiv_version") or "") != str(paper.version or ""):
            return None
        if str(existing_state.get("parser_name") or "") != ingest_config.parser_name:
            return None
        if str(existing_state.get("parser_version") or "") != ingest_config.parser_version:
            return None
        if str(existing_state.get("chunker_name") or "") != ingest_config.chunker_name:
            return None
        if (
            str(existing_state.get("chunker_config_hash") or "")
            != ingest_config.chunker_config_hash
        ):
            return None
        stored_pdf_sha = str(existing_state.get("pdf_sha256") or "")
        if not stored_pdf_sha:
            return None

        cached_pdf_path = _cached_pdf_path(
            pdf_dir=self.settings.app_pdf_dir,
            arxiv_id=paper.arxiv_id,
            version=paper.version,
        )
        if not cached_pdf_path.exists():
            return None
        local_sha = _sha256_file(cached_pdf_path)
        if local_sha != stored_pdf_sha:
            return None
        return "already_parsed_unchanged"


def _paper_id(arxiv_id: str) -> str:
    normalized = arxiv_id.strip().lower()
    return f"paper_{normalized.replace('/', '_')}"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _current_ingest_config(parser_name: str) -> IngestConfig:
    return IngestConfig(
        parser_name=parser_name,
        parser_version=_parser_version(),
        chunker_name=current_chunker_name(),
        chunker_config_hash=current_chunker_config_hash(),
    )


def _parser_version() -> str:
    try:
        return package_version("docling")
    except PackageNotFoundError:
        return "unknown"


def _cached_pdf_path(pdf_dir: Path, arxiv_id: str, version: str | None) -> Path:
    file_name = f"{arxiv_id}{version or ''}.pdf".replace("/", "_")
    return pdf_dir / file_name


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit_ingest_warning(
    *,
    settings: Settings,
    event: str,
    idx: int,
    total: int,
    paper_id: str,
    arxiv_id: str,
    title: str,
    status: str,
    error: str | None,
) -> None:
    payload = {
        "idx": idx,
        "total": total,
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "title": title,
        "status": status,
        "error": error,
    }
    write_jsonl_event(
        settings.app_log_jsonl,
        {
            "level": "warning",
            "event": event,
            "payload": payload,
        },
    )
    print(
        "ingest.warning "
        f"event={event} idx={idx}/{total} arxiv_id={arxiv_id} "
        f"paper_id={paper_id} status={status} error={error or '-'}"
    )


def _emit_ingest_progress(
    *,
    settings: Settings,
    event: str,
    idx: int,
    total: int,
    paper_id: str,
    arxiv_id: str,
    title: str,
    status: str,
    reason: str | None = None,
) -> None:
    payload = {
        "idx": idx,
        "total": total,
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "title": title,
        "status": status,
        "reason": reason,
    }
    write_jsonl_event(
        settings.app_log_jsonl,
        {
            "level": "info",
            "event": event,
            "payload": payload,
        },
    )
    print(
        "ingest.info "
        f"event={event} idx={idx}/{total} arxiv_id={arxiv_id} "
        f"paper_id={paper_id} status={status} reason={reason or '-'}"
    )
