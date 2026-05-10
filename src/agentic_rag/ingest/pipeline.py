from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings
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
from agentic_rag.tools.schemas import ArxivPaperMetadata


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
        self.parser = DoclingParser()
        self.store = SQLiteStore(settings.app_db_path)
        self.paper_repo = PaperRepository(self.store)
        self.parent_repo = ParentRepository(self.store)
        self.chunk_repo = ChunkRepository(self.store)

    def run(
        self,
        limit: int,
        force: bool = False,
    ) -> IngestSummary:
        ingest_config = _current_ingest_config(parser_name=self.parser.parser_name)
        discovered = _load_manifest_papers(self.settings.app_corpus_manifest, limit=limit)
        expected_sha_by_id = _manifest_sha_by_id(self.settings.app_corpus_manifest)

        downloaded = 0
        parsed = 0
        parse_failed = 0
        download_failed = 0
        skipped = 0
        errors: list[str] = []
        total = len(discovered)

        for idx, paper in enumerate(discovered, start=1):
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
            local_pdf = _cached_pdf_path(
                pdf_dir=self.settings.app_pdf_dir,
                arxiv_id=paper.arxiv_id,
                version=paper.version,
            )
            expected_sha = expected_sha_by_id.get(paper.arxiv_id, "")
            local_err = _validate_local_pdf(
                paper=paper,
                local_pdf=local_pdf,
                expected_sha=expected_sha,
            )
            if local_err is not None:
                download_failed += 1
                errors.append(local_err)
                _emit_ingest_warning(
                    settings=self.settings,
                    event="pdf.missing_or_corrupt",
                    idx=idx,
                    total=total,
                    paper_id=paper_id,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    status="download_failed",
                    error=local_err,
                )
                continue
            local_sha = _sha256_file(local_pdf)

            skip_reason = self._skip_reason(
                paper=paper,
                local_sha=local_sha,
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

            downloaded += 1
            parse_result = self.parser.parse_pdf(local_pdf)
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
                        _paper_record(
                            paper=paper,
                            paper_id=paper_id,
                            source_query="corpus_manifest",
                            filter_terms=[],
                            filter_reason="manifest_locked",
                            days_back=90,
                            pdf_sha256=local_sha,
                            parse_status="parse_failed",
                            ingest_config=ingest_config,
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
                        _paper_record(
                            paper=paper,
                            paper_id=paper_id,
                            source_query="corpus_manifest",
                            filter_terms=[],
                            filter_reason="manifest_locked",
                            days_back=90,
                            pdf_sha256=local_sha,
                            parse_status="parse_empty",
                            ingest_config=ingest_config,
                        )
                    )
                continue

            self.paper_repo.upsert(
                _paper_record(
                    paper=paper,
                    paper_id=paper_id,
                    source_query="corpus_manifest",
                    filter_terms=[],
                    filter_reason="manifest_locked",
                    days_back=90,
                    pdf_sha256=local_sha,
                    parse_status="parsed",
                    ingest_config=ingest_config,
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
        paper: ArxivPaperMetadata,
        local_sha: str,
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
        if local_sha != stored_pdf_sha:
            return None
        return "already_parsed_unchanged"


def _paper_record(
    *,
    paper: ArxivPaperMetadata,
    paper_id: str,
    source_query: str,
    filter_terms: list[str],
    filter_reason: str,
    days_back: int,
    pdf_sha256: str,
    parse_status: str,
    ingest_config: IngestConfig,
) -> PaperRecord:
    return PaperRecord(
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
        source_query=source_query,
        discovery_source_query=source_query,
        discovery_filter_terms=filter_terms,
        discovery_filter_reason=filter_reason,
        discovery_days_back=days_back,
        pdf_sha256=pdf_sha256,
        parse_status=parse_status,
        parser_name=ingest_config.parser_name,
        parser_version=ingest_config.parser_version,
        chunker_name=ingest_config.chunker_name,
        chunker_config_hash=ingest_config.chunker_config_hash,
    )


def _load_manifest_papers(manifest_path: Path, limit: int) -> list[ArxivPaperMetadata]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = payload.get("papers", [])
    out: list[ArxivPaperMetadata] = []
    for row in papers[:limit]:
        out.append(
            ArxivPaperMetadata(
                arxiv_id=str(row.get("arxiv_id", "")),
                version=row.get("arxiv_version"),
                title=str(row.get("title", "")),
                authors=list(row.get("authors") or []),
                abstract=row.get("abstract"),
                categories=list(row.get("categories") or []),
                primary_category=row.get("primary_category"),
                published_at=row.get("published_at"),
                updated_at=row.get("updated_at"),
                abs_url=row.get("abs_url"),
                pdf_url=row.get("pdf_url"),
                doi=row.get("doi"),
                comment=row.get("comment"),
            )
        )
    return out


def _manifest_sha_by_id(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id: dict[str, str] = {}
    for row in payload.get("papers", []):
        arxiv_id = str(row.get("arxiv_id", "")).strip()
        if arxiv_id:
            by_id[arxiv_id] = str(row.get("pdf_sha256") or "")
    return by_id


def _validate_local_pdf(
    *,
    paper: ArxivPaperMetadata,
    local_pdf: Path,
    expected_sha: str,
) -> str | None:
    hint = "Run `uv run python scripts/download_corpus_pdfs.py`"
    if not local_pdf.exists():
        return f"{paper.arxiv_id}: missing pdf `{local_pdf}`. {hint}"
    if expected_sha:
        local_sha = _sha256_file(local_pdf)
        if local_sha != expected_sha:
            return f"{paper.arxiv_id}: sha mismatch for `{local_pdf}`. {hint}"
    return None


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
