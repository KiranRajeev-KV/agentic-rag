from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from agentic_rag.tools.schemas import ArxivPaperMetadata


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    paper_path: Path | None
    pdf_sha256: str | None
    error: str | None = None


def download_pdf(
    paper: ArxivPaperMetadata,
    pdf_dir: Path,
    user_agent: str,
    timeout_seconds: float = 60.0,
) -> DownloadResult:
    if not paper.pdf_url:
        return DownloadResult(ok=False, paper_path=None, pdf_sha256=None, error="missing pdf_url")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{paper.arxiv_id}{paper.version or ''}.pdf".replace("/", "_")
    target = pdf_dir / file_name

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.get(paper.pdf_url, headers={"User-Agent": user_agent})
            response.raise_for_status()
            content = response.content
    except httpx.HTTPError as err:
        return DownloadResult(ok=False, paper_path=None, pdf_sha256=None, error=str(err))

    target.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return DownloadResult(ok=True, paper_path=target, pdf_sha256=digest)
