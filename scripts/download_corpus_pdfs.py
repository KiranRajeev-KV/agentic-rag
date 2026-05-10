from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.corpus.download import download_pdf
from agentic_rag.tools.schemas import ArxivPaperMetadata


@dataclass(frozen=True)
class DownloadStats:
    total: int
    ok: int
    skipped_ok: int
    downloaded: int
    redownloaded: int
    failed: int


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    settings = get_settings()
    manifest_path = settings.app_corpus_manifest
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = payload.get("papers", [])

    total = len(papers)
    ok = 0
    skipped_ok = 0
    downloaded = 0
    redownloaded = 0
    failed = 0

    for idx, row in enumerate(papers, start=1):
        paper = ArxivPaperMetadata(
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
        expected_sha = str(row.get("pdf_sha256") or "")
        file_name = f"{paper.arxiv_id}{paper.version or ''}.pdf".replace("/", "_")
        target = settings.app_pdf_dir / file_name

        if target.exists() and expected_sha:
            local_sha = _sha256_file(target)
            if local_sha == expected_sha:
                skipped_ok += 1
                ok += 1
                print(
                    f"download.skip idx={idx}/{total} arxiv_id={paper.arxiv_id} "
                    "reason=already_present_sha_ok"
                )
                continue

        had_existing = target.exists()
        result = download_pdf(
            paper=paper, pdf_dir=settings.app_pdf_dir, user_agent=settings.arxiv_user_agent
        )
        if not result.ok or not result.paper_path:
            failed += 1
            print(f"download.fail idx={idx}/{total} arxiv_id={paper.arxiv_id} error={result.error}")
            continue

        if expected_sha and result.pdf_sha256 != expected_sha:
            failed += 1
            print(
                f"download.fail idx={idx}/{total} arxiv_id={paper.arxiv_id} "
                "error=sha_mismatch_after_download"
            )
            continue

        ok += 1
        if had_existing:
            redownloaded += 1
        else:
            downloaded += 1
        print(
            f"download.ok idx={idx}/{total} arxiv_id={paper.arxiv_id} "
            f"sha={result.pdf_sha256 or '-'}"
        )
        # arXiv legacy API guidance recommends >=3s pacing; apply to PDF fetches too.
        time.sleep(3.1)

    stats = DownloadStats(
        total=total,
        ok=ok,
        skipped_ok=skipped_ok,
        downloaded=downloaded,
        redownloaded=redownloaded,
        failed=failed,
    )
    print(
        "download.summary "
        f"total={stats.total} ok={stats.ok} skipped_ok={stats.skipped_ok} "
        f"downloaded={stats.downloaded} redownloaded={stats.redownloaded} failed={stats.failed}"
    )


if __name__ == "__main__":
    main()
