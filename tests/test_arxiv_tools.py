from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.config import get_settings
from agentic_rag.tools.arxiv_tools import ArxivToolset
from agentic_rag.tools.schemas import (
    ArxivLookupByIdInput,
    ArxivPaperMetadata,
    ArxivSearchInput,
    ToolStatus,
)


class _FakeArxivClient:
    def __init__(self) -> None:
        self.lookup_calls = 0
        self.search_calls = 0

    def lookup_by_ids(self, arxiv_ids: list[str]) -> list[ArxivPaperMetadata]:
        self.lookup_calls += 1
        return [
            ArxivPaperMetadata(
                arxiv_id=arxiv_ids[0],
                version="v1",
                title="Test Paper",
                authors=["A. Author"],
                abstract="Abstract",
                categories=["cs.AI"],
                primary_category="cs.AI",
            )
        ]

    def search(self, *args, **kwargs) -> list[ArxivPaperMetadata]:  # noqa: ANN002, ANN003
        self.search_calls += 1
        return [
            ArxivPaperMetadata(
                arxiv_id="2501.00001",
                version="v1",
                title="Search Result",
                authors=["B. Author"],
                abstract="Summary",
                categories=["cs.AI"],
                primary_category="cs.AI",
            )
        ]


def _build_toolset(tmp_path: Path, monkeypatch) -> ArxivToolset:
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()
    return ArxivToolset(settings=settings, cache_ttl_seconds=3600)


def test_lookup_by_id_uses_cache(tmp_path: Path, monkeypatch) -> None:
    toolset = _build_toolset(tmp_path=tmp_path, monkeypatch=monkeypatch)
    fake = _FakeArxivClient()
    toolset._client = fake  # noqa: SLF001

    payload = ArxivLookupByIdInput(arxiv_ids=["2501.00001v1"])
    first = toolset.arxiv_lookup_by_id(payload)
    second = toolset.arxiv_lookup_by_id(payload)

    assert first.status == ToolStatus.ok
    assert first.source == "arxiv_api"
    assert second.source == "cache"
    assert fake.lookup_calls == 1


def test_arxiv_search_schema_enforces_max_results() -> None:
    bad = {"query": "agent memory", "max_results": 15}
    with pytest.raises(ValidationError):
        ArxivSearchInput.model_validate(bad)


def test_lookup_error_returns_error_status(tmp_path: Path, monkeypatch) -> None:
    toolset = _build_toolset(tmp_path=tmp_path, monkeypatch=monkeypatch)

    class _ErrorClient:
        def __init__(self) -> None:
            self.calls = 0

        def lookup_by_ids(self, arxiv_ids: list[str]) -> list[ArxivPaperMetadata]:
            del arxiv_ids
            self.calls += 1
            raise RuntimeError("network down")

    error_client = _ErrorClient()
    toolset._client = error_client  # type: ignore[assignment]  # noqa: SLF001
    payload = ArxivLookupByIdInput(arxiv_ids=["2501.00002v1"])
    first = toolset.arxiv_lookup_by_id(payload)
    second = toolset.arxiv_lookup_by_id(payload)
    assert first.status == ToolStatus.error
    assert second.status == ToolStatus.error
    assert first.errors
    assert error_client.calls == 2
