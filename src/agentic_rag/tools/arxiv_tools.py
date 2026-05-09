from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from agentic_rag.config import Settings
from agentic_rag.corpus.arxiv_client import ArxivApiClient
from agentic_rag.tools.schemas import (
    ArxivGetRecentInput,
    ArxivLookupByIdInput,
    ArxivSearchInput,
    ArxivSortBy,
    ArxivSortOrder,
    ArxivToolOutput,
    ToolStatus,
)


class ArxivToolset:
    def __init__(self, settings: Settings, cache_ttl_seconds: int = 6 * 60 * 60) -> None:
        self._client = ArxivApiClient(user_agent=settings.arxiv_user_agent)
        self._cache = _JsonCache(
            root_dir=settings.app_runs_dir / "cache" / "arxiv", ttl_seconds=cache_ttl_seconds
        )

    def arxiv_lookup_by_id(self, payload: ArxivLookupByIdInput) -> ArxivToolOutput:
        return self._run_with_cache(
            "arxiv_lookup_by_id", payload.model_dump(), self._lookup_by_id, payload
        )

    def arxiv_search(self, payload: ArxivSearchInput) -> ArxivToolOutput:
        return self._run_with_cache(
            "arxiv_search", payload.model_dump(mode="json"), self._search, payload
        )

    def arxiv_get_recent(self, payload: ArxivGetRecentInput) -> ArxivToolOutput:
        return self._run_with_cache(
            "arxiv_get_recent", payload.model_dump(), self._get_recent, payload
        )

    def _lookup_by_id(self, payload: ArxivLookupByIdInput) -> ArxivToolOutput:
        try:
            papers = self._client.lookup_by_ids(payload.arxiv_ids)
            if not payload.include_abstract:
                papers = [paper.model_copy(update={"abstract": None}) for paper in papers]
            return ArxivToolOutput(status=ToolStatus.ok, papers=papers, errors=[])
        except Exception as err:  # noqa: BLE001
            return ArxivToolOutput(status=ToolStatus.error, papers=[], errors=[str(err)])

    def _search(self, payload: ArxivSearchInput) -> ArxivToolOutput:
        try:
            papers = self._client.search(
                query=payload.query,
                categories=payload.categories,
                max_results=payload.max_results,
                sort_by=payload.sort_by,
                sort_order=payload.sort_order,
                date_from=payload.date_from,
                date_to=payload.date_to,
            )
            return ArxivToolOutput(status=ToolStatus.ok, papers=papers, errors=[])
        except Exception as err:  # noqa: BLE001
            return ArxivToolOutput(status=ToolStatus.error, papers=[], errors=[str(err)])

    def _get_recent(self, payload: ArxivGetRecentInput) -> ArxivToolOutput:
        query = payload.query_filter if payload.query_filter else "*"
        try:
            papers = self._client.search(
                query=query,
                categories=[payload.category],
                max_results=payload.max_results,
                sort_by=ArxivSortBy.submitted_date,
                sort_order=ArxivSortOrder.descending,
            )
            return ArxivToolOutput(status=ToolStatus.ok, papers=papers, errors=[])
        except Exception as err:  # noqa: BLE001
            return ArxivToolOutput(status=ToolStatus.error, papers=[], errors=[str(err)])

    def _run_with_cache(
        self,
        tool_name: str,
        payload: dict[str, object],
        handler: callable,
        payload_model: object,
    ) -> ArxivToolOutput:
        cache_key = _stable_cache_key(tool_name=tool_name, payload=payload)
        cached = self._cache.get(cache_key)
        if cached:
            try:
                return ArxivToolOutput.model_validate({**cached, "source": "cache"})
            except ValidationError:
                self._cache.delete(cache_key)

        result = handler(payload_model)
        self._cache.set(cache_key, result.model_dump(mode="json"))
        return result


def _stable_cache_key(tool_name: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{tool_name}_{digest}"


class _JsonCache:
    def __init__(self, root_dir: Path, ttl_seconds: int) -> None:
        self.root_dir = root_dir
        self.ttl_seconds = ttl_seconds
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, object] | None:
        path = self.root_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        created_at_ts = float(content.get("created_at_ts", 0.0))
        if created_at_ts <= 0:
            return None
        now_ts = __import__("time").time()
        if (now_ts - created_at_ts) > self.ttl_seconds:
            return None
        payload = content.get("payload")
        return payload if isinstance(payload, dict) else None

    def set(self, key: str, payload: dict[str, object]) -> None:
        now_ts = __import__("time").time()
        target = {"created_at_ts": now_ts, "payload": payload}
        (self.root_dir / f"{key}.json").write_text(
            json.dumps(target, ensure_ascii=True), encoding="utf-8"
        )

    def delete(self, key: str) -> None:
        path = self.root_dir / f"{key}.json"
        if path.exists():
            path.unlink()
