from __future__ import annotations

import json
from typing import Any

from agentic_rag.storage.repositories import TraceRepository
from agentic_rag.storage.sqlite import SQLiteStore


class TraceReader:
    def __init__(self, store: SQLiteStore) -> None:
        self.repo = TraceRepository(store)

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.repo.list_recent(limit=limit)

    def show(self, trace_id: str) -> dict[str, Any] | None:
        record = self.repo.get_trace_with_events(trace_id)
        if not record:
            return None
        decoded_events: list[dict[str, Any]] = []
        for event in record["events"]:
            payload_text = str(event.get("payload_json", "{}"))
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                payload = {"raw": payload_text}
            decoded_events.append({**event, "payload": payload})
        return {"trace": record["trace"], "events": decoded_events}
