from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from agentic_rag.storage.repositories import TraceRepository
from agentic_rag.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    thread_id: str
    turn_id: str
    run_mode: str


class TraceWriter:
    def __init__(self, store: SQLiteStore) -> None:
        self.repo = TraceRepository(store)

    def start(self, thread_id: str, run_mode: str) -> TraceContext:
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        self.repo.start_trace(
            trace_id=trace_id, thread_id=thread_id, turn_id=turn_id, run_mode=run_mode
        )
        self.event(trace_id, "info", "turn.started", {"thread_id": thread_id, "turn_id": turn_id})
        return TraceContext(
            trace_id=trace_id, thread_id=thread_id, turn_id=turn_id, run_mode=run_mode
        )

    def event(
        self,
        trace_id: str,
        level: str,
        event: str,
        payload: dict[str, Any],
        node: str | None = None,
    ) -> None:
        self.repo.add_event(
            trace_id=trace_id,
            level=level,
            event=event,
            node=node,
            payload_json=json.dumps(payload, ensure_ascii=True, sort_keys=True),
        )

    def complete(self, trace_id: str, final_action: str) -> None:
        self.event(trace_id, "info", "turn.completed", {"final_action": final_action})
        self.repo.complete_trace(trace_id)
