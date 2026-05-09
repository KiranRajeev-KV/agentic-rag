from __future__ import annotations

import uuid

from agentic_rag.storage.repositories import SemanticMemoryRepository


class MemoryService:
    def __init__(self, repo: SemanticMemoryRepository, namespace: str = "project") -> None:
        self.repo = repo
        self.namespace = namespace

    def read_recent(self, limit: int = 10) -> list[dict[str, str]]:
        rows = self.repo.list_active(namespace=self.namespace, limit=limit)
        return [
            {
                "memory_id": str(row["memory_id"]),
                "kind": str(row["kind"]),
                "key": str(row["key"]),
                "value": str(row["value"]),
            }
            for row in rows
        ]

    def write_decision(self, key: str, value: str, source_turn_id: str) -> str:
        memory_id = f"mem_{uuid.uuid4().hex[:10]}"
        self.repo.upsert(
            memory_id=memory_id,
            namespace=self.namespace,
            kind="decision",
            key=key,
            value=value,
            confidence=0.8,
            source_turn_id=source_turn_id,
        )
        return memory_id
