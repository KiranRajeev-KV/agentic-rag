from __future__ import annotations

import re
import uuid

from agentic_rag.storage.repositories import (
    EpisodeRepository,
    SemanticMemoryRepository,
)


class MemoryService:
    def __init__(
        self,
        semantic_repo: SemanticMemoryRepository,
        episode_repo: EpisodeRepository,
        namespace: str = "project",
    ) -> None:
        self.semantic_repo = semantic_repo
        self.episode_repo = episode_repo
        self.namespace = namespace

    def read_recent(self, limit: int = 10) -> list[dict[str, str]]:
        return self.read_semantic(limit=limit)

    def read_semantic(self, limit: int = 10) -> list[dict[str, str]]:
        rows = self.semantic_repo.list_active(namespace=self.namespace, limit=limit)
        return [
            {
                "memory_id": str(row["memory_id"]),
                "kind": str(row["kind"]),
                "key": str(row["key"]),
                "value": str(row["value"]),
            }
            for row in rows
        ]

    def read_episodes(self, thread_id: str, limit: int = 6) -> list[dict[str, object]]:
        return self.episode_repo.list_recent_by_thread(thread_id=thread_id, limit=limit)

    def write(
        self,
        *,
        kind: str,
        key: str,
        value: str,
        confidence: float,
        source_turn_id: str,
    ) -> str:
        memory_id = f"mem_{uuid.uuid4().hex[:10]}"
        self.semantic_repo.upsert(
            memory_id=memory_id,
            namespace=self.namespace,
            kind=kind,
            key=key,
            value=value,
            confidence=confidence,
            source_turn_id=source_turn_id,
        )
        return memory_id

    def write_episode(
        self,
        *,
        thread_id: str,
        turn_id: str,
        user_query: str,
        route_action: str,
        route_confidence: float | None,
        retrieved_child_ids: list[str],
        selected_parent_ids: list[str],
        tool_calls: list[dict[str, object]],
        final_action: str,
        final_answer_summary: str,
    ) -> str:
        return self.episode_repo.insert_episode(
            thread_id=thread_id,
            turn_id=turn_id,
            user_query=user_query,
            route_action=route_action,
            route_confidence=route_confidence,
            retrieved_child_ids=retrieved_child_ids,
            selected_parent_ids=selected_parent_ids,
            tool_calls=tool_calls,
            final_action=final_action,
            final_answer_summary=final_answer_summary,
        )


def extract_arxiv_ids(text: str) -> list[str]:
    found = re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text)
    deduped: list[str] = []
    for item in found:
        if item not in deduped:
            deduped.append(item)
    return deduped[:8]


def extract_focus(*, user_query: str, assistant_answer: str) -> str:
    for text in (user_query, assistant_answer):
        ids = extract_arxiv_ids(text)
        if ids:
            return f"arxiv:{ids[0]}"
    return user_query[:120]
