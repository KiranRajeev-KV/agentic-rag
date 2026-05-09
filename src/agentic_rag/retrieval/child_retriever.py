from __future__ import annotations

from qdrant_client.http import models

from agentic_rag.config import Settings
from agentic_rag.ingest.embeddings import OpenAIEmbedder
from agentic_rag.storage.qdrant import get_client

from .types import ChildHit


class ChildRetriever:
    def __init__(self, settings: Settings, embedder: OpenAIEmbedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or OpenAIEmbedder(settings=settings)
        self.client = get_client(settings.qdrant_url)

    def retrieve(
        self,
        query: str,
        top_n: int = 30,
        include_references: bool = False,
    ) -> list[ChildHit]:
        query_embedding = self.embedder.encode([query], batch_size=1).dense_vectors[0]
        response = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_embedding,
            limit=top_n,
            with_payload=True,
            with_vectors=False,
        )

        hits: list[ChildHit] = []
        for point in response.points:
            payload = point.payload or {}
            section_type = str(payload.get("section_type", "content") or "content")
            score = float(point.score)

            if section_type == "references" and not include_references:
                score -= 0.25
            if score <= 0:
                continue

            chunk_id = str(payload.get("chunk_id", point.id))
            hits.append(
                ChildHit(
                    chunk_id=chunk_id,
                    parent_id=str(payload.get("parent_id", "")),
                    paper_id=str(payload.get("paper_id", "")),
                    section_path=str(payload.get("section_path", "")),
                    section_type=section_type,
                    content_type=str(payload.get("content_type", "text")),
                    score=score,
                    payload=payload,
                )
            )

        return sorted(hits, key=lambda item: item.score, reverse=True)


def references_filter(include_references: bool) -> models.Filter | None:
    if include_references:
        return None
    return models.Filter(
        must_not=[
            models.FieldCondition(
                key="section_type",
                match=models.MatchValue(value="references"),
            )
        ]
    )
