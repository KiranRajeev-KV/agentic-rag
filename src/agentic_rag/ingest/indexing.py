from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client.http import models

from agentic_rag.config import Settings
from agentic_rag.ingest.embeddings import BgeM3DenseEmbedder
from agentic_rag.storage.qdrant import ensure_child_chunk_collection, get_client
from agentic_rag.storage.repositories import ChunkRepository
from agentic_rag.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class IndexSummary:
    selected_chunks: int
    indexed_chunks: int
    model_name: str
    device: str


class ChildChunkIndexer:
    def __init__(self, settings: Settings, embedder: BgeM3DenseEmbedder | None = None) -> None:
        self.settings = settings
        self.store = SQLiteStore(settings.app_db_path)
        self.chunk_repo = ChunkRepository(self.store)
        self.embedder = embedder or BgeM3DenseEmbedder(settings=settings)

    def index_unembedded_chunks(self, limit: int = 500, batch_size: int = 32) -> IndexSummary:
        client = get_client(self.settings.qdrant_url)
        ensure_child_chunk_collection(
            client=client, collection_name=self.settings.qdrant_collection
        )

        rows = self.chunk_repo.fetch_chunks_for_indexing(limit=limit)
        if not rows:
            return IndexSummary(
                selected_chunks=0,
                indexed_chunks=0,
                model_name=self.settings.bge_model_name,
                device=self.embedder.device,
            )

        texts = [row["chunk_text"] for row in rows]
        batch = self.embedder.encode(texts=texts, batch_size=batch_size)

        points: list[models.PointStruct] = []
        for row, vector in zip(rows, batch.dense_vectors, strict=False):
            points.append(
                models.PointStruct(
                    id=row["chunk_id"],
                    vector=vector,
                    payload=_payload_from_row(row),
                )
            )

        client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=points,
            wait=True,
        )
        self.chunk_repo.mark_embedded(
            chunk_ids=[row["chunk_id"] for row in rows], model_name=batch.model_name
        )
        return IndexSummary(
            selected_chunks=len(rows),
            indexed_chunks=len(points),
            model_name=batch.model_name,
            device=batch.device,
        )


def _payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],
        "parent_id": row["parent_id"],
        "paper_id": row["paper_id"],
        "arxiv_id": row["arxiv_id"],
        "title": row["title"],
        "primary_category": row["primary_category"],
        "categories": row["categories"],
        "published_at": row["published_at"],
        "updated_at": row["updated_at"],
        "section_path": row["section_path"],
        "section_type": row["section_type"],
        "content_type": row["content_type"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "chunk_index": row["chunk_index"],
        "token_count": row["token_count"],
    }
