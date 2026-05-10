from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client.http import models

from agentic_rag.config import Settings
from agentic_rag.ingest.embeddings import OpenAIEmbedder
from agentic_rag.storage.qdrant import ensure_child_chunk_collection, get_client
from agentic_rag.storage.repositories import ChunkRepository
from agentic_rag.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class IndexSummary:
    total_chunks: int
    pending_chunks: int
    selected_chunks: int
    indexed_chunks: int
    skipped_as_up_to_date: int
    force: bool
    model_name: str
    config_hash: str
    dimensions: int


class ChildChunkIndexer:
    _QDRANT_UPSERT_POINTS_BATCH_SIZE = 512

    def __init__(self, settings: Settings, embedder: OpenAIEmbedder | None = None) -> None:
        self.settings = settings
        self.store = SQLiteStore(settings.app_db_path)
        self.chunk_repo = ChunkRepository(self.store)
        self.embedder = embedder or OpenAIEmbedder(settings=settings)

    def index_unembedded_chunks(
        self,
        limit: int | None = None,
        batch_size: int = 32,
        force: bool = False,
    ) -> IndexSummary:
        config_hash = self._embedding_config_hash()
        client = get_client(self.settings.qdrant_url)
        ensure_child_chunk_collection(
            client=client,
            collection_name=self.settings.qdrant_collection,
            vector_size=self.settings.embedding_dimensions,
        )

        total_chunks = self.chunk_repo.count_chunks_total()
        pending_chunks = (
            total_chunks
            if force
            else self.chunk_repo.count_chunks_pending(
                model_name=self.settings.embedding_model,
                config_hash=config_hash,
            )
        )
        empty_pending_chunks = self._count_empty_pending_chunks(
            model_name=self.settings.embedding_model,
            config_hash=config_hash,
            force=force,
        )
        selected_chunks = 0
        indexed_chunks = 0
        skipped_as_up_to_date = total_chunks - pending_chunks
        skipped_empty_chunks = 0
        batch_no = 0

        print(
            "index.info "
            f"pending_chunks={pending_chunks} empty_pending_chunks={empty_pending_chunks} "
            "empty chunks will be skipped from embedding",
            flush=True,
        )

        while True:
            fetch_limit: int | None = None
            if limit is not None:
                remaining_target = limit - selected_chunks
                if remaining_target <= 0:
                    break
                fetch_limit = remaining_target

            rows = self.chunk_repo.fetch_chunks_for_indexing(
                limit=fetch_limit,
                model_name=self.settings.embedding_model,
                config_hash=config_hash,
                force=force,
            )
            if not rows:
                break

            batch_no += 1
            print(
                "index.batch.start "
                f"batch={batch_no} rows={len(rows)} "
                f"selected_so_far={selected_chunks} indexed_so_far={indexed_chunks} "
                f"limit={limit if limit is not None else 'all'}",
                flush=True,
            )
            rows_to_embed = [row for row in rows if str(row.get("chunk_text") or "").strip()]
            skipped_rows = [row for row in rows if not str(row.get("chunk_text") or "").strip()]

            if skipped_rows:
                skipped_empty_chunks += len(skipped_rows)
                self.chunk_repo.mark_embedded(
                    chunk_ids=[row["chunk_id"] for row in skipped_rows],
                    model_name=self.settings.embedding_model,
                    config_hash=config_hash,
                )
                print(
                    "index.batch.skip "
                    f"batch={batch_no} skipped_empty={len(skipped_rows)} "
                    f"skipped_empty_total={skipped_empty_chunks}",
                    flush=True,
                )

            if not rows_to_embed:
                if force:
                    break
                continue

            texts = [row["chunk_text"] for row in rows_to_embed]
            batch = self.embedder.encode(texts=texts, batch_size=batch_size)

            points: list[models.PointStruct] = []
            for row, vector in zip(rows_to_embed, batch.dense_vectors, strict=False):
                points.append(
                    models.PointStruct(
                        id=_point_uuid(row["chunk_id"]),
                        vector=vector,
                        payload=_payload_from_row(row),
                    )
                )

            point_sub_batches = [
                points[i : i + self._QDRANT_UPSERT_POINTS_BATCH_SIZE]
                for i in range(0, len(points), self._QDRANT_UPSERT_POINTS_BATCH_SIZE)
            ]
            print(
                "index.batch.upsert "
                f"batch={batch_no} sub_batches={len(point_sub_batches)} "
                f"points_per_batch={self._QDRANT_UPSERT_POINTS_BATCH_SIZE}",
                flush=True,
            )
            for sub_batch_idx, point_sub_batch in enumerate(point_sub_batches, start=1):
                client.upsert(
                    collection_name=self.settings.qdrant_collection,
                    points=point_sub_batch,
                    wait=True,
                )
                print(
                    "index.batch.upsert_done "
                    f"batch={batch_no} sub_batch={sub_batch_idx}/{len(point_sub_batches)} "
                    f"points={len(point_sub_batch)}",
                    flush=True,
                )
            self.chunk_repo.mark_embedded(
                chunk_ids=[row["chunk_id"] for row in rows_to_embed],
                model_name=batch.model_name,
                config_hash=config_hash,
            )
            selected_chunks += len(rows_to_embed)
            indexed_chunks += len(points)
            remaining = max(pending_chunks - selected_chunks, 0)
            print(
                "index.batch.done "
                f"batch={batch_no} indexed_batch={len(points)} "
                f"indexed_total={indexed_chunks} remaining_estimate={remaining}",
                flush=True,
            )

            if force:
                break

        return IndexSummary(
            total_chunks=total_chunks,
            pending_chunks=pending_chunks,
            selected_chunks=selected_chunks,
            indexed_chunks=indexed_chunks,
            skipped_as_up_to_date=skipped_as_up_to_date,
            force=force,
            model_name=self.settings.embedding_model,
            config_hash=config_hash,
            dimensions=self.settings.embedding_dimensions,
        )

    def _count_empty_pending_chunks(self, model_name: str, config_hash: str, force: bool) -> int:
        if force:
            row = self.store.fetchall(
                """
                SELECT COUNT(*) AS count
                FROM child_chunks
                WHERE TRIM(COALESCE(chunk_text, '')) = ''
                """
            )
            return int(row[0]["count"]) if row else 0

        row = self.store.fetchall(
            """
            SELECT COUNT(*) AS count
            FROM child_chunks c
            WHERE TRIM(COALESCE(c.chunk_text, '')) = ''
              AND (
                c.embedding_model IS NULL OR c.embedding_model = ''
                OR c.embedding_model != ?
                OR c.embedding_config_hash IS NULL OR c.embedding_config_hash = ''
                OR c.embedding_config_hash != ?
                OR c.indexed_embedding_text_hash IS NULL OR c.indexed_embedding_text_hash = ''
                OR c.indexed_embedding_text_hash != c.embedding_text_hash
              )
            """,
            (model_name, config_hash),
        )
        return int(row[0]["count"]) if row else 0

    def _embedding_config_hash(self) -> str:
        material = (
            f"provider={self.settings.embedding_provider}|"
            f"model={self.settings.embedding_model}|"
            f"dimensions={self.settings.embedding_dimensions}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


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


def _point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentic-rag:{chunk_id}"))
