from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agentic_rag.storage.sqlite import SQLiteStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    parse_status: str = "pending"
    ingested_at: str = ""
    arxiv_version: str | None = None
    primary_category: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    pdf_url: str | None = None
    abs_url: str | None = None
    doi: str | None = None
    comment: str | None = None
    source_query: str | None = None
    pdf_sha256: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    chunker_name: str | None = None
    chunker_config_hash: str | None = None


class PaperRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def upsert(self, record: PaperRecord) -> None:
        ingested_at = record.ingested_at or _utc_now()
        self.store.execute(
            """
            INSERT INTO papers (
              paper_id, arxiv_id, arxiv_version, title, authors, abstract,
              primary_category, categories, published_at, updated_at, pdf_url,
              abs_url, doi, comment, source_query, ingested_at, pdf_sha256,
              parse_status, parser_name, parser_version, chunker_name,
              chunker_config_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
              arxiv_id=excluded.arxiv_id,
              arxiv_version=excluded.arxiv_version,
              title=excluded.title,
              authors=excluded.authors,
              abstract=excluded.abstract,
              primary_category=excluded.primary_category,
              categories=excluded.categories,
              published_at=excluded.published_at,
              updated_at=excluded.updated_at,
              pdf_url=excluded.pdf_url,
              abs_url=excluded.abs_url,
              doi=excluded.doi,
              comment=excluded.comment,
              source_query=excluded.source_query,
              ingested_at=excluded.ingested_at,
              pdf_sha256=excluded.pdf_sha256,
              parse_status=excluded.parse_status,
              parser_name=excluded.parser_name,
              parser_version=excluded.parser_version,
              chunker_name=excluded.chunker_name,
              chunker_config_hash=excluded.chunker_config_hash
            """,
            (
                record.paper_id,
                record.arxiv_id,
                record.arxiv_version,
                record.title,
                json.dumps(record.authors, ensure_ascii=True),
                record.abstract,
                record.primary_category,
                json.dumps(record.categories, ensure_ascii=True),
                record.published_at,
                record.updated_at,
                record.pdf_url,
                record.abs_url,
                record.doi,
                record.comment,
                record.source_query,
                ingested_at,
                record.pdf_sha256,
                record.parse_status,
                record.parser_name,
                record.parser_version,
                record.chunker_name,
                record.chunker_config_hash,
            ),
        )

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.store.fetchall(
            """
            SELECT paper_id, arxiv_id, title, parse_status, ingested_at
            FROM papers
            ORDER BY ingested_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]


class ParentRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def insert_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.store.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO parent_sections (
                  parent_id, paper_id, section_path, section_heading, section_type,
                  section_level, parent_index, page_start, page_end, token_count,
                  char_count, content_types, child_chunk_ids, text_hash, parent_text
                ) VALUES (
                  :parent_id, :paper_id, :section_path, :section_heading, :section_type,
                  :section_level, :parent_index, :page_start, :page_end, :token_count,
                  :char_count, :content_types, :child_chunk_ids, :text_hash, :parent_text
                )
                """,
                rows,
            )
            conn.commit()


class ChunkRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def insert_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.store.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO child_chunks (
                  chunk_id, parent_id, paper_id, chunk_index, section_path,
                  section_type, content_type, page_start, page_end, token_count,
                  char_start, char_end, docling_ref_ids, embedding_text_hash,
                  embedding_model, created_at, chunk_text
                ) VALUES (
                  :chunk_id, :parent_id, :paper_id, :chunk_index, :section_path,
                  :section_type, :content_type, :page_start, :page_end, :token_count,
                  :char_start, :char_end, :docling_ref_ids,
                  :embedding_text_hash, :embedding_model, :created_at, :chunk_text
                )
                """,
                rows,
            )
            conn.commit()

    def fetch_chunks_for_indexing(
        self, limit: int, embedded_model: str | None = None
    ) -> list[dict[str, Any]]:
        predicate = "AND (c.embedding_model IS NULL OR c.embedding_model = '')"
        params: list[Any] = [limit]
        if embedded_model:
            predicate = (
                "AND (c.embedding_model IS NULL OR c.embedding_model = '' "
                "OR c.embedding_model != ?)"
            )
            params = [embedded_model, limit]
        rows = self.store.fetchall(
            f"""
            SELECT
              c.chunk_id,
              c.parent_id,
              c.paper_id,
              c.chunk_index,
              c.section_path,
              c.section_type,
              c.content_type,
              c.page_start,
              c.page_end,
              c.token_count,
              c.chunk_text,
              p.arxiv_id,
              p.title,
              p.primary_category,
              p.categories,
              p.published_at,
              p.updated_at
            FROM child_chunks c
            JOIN papers p ON p.paper_id = c.paper_id
            WHERE 1=1 {predicate}
            ORDER BY p.ingested_at DESC, c.chunk_index ASC
            LIMIT ?
            """,
            tuple(params),
        )
        return [dict(row) for row in rows]

    def mark_embedded(self, chunk_ids: list[str], model_name: str) -> None:
        if not chunk_ids:
            return
        with self.store.connect() as conn:
            conn.executemany(
                "UPDATE child_chunks SET embedding_model = ? WHERE chunk_id = ?",
                [(model_name, chunk_id) for chunk_id in chunk_ids],
            )
            conn.commit()


class SemanticMemoryRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def upsert(
        self,
        memory_id: str,
        namespace: str,
        kind: str,
        key: str,
        value: str,
        confidence: float,
        source_turn_id: str | None = None,
    ) -> None:
        now = _utc_now()
        self.store.execute(
            """
            INSERT INTO semantic_memories (
              memory_id, namespace, kind, key, value, confidence, source_turn_id,
              created_at, updated_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(memory_id) DO UPDATE SET
              namespace=excluded.namespace,
              kind=excluded.kind,
              key=excluded.key,
              value=excluded.value,
              confidence=excluded.confidence,
              source_turn_id=excluded.source_turn_id,
              updated_at=excluded.updated_at,
              is_active=1
            """,
            (memory_id, namespace, kind, key, value, confidence, source_turn_id, now, now),
        )


class TraceRepository:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def start_trace(self, trace_id: str, thread_id: str, turn_id: str, run_mode: str) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO traces (
              trace_id, thread_id, turn_id, run_mode, started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (trace_id, thread_id, turn_id, run_mode, _utc_now()),
        )

    def complete_trace(self, trace_id: str) -> None:
        self.store.execute(
            "UPDATE traces SET completed_at = ? WHERE trace_id = ?", (_utc_now(), trace_id)
        )

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.store.fetchall(
            """
            SELECT trace_id, thread_id, turn_id, run_mode, started_at, completed_at
            FROM traces
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]
