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

    def get_by_ids(self, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not paper_ids:
            return {}
        placeholders = ",".join("?" for _ in paper_ids)
        rows = self.store.fetchall(
            f"""
            SELECT paper_id, arxiv_id, title, primary_category, categories, published_at, updated_at
            FROM papers
            WHERE paper_id IN ({placeholders})
            """,
            tuple(paper_ids),
        )
        return {str(row["paper_id"]): dict(row) for row in rows}


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

    def get_by_ids(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not parent_ids:
            return {}
        placeholders = ",".join("?" for _ in parent_ids)
        rows = self.store.fetchall(
            f"""
            SELECT
              parent_id,
              paper_id,
              section_path,
              section_heading,
              section_type,
              page_start,
              page_end,
              parent_text
            FROM parent_sections
            WHERE parent_id IN ({placeholders})
            """,
            tuple(parent_ids),
        )
        return {str(row["parent_id"]): dict(row) for row in rows}


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
                  embedding_model, embedding_config_hash, indexed_embedding_text_hash,
                  last_indexed_at, created_at, chunk_text
                ) VALUES (
                  :chunk_id, :parent_id, :paper_id, :chunk_index, :section_path,
                  :section_type, :content_type, :page_start, :page_end, :token_count,
                  :char_start, :char_end, :docling_ref_ids,
                  :embedding_text_hash, :embedding_model, :embedding_config_hash,
                  :indexed_embedding_text_hash, :last_indexed_at, :created_at, :chunk_text
                )
                """,
                [self._with_index_defaults(row) for row in rows],
            )
            conn.commit()

    def fetch_chunks_for_indexing(
        self,
        limit: int,
        model_name: str,
        config_hash: str,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        predicate = ""
        params: list[Any] = []
        if not force:
            predicate = (
                "AND (c.embedding_model IS NULL OR c.embedding_model = '' "
                "OR c.embedding_model != ? "
                "OR c.embedding_config_hash IS NULL OR c.embedding_config_hash = '' "
                "OR c.embedding_config_hash != ? "
                "OR c.indexed_embedding_text_hash IS NULL "
                "OR c.indexed_embedding_text_hash = '' "
                "OR c.indexed_embedding_text_hash != c.embedding_text_hash)"
            )
            params.extend([model_name, config_hash])
        params.append(limit)
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
              c.embedding_text_hash,
              c.embedding_model,
              c.embedding_config_hash,
              c.indexed_embedding_text_hash,
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

    def count_chunks_total(self) -> int:
        rows = self.store.fetchall("SELECT COUNT(*) AS count FROM child_chunks")
        return int(rows[0]["count"]) if rows else 0

    def count_chunks_pending(self, model_name: str, config_hash: str) -> int:
        rows = self.store.fetchall(
            """
            SELECT COUNT(*) AS count
            FROM child_chunks c
            WHERE
              c.embedding_model IS NULL OR c.embedding_model = ''
              OR c.embedding_model != ?
              OR c.embedding_config_hash IS NULL OR c.embedding_config_hash = ''
              OR c.embedding_config_hash != ?
              OR c.indexed_embedding_text_hash IS NULL OR c.indexed_embedding_text_hash = ''
              OR c.indexed_embedding_text_hash != c.embedding_text_hash
            """,
            (model_name, config_hash),
        )
        return int(rows[0]["count"]) if rows else 0

    def mark_embedded(self, chunk_ids: list[str], model_name: str, config_hash: str) -> None:
        if not chunk_ids:
            return
        now = _utc_now()
        with self.store.connect() as conn:
            conn.executemany(
                """
                UPDATE child_chunks
                SET
                  embedding_model = ?,
                  embedding_config_hash = ?,
                  indexed_embedding_text_hash = embedding_text_hash,
                  last_indexed_at = ?
                WHERE chunk_id = ?
                """,
                [(model_name, config_hash, now, chunk_id) for chunk_id in chunk_ids],
            )
            conn.commit()

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.store.fetchall(
            f"""
            SELECT
              chunk_id,
              parent_id,
              paper_id,
              chunk_index,
              section_path,
              section_type,
              content_type,
              page_start,
              page_end,
              token_count,
              chunk_text
            FROM child_chunks
            WHERE chunk_id IN ({placeholders})
            """,
            tuple(chunk_ids),
        )
        return {str(row["chunk_id"]): dict(row) for row in rows}

    @staticmethod
    def _with_index_defaults(row: dict[str, Any]) -> dict[str, Any]:
        return {
            **row,
            "embedding_model": row.get("embedding_model", ""),
            "embedding_config_hash": row.get("embedding_config_hash", ""),
            "indexed_embedding_text_hash": row.get("indexed_embedding_text_hash", ""),
            "last_indexed_at": row.get("last_indexed_at"),
        }


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

    def list_active(self, namespace: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.store.fetchall(
            """
            SELECT memory_id, namespace, kind, key, value, confidence, source_turn_id, updated_at
            FROM semantic_memories
            WHERE namespace = ? AND is_active = 1
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (namespace, limit),
        )
        return [dict(row) for row in rows]


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

    def add_event(
        self,
        trace_id: str,
        level: str,
        event: str,
        payload_json: str,
        node: str | None = None,
    ) -> None:
        self.store.execute(
            """
            INSERT INTO trace_events (trace_id, ts, level, event, node, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trace_id, _utc_now(), level, event, node, payload_json),
        )

    def get_trace_with_events(self, trace_id: str) -> dict[str, Any] | None:
        traces = self.store.fetchall(
            """
            SELECT trace_id, thread_id, turn_id, run_mode, started_at, completed_at
            FROM traces
            WHERE trace_id = ?
            """,
            (trace_id,),
        )
        if not traces:
            return None
        events = self.store.fetchall(
            """
            SELECT ts, level, event, node, payload_json
            FROM trace_events
            WHERE trace_id = ?
            ORDER BY event_id ASC
            """,
            (trace_id,),
        )
        return {
            "trace": dict(traces[0]),
            "events": [dict(row) for row in events],
        }
