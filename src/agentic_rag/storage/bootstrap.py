from __future__ import annotations

from pathlib import Path

from agentic_rag.config import Settings
from agentic_rag.storage.qdrant import ensure_child_chunk_collection, get_client
from agentic_rag.storage.sqlite import SQLiteStore


def schema_file_path() -> Path:
    return Path(__file__).with_name("schema.sql")


def initialize_storage(settings: Settings, init_qdrant: bool = False) -> None:
    store = SQLiteStore(settings.app_db_path)
    store.initialize_schema(schema_file_path())
    _migrate_child_chunk_columns(store)

    if init_qdrant:
        client = get_client(settings.qdrant_url)
        ensure_child_chunk_collection(client=client, collection_name=settings.qdrant_collection)


def _migrate_child_chunk_columns(store: SQLiteStore) -> None:
    expected_columns: dict[str, str] = {
        "embedding_config_hash": "TEXT",
        "indexed_embedding_text_hash": "TEXT",
        "last_indexed_at": "TEXT",
    }
    with store.connect() as conn:
        existing_rows = conn.execute("PRAGMA table_info(child_chunks)").fetchall()
        existing = {row["name"] for row in existing_rows}
        for column, definition in expected_columns.items():
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE child_chunks ADD COLUMN {column} {definition}")
        conn.commit()
