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

    if init_qdrant:
        client = get_client(settings.qdrant_url)
        ensure_child_chunk_collection(
            client=client,
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dimensions,
        )
