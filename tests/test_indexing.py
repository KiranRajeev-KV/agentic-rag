from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.ingest.embeddings import EmbeddingBatch
from agentic_rag.ingest.indexing import ChildChunkIndexer
from agentic_rag.storage.bootstrap import initialize_storage
from agentic_rag.storage.repositories import (
    ChunkRepository,
    PaperRecord,
    PaperRepository,
    ParentRepository,
)
from agentic_rag.storage.sqlite import SQLiteStore


def test_child_chunk_indexing_with_fake_embedder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()

    initialize_storage(settings=settings, init_qdrant=False)
    store = SQLiteStore(settings.app_db_path)
    paper_repo = PaperRepository(store)
    parent_repo = ParentRepository(store)
    chunk_repo = ChunkRepository(store)

    paper_repo.upsert(
        PaperRecord(
            paper_id="paper_1",
            arxiv_id="2501.00001",
            title="Agentic Paper",
            authors=["A"],
            abstract="retrieval memory",
            categories=["cs.AI"],
            parse_status="parsed",
        )
    )
    parent_repo.insert_many(
        [
            {
                "parent_id": "parent_1",
                "paper_id": "paper_1",
                "section_path": "1 Intro",
                "section_heading": "Intro",
                "section_type": "content",
                "section_level": 1,
                "parent_index": 0,
                "page_start": 1,
                "page_end": 1,
                "token_count": 10,
                "char_count": 50,
                "content_types": '["text"]',
                "child_chunk_ids": '["chunk_1"]',
                "text_hash": "x",
                "parent_text": "text",
            }
        ]
    )
    chunk_repo.insert_many(
        [
            {
                "chunk_id": "chunk_1",
                "parent_id": "parent_1",
                "paper_id": "paper_1",
                "chunk_index": 0,
                "section_path": "1 Intro",
                "section_type": "content",
                "content_type": "text",
                "page_start": 1,
                "page_end": 1,
                "token_count": 10,
                "char_start": None,
                "char_end": None,
                "docling_ref_ids": "[]",
                "embedding_text_hash": "hash",
                "embedding_model": "",
                "created_at": "2026-01-01T00:00:00+00:00",
                "chunk_text": "hello world",
            }
        ]
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.points = []

        def upsert(self, collection_name, points, wait) -> None:  # noqa: ANN001
            assert wait
            assert collection_name == settings.qdrant_collection
            self.points = points

    fake_client = _FakeClient()
    monkeypatch.setattr("agentic_rag.ingest.indexing.get_client", lambda url: fake_client)  # noqa: ARG005
    monkeypatch.setattr(
        "agentic_rag.ingest.indexing.ensure_child_chunk_collection",
        lambda client, collection_name: None,  # noqa: ARG005
    )

    class _FakeEmbedder:
        device = "cpu"

        def encode(
            self, texts: list[str], batch_size: int = 32, max_length: int = 2048
        ) -> EmbeddingBatch:  # noqa: ARG002
            assert texts == ["hello world"]
            return EmbeddingBatch(
                dense_vectors=[[0.1] * 1024], model_name="BAAI/bge-m3", device="cpu"
            )

    indexer = ChildChunkIndexer(settings=settings, embedder=_FakeEmbedder())
    summary = indexer.index_unembedded_chunks(limit=100, batch_size=8)

    assert summary.total_chunks == 1
    assert summary.pending_chunks == 1
    assert summary.selected_chunks == 1
    assert summary.indexed_chunks == 1
    assert summary.model_name == "BAAI/bge-m3"
    assert len(fake_client.points) == 1


def test_child_chunk_indexing_idempotent_skip_and_force(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()

    initialize_storage(settings=settings, init_qdrant=False)
    store = SQLiteStore(settings.app_db_path)
    paper_repo = PaperRepository(store)
    parent_repo = ParentRepository(store)
    chunk_repo = ChunkRepository(store)

    paper_repo.upsert(
        PaperRecord(
            paper_id="paper_2",
            arxiv_id="2501.00002",
            title="Agentic Paper 2",
            authors=["B"],
            abstract="retrieval memory",
            categories=["cs.AI"],
            parse_status="parsed",
        )
    )
    parent_repo.insert_many(
        [
            {
                "parent_id": "parent_2",
                "paper_id": "paper_2",
                "section_path": "1 Intro",
                "section_heading": "Intro",
                "section_type": "content",
                "section_level": 1,
                "parent_index": 0,
                "page_start": 1,
                "page_end": 1,
                "token_count": 10,
                "char_count": 50,
                "content_types": '["text"]',
                "child_chunk_ids": '["chunk_2"]',
                "text_hash": "y",
                "parent_text": "text",
            }
        ]
    )
    chunk_repo.insert_many(
        [
            {
                "chunk_id": "chunk_2",
                "parent_id": "parent_2",
                "paper_id": "paper_2",
                "chunk_index": 0,
                "section_path": "1 Intro",
                "section_type": "content",
                "content_type": "text",
                "page_start": 1,
                "page_end": 1,
                "token_count": 10,
                "char_start": None,
                "char_end": None,
                "docling_ref_ids": "[]",
                "embedding_text_hash": "same_hash",
                "embedding_model": "BAAI/bge-m3",
                "embedding_config_hash": "config_hash",
                "indexed_embedding_text_hash": "same_hash",
                "last_indexed_at": "2026-01-01T00:00:00+00:00",
                "created_at": "2026-01-01T00:00:00+00:00",
                "chunk_text": "hello world",
            }
        ]
    )

    class _NoopClient:
        def __init__(self) -> None:
            self.calls = 0

        def upsert(self, collection_name, points, wait) -> None:  # noqa: ANN001
            del collection_name, points, wait
            self.calls += 1

    fake_client = _NoopClient()
    monkeypatch.setattr("agentic_rag.ingest.indexing.get_client", lambda url: fake_client)  # noqa: ARG005
    monkeypatch.setattr(
        "agentic_rag.ingest.indexing.ensure_child_chunk_collection",
        lambda client, collection_name: None,  # noqa: ARG005
    )

    class _FakeEmbedder:
        device = "cpu"

        def encode(
            self, texts: list[str], batch_size: int = 32, max_length: int = 2048
        ) -> EmbeddingBatch:  # noqa: ARG002
            del texts
            return EmbeddingBatch(
                dense_vectors=[[0.1] * 1024], model_name="BAAI/bge-m3", device="cpu"
            )

    indexer = ChildChunkIndexer(settings=settings, embedder=_FakeEmbedder())
    monkeypatch.setattr(indexer, "_embedding_config_hash", lambda: "config_hash")

    idempotent = indexer.index_unembedded_chunks(limit=100, batch_size=8, force=False)
    assert idempotent.pending_chunks == 0
    assert idempotent.indexed_chunks == 0

    forced = indexer.index_unembedded_chunks(limit=100, batch_size=8, force=True)
    assert forced.pending_chunks == 1
    assert forced.indexed_chunks == 1
