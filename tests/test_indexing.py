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
        lambda client, collection_name, vector_size=1536: None,  # noqa: ARG005
    )

    class _FakeEmbedder:
        def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingBatch:  # noqa: ARG002
            assert texts == ["hello world"]
            return EmbeddingBatch(
                dense_vectors=[[0.1] * 1536],
                model_name="text-embedding-3-small",
                dimensions=1536,
            )

    indexer = ChildChunkIndexer(settings=settings, embedder=_FakeEmbedder())
    summary = indexer.index_unembedded_chunks(limit=100, batch_size=8)

    assert summary.total_chunks == 1
    assert summary.pending_chunks == 1
    assert summary.selected_chunks == 1
    assert summary.indexed_chunks == 1
    assert summary.model_name == "text-embedding-3-small"
    assert summary.dimensions == 1536
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
                "embedding_model": "text-embedding-3-small",
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
        lambda client, collection_name, vector_size=1536: None,  # noqa: ARG005
    )

    class _FakeEmbedder:
        def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingBatch:  # noqa: ARG002
            del texts
            return EmbeddingBatch(
                dense_vectors=[[0.1] * 1536],
                model_name="text-embedding-3-small",
                dimensions=1536,
            )

    indexer = ChildChunkIndexer(settings=settings, embedder=_FakeEmbedder())
    monkeypatch.setattr(indexer, "_embedding_config_hash", lambda: "config_hash")

    idempotent = indexer.index_unembedded_chunks(limit=100, batch_size=8, force=False)
    assert idempotent.pending_chunks == 0
    assert idempotent.indexed_chunks == 0

    forced = indexer.index_unembedded_chunks(limit=100, batch_size=8, force=True)
    assert forced.pending_chunks == 1
    assert forced.indexed_chunks == 1


def test_force_without_limit_runs_single_full_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    get_settings.cache_clear()
    settings = get_settings()

    class _FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def upsert(self, collection_name, points, wait) -> None:  # noqa: ANN001
            del collection_name, points, wait
            self.calls += 1

    class _FakeEmbedder:
        def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingBatch:  # noqa: ARG002
            return EmbeddingBatch(
                dense_vectors=[[0.1] * 1536 for _ in texts],
                model_name="text-embedding-3-small",
                dimensions=1536,
            )

    fake_client = _FakeClient()
    monkeypatch.setattr("agentic_rag.ingest.indexing.get_client", lambda url: fake_client)  # noqa: ARG005
    monkeypatch.setattr(
        "agentic_rag.ingest.indexing.ensure_child_chunk_collection",
        lambda client, collection_name, vector_size=1536: None,  # noqa: ARG005
    )

    indexer = ChildChunkIndexer(settings=settings, embedder=_FakeEmbedder())
    monkeypatch.setattr(indexer.chunk_repo, "count_chunks_total", lambda: 1)
    monkeypatch.setattr(
        indexer.chunk_repo,
        "count_chunks_pending",
        lambda model_name, config_hash: 1,
    )

    calls = {"fetch": 0}
    sample_row = {
        "chunk_id": "chunk_1",
        "parent_id": "parent_1",
        "paper_id": "paper_1",
        "arxiv_id": "2501.00001",
        "title": "Title",
        "primary_category": "cs.AI",
        "categories": '["cs.AI"]',
        "published_at": None,
        "updated_at": None,
        "section_path": "1 Intro",
        "section_type": "content",
        "content_type": "text",
        "page_start": 1,
        "page_end": 1,
        "chunk_index": 0,
        "token_count": 10,
        "chunk_text": "hello world",
    }

    def _fetch_once(**kwargs):  # noqa: ANN003
        del kwargs
        if calls["fetch"] > 0:
            raise AssertionError("force+limit=None should not fetch more than once")
        calls["fetch"] += 1
        return [sample_row]

    monkeypatch.setattr(indexer.chunk_repo, "fetch_chunks_for_indexing", _fetch_once)
    monkeypatch.setattr(
        indexer.chunk_repo,
        "mark_embedded",
        lambda chunk_ids, model_name, config_hash: None,  # noqa: ARG005
    )

    summary = indexer.index_unembedded_chunks(limit=None, batch_size=8, force=True)
    assert summary.selected_chunks == 1
    assert summary.indexed_chunks == 1
    assert fake_client.calls == 1


def test_indexing_config_hash_uses_provider_model_dimensions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    get_settings.cache_clear()
    settings = get_settings()

    indexer = ChildChunkIndexer(settings=settings, embedder=None)
    h1 = indexer._embedding_config_hash()  # noqa: SLF001
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    get_settings.cache_clear()
    settings2 = get_settings()
    indexer2 = ChildChunkIndexer(settings=settings2, embedder=None)
    h2 = indexer2._embedding_config_hash()  # noqa: SLF001
    assert h1 != h2
