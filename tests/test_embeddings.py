from pathlib import Path

import pytest

from agentic_rag.config import get_settings
from agentic_rag.ingest.embeddings import OpenAIEmbedder


def test_openai_embedder_encode_with_mocked_client(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    settings = get_settings()

    class _Item:
        def __init__(self, embedding: list[float]) -> None:
            self.embedding = embedding

    class _Response:
        def __init__(self, rows: int, dims: int) -> None:
            self.data = [_Item([0.1] * dims) for _ in range(rows)]

    class _Embeddings:
        def create(self, model, input, dimensions):  # noqa: ANN001, A002
            assert model == "text-embedding-3-small"
            assert dimensions == 1536
            return _Response(rows=len(input), dims=dimensions)

    class _Client:
        embeddings = _Embeddings()

    embedder = OpenAIEmbedder(settings=settings)
    embedder._client = _Client()  # noqa: SLF001
    batch = embedder.encode(texts=["a", "b"], batch_size=2)
    assert len(batch.dense_vectors) == 2
    assert len(batch.dense_vectors[0]) == 1536
    assert batch.model_name == "text-embedding-3-small"
    assert batch.dimensions == 1536


def test_openai_embedder_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()

    embedder = OpenAIEmbedder(settings=settings)
    with pytest.raises(RuntimeError):
        embedder.encode(texts=["a"])
