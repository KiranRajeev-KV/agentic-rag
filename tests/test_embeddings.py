from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.ingest.embeddings import BgeM3DenseEmbedder


def test_bge_embedder_device_policy_and_encode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("BGE_DEVICE", "auto")
    get_settings.cache_clear()
    settings = get_settings()

    monkeypatch.setattr("agentic_rag.ingest.embeddings.torch.cuda.is_available", lambda: False)

    class _Vec:
        def __init__(self, rows: int, dim: int) -> None:
            self._rows = rows
            self._dim = dim

        def tolist(self) -> list[list[float]]:
            return [[0.0] * self._dim for _ in range(self._rows)]

    class _FakeModel:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

        def encode(self, sentences, **kwargs):  # noqa: ANN001, ANN003
            del kwargs
            return {"dense_vecs": _Vec(rows=len(sentences), dim=1024)}

    monkeypatch.setattr("agentic_rag.ingest.embeddings.BGEM3FlagModel", _FakeModel)

    embedder = BgeM3DenseEmbedder(settings=settings)
    batch = embedder.encode(texts=["a", "b"], batch_size=2)
    assert embedder.device == "cpu"
    assert len(batch.dense_vectors) == 2
    assert len(batch.dense_vectors[0]) == 1024
