from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from agentic_rag.config import Settings


@dataclass(frozen=True)
class EmbeddingBatch:
    dense_vectors: list[list[float]]
    model_name: str
    dimensions: int


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(
                dense_vectors=[],
                model_name=self.settings.embedding_model,
                dimensions=self.settings.embedding_dimensions,
            )
        vectors: list[list[float]] = []
        for idx in range(0, len(texts), batch_size):
            chunk = texts[idx : idx + batch_size]
            response = self._client_or_raise().embeddings.create(
                model=self.settings.embedding_model,
                input=chunk,
                dimensions=self.settings.embedding_dimensions,
            )
            vectors.extend([list(item.embedding) for item in response.data])
        return EmbeddingBatch(
            dense_vectors=vectors,
            model_name=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
        )

    def _client_or_raise(self) -> OpenAI:
        if self._client is not None:
            return self._client
        if self.settings.embedding_provider.lower() != "openai":
            raise RuntimeError(
                f"Unsupported embedding provider: {self.settings.embedding_provider}"
            )
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
        self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
