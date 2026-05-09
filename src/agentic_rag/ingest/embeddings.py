from __future__ import annotations

from dataclasses import dataclass

import torch
from FlagEmbedding import BGEM3FlagModel

from agentic_rag.config import Settings


@dataclass(frozen=True)
class EmbeddingBatch:
    dense_vectors: list[list[float]]
    model_name: str
    device: str


class BgeM3DenseEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.bge_device)
        self._model: BGEM3FlagModel | None = None
        self._model_init_error: str | None = None

    def encode(
        self, texts: list[str], batch_size: int = 16, max_length: int = 2048
    ) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(
                dense_vectors=[],
                model_name=self.settings.bge_model_name,
                device=self.device,
            )
        model = self._get_model()
        output = model.encode(
            sentences=texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        return EmbeddingBatch(
            dense_vectors=dense.tolist(),
            model_name=self.settings.bge_model_name,
            device=self.device,
        )

    def _get_model(self) -> BGEM3FlagModel:
        if self._model_init_error is not None:
            raise RuntimeError(self._model_init_error)
        if self._model is None:
            try:
                self._model = BGEM3FlagModel(
                    model_name_or_path=self.settings.bge_model_name,
                    use_fp16=self.device.startswith("cuda"),
                    devices=self.device,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            except Exception as err:  # noqa: BLE001
                self._model_init_error = (
                    f"Failed to initialize embedding model {self.settings.bge_model_name}: {err}"
                )
                raise RuntimeError(self._model_init_error) from err
        return self._model

    @staticmethod
    def _resolve_device(configured: str) -> str:
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured
