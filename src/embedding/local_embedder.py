from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingModel, EmbeddingResult


class LocalEmbeddingModel(BaseEmbeddingModel):
    """Local embedding using sentence-transformers (simulates textembedding-gecko)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_embedding_dimension()

    def get_embeddings(self, texts: list[str]) -> list[EmbeddingResult]:
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return [EmbeddingResult(values=vec.tolist()) for vec in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension
