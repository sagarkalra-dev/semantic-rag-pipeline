from __future__ import annotations

import time

from src.embedding.base import BaseEmbeddingModel
from src.storage.base import BaseVectorStore

from .base import RetrievalResult, RetrievalStrategy


class RawVectorSearch(RetrievalStrategy):
    """Strategy A: Embed the query as-is and search."""

    def __init__(
        self, embedder: BaseEmbeddingModel, store: BaseVectorStore
    ) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()
        embedding = self._embedder.get_embeddings([query])[0].to_numpy()
        results = self._store.search(embedding, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            strategy_name=self.name,
            results=results,
            expanded_query=None,
            latency_ms=elapsed,
        )

    @property
    def name(self) -> str:
        return "raw_vector_search"
