from __future__ import annotations

import time

from src.embedding.base import BaseEmbeddingModel
from src.storage.base import BaseVectorStore

from .base import RetrievalResult, RetrievalStrategy
from .query_expander import BaseQueryExpander


class AIEnhancedSearch(RetrievalStrategy):
    """Strategy B: Expand query via AI model, then embed and search."""

    def __init__(
        self,
        embedder: BaseEmbeddingModel,
        store: BaseVectorStore,
        expander: BaseQueryExpander,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._expander = expander

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()
        expanded = self._expander.expand(query)
        embedding = self._embedder.get_embeddings([expanded])[0].to_numpy()
        results = self._store.search(embedding, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            strategy_name=self.name,
            results=results,
            expanded_query=expanded,
            latency_ms=elapsed,
        )

    @property
    def name(self) -> str:
        return "ai_enhanced_search"
