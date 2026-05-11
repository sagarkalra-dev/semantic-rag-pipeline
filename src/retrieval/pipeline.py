from __future__ import annotations

from typing import Literal

import numpy as np

from data.corpus import Document
from src.embedding.base import BaseEmbeddingModel
from src.storage.base import BaseVectorStore

from .ai_enhanced_search import AIEnhancedSearch
from .base import RetrievalResult, RetrievalStrategy
from .query_expander import BaseQueryExpander
from .raw_vector_search import RawVectorSearch


class RAGPipeline:
    """Orchestrates document ingestion and retrieval across both strategies."""

    def __init__(
        self,
        embedder: BaseEmbeddingModel,
        store: BaseVectorStore,
        expander: BaseQueryExpander,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._strategy_a = RawVectorSearch(embedder, store)
        self._strategy_b = AIEnhancedSearch(embedder, store, expander)

    def ingest(self, documents: list[Document]) -> int:
        texts = [doc.content for doc in documents]
        doc_ids = [doc.id for doc in documents]
        embeddings_list = self._embedder.get_embeddings(texts)
        matrix = np.array(
            [e.to_numpy() for e in embeddings_list], dtype=np.float32
        )
        self._store.add(doc_ids, matrix)
        return len(documents)

    def search(
        self,
        query: str,
        strategy: Literal["A", "B"] = "A",
        top_k: int = 5,
    ) -> RetrievalResult:
        if strategy == "A":
            return self._strategy_a.retrieve(query, top_k)
        elif strategy == "B":
            return self._strategy_b.retrieve(query, top_k)
        else:
            raise ValueError(f"Unknown strategy '{strategy}', expected 'A' or 'B'")

    @property
    def strategies(self) -> dict[str, RetrievalStrategy]:
        return {"A": self._strategy_a, "B": self._strategy_b}
