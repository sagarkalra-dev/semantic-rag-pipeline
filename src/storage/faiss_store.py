from __future__ import annotations

import faiss
import numpy as np
from numpy.typing import NDArray

from .base import BaseVectorStore, SearchResult


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS vector store using IndexFlatIP for cosine similarity.

    Uses inner product on L2-normalized vectors, which equals cosine similarity.
    This matches Vertex AI Matching Engine's default COSINE distance type.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._doc_ids: list[str] = []

    def add(self, doc_ids: list[str], embeddings: NDArray[np.float32]) -> None:
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Expected {self._dimension}-dim vectors, got {embeddings.shape[1]}"
            )
        vectors = embeddings.copy()
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._doc_ids.extend(doc_ids)

    def search(
        self, query_embedding: NDArray[np.float32], top_k: int = 5
    ) -> list[SearchResult]:
        if self.count == 0:
            return []
        query = query_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, self.count))
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    doc_id=self._doc_ids[idx],
                    score=float(score),
                    rank=rank + 1,
                )
            )
        return results

    def reset(self) -> None:
        self._index.reset()
        self._doc_ids.clear()

    @property
    def count(self) -> int:
        return self._index.ntotal
