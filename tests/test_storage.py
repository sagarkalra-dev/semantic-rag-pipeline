from __future__ import annotations

import numpy as np

from src.embedding.local_embedder import LocalEmbeddingModel
from src.storage.faiss_store import FAISSVectorStore


class TestFAISSVectorStore:
    def test_add_and_search(
        self, embedder: LocalEmbeddingModel, store: FAISSVectorStore
    ) -> None:
        texts = ["scaling systems", "caching data", "monitoring health"]
        embeddings = embedder.get_embeddings(texts)
        matrix = np.array([e.to_numpy() for e in embeddings], dtype=np.float32)
        store.add(["doc1", "doc2", "doc3"], matrix)

        query_emb = embedder.get_embeddings(["autoscaling"])[0].to_numpy()
        results = store.search(query_emb, top_k=2)
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_results_sorted_by_score(
        self, embedder: LocalEmbeddingModel, store: FAISSVectorStore
    ) -> None:
        texts = ["apples", "oranges", "bananas"]
        embeddings = embedder.get_embeddings(texts)
        matrix = np.array([e.to_numpy() for e in embeddings], dtype=np.float32)
        store.add(["a", "b", "c"], matrix)

        query_emb = embedder.get_embeddings(["fruit"])[0].to_numpy()
        results = store.search(query_emb, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reset(
        self, embedder: LocalEmbeddingModel, store: FAISSVectorStore
    ) -> None:
        emb = embedder.get_embeddings(["test"])[0].to_numpy()
        store.add(["doc1"], emb.reshape(1, -1))
        assert store.count == 1
        store.reset()
        assert store.count == 0

    def test_count(self, store: FAISSVectorStore) -> None:
        assert store.count == 0

    def test_search_empty_store(
        self, embedder: LocalEmbeddingModel, store: FAISSVectorStore
    ) -> None:
        query_emb = embedder.get_embeddings(["anything"])[0].to_numpy()
        results = store.search(query_emb, top_k=3)
        assert results == []

    def test_top_k_larger_than_store(
        self, embedder: LocalEmbeddingModel, store: FAISSVectorStore
    ) -> None:
        emb = embedder.get_embeddings(["single doc"])[0].to_numpy()
        store.add(["only_one"], emb.reshape(1, -1))
        results = store.search(emb, top_k=10)
        assert len(results) == 1
