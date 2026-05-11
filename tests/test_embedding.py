from __future__ import annotations

import numpy as np

from src.embedding.local_embedder import LocalEmbeddingModel


class TestLocalEmbeddingModel:
    def test_returns_correct_count(self, embedder: LocalEmbeddingModel) -> None:
        results = embedder.get_embeddings(["hello", "world"])
        assert len(results) == 2

    def test_embedding_dimension(self, embedder: LocalEmbeddingModel) -> None:
        results = embedder.get_embeddings(["test sentence"])
        assert len(results[0].values) == embedder.dimension

    def test_embeddings_are_normalized(self, embedder: LocalEmbeddingModel) -> None:
        results = embedder.get_embeddings(["test normalization"])
        vec = results[0].to_numpy()
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_closer(self, embedder: LocalEmbeddingModel) -> None:
        results = embedder.get_embeddings([
            "distributed system scaling",
            "horizontal autoscaling architecture",
            "chocolate cake recipe",
        ])
        v0, v1, v2 = [r.to_numpy() for r in results]
        sim_related = float(np.dot(v0, v1))
        sim_unrelated = float(np.dot(v0, v2))
        assert sim_related > sim_unrelated

    def test_to_numpy_dtype(self, embedder: LocalEmbeddingModel) -> None:
        results = embedder.get_embeddings(["dtype check"])
        vec = results[0].to_numpy()
        assert vec.dtype == np.float32
