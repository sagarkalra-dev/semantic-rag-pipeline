from __future__ import annotations

import numpy as np

from data.corpus import Document
from src.embedding.local_embedder import LocalEmbeddingModel
from src.retrieval.ai_enhanced_search import AIEnhancedSearch
from src.retrieval.pipeline import RAGPipeline
from src.retrieval.query_expander import LocalQueryExpander
from src.retrieval.raw_vector_search import RawVectorSearch
from src.storage.faiss_store import FAISSVectorStore
from tests.conftest import TEST_DOCUMENTS


def _ingest(
    embedder: LocalEmbeddingModel,
    store: FAISSVectorStore,
    documents: list[Document],
) -> None:
    texts = [d.content for d in documents]
    ids = [d.id for d in documents]
    embs = embedder.get_embeddings(texts)
    matrix = np.array([e.to_numpy() for e in embs], dtype=np.float32)
    store.add(ids, matrix)


class TestRawVectorSearch:
    def test_returns_retrieval_result(
        self, embedder: LocalEmbeddingModel, store: FAISSVectorStore
    ) -> None:
        _ingest(embedder, store, TEST_DOCUMENTS)
        strategy = RawVectorSearch(embedder, store)
        result = strategy.retrieve("scaling", top_k=2)
        assert result.strategy_name == "raw_vector_search"
        assert result.expanded_query is None
        assert len(result.results) == 2
        assert result.latency_ms > 0


class TestAIEnhancedSearch:
    def test_populates_expanded_query(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        _ingest(embedder, store, TEST_DOCUMENTS)
        strategy = AIEnhancedSearch(embedder, store, expander)
        result = strategy.retrieve("scaling", top_k=2)
        assert result.strategy_name == "ai_enhanced_search"
        assert result.expanded_query is not None
        assert len(result.expanded_query) > len("scaling")


class TestRAGPipeline:
    def test_ingest_populates_store(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        pipeline = RAGPipeline(embedder, store, expander)
        count = pipeline.ingest(TEST_DOCUMENTS)
        assert count == 3
        assert store.count == 3

    def test_search_strategy_a(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        pipeline = RAGPipeline(embedder, store, expander)
        pipeline.ingest(TEST_DOCUMENTS)
        result = pipeline.search("caching data", strategy="A", top_k=2)
        assert result.strategy_name == "raw_vector_search"

    def test_search_strategy_b(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        pipeline = RAGPipeline(embedder, store, expander)
        pipeline.ingest(TEST_DOCUMENTS)
        result = pipeline.search("caching data", strategy="B", top_k=2)
        assert result.strategy_name == "ai_enhanced_search"
