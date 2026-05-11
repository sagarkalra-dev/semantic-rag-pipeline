from __future__ import annotations

import pytest

from data.corpus import Document
from src.embedding.local_embedder import LocalEmbeddingModel
from src.retrieval.query_expander import LocalQueryExpander
from src.storage.faiss_store import FAISSVectorStore


TEST_DOCUMENTS = [
    Document(
        id="test_scaling",
        title="Scaling",
        content="The system uses horizontal scaling with autoscaling policies to handle increased load.",
        tags=["scaling"],
    ),
    Document(
        id="test_caching",
        title="Caching",
        content="A distributed Redis cache stores frequently accessed data with TTL-based expiry.",
        tags=["caching"],
    ),
    Document(
        id="test_monitoring",
        title="Monitoring",
        content="Prometheus metrics and distributed tracing provide observability into system health.",
        tags=["monitoring"],
    ),
]


@pytest.fixture(scope="session")
def embedder() -> LocalEmbeddingModel:
    return LocalEmbeddingModel()


@pytest.fixture
def store(embedder: LocalEmbeddingModel) -> FAISSVectorStore:
    return FAISSVectorStore(dimension=embedder.dimension)


@pytest.fixture
def expander() -> LocalQueryExpander:
    return LocalQueryExpander()


@pytest.fixture
def test_documents() -> list[Document]:
    return TEST_DOCUMENTS
