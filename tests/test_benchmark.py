from __future__ import annotations

import json

import numpy as np

from data.corpus import Document
from src.benchmark.metrics import (
    average_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.benchmark.queries import BENCHMARK_QUERIES
from src.benchmark.runner import BenchmarkRunner
from src.embedding.local_embedder import LocalEmbeddingModel
from src.retrieval.pipeline import RAGPipeline
from src.retrieval.query_expander import LocalQueryExpander
from src.storage.base import SearchResult
from src.storage.faiss_store import FAISSVectorStore
from tests.conftest import TEST_DOCUMENTS


class TestMetrics:
    def test_precision_at_k(self) -> None:
        assert precision_at_k(["a", "b", "c"], ["a", "c"], k=3) == 2 / 3
        assert precision_at_k(["a", "b", "c"], ["d"], k=3) == 0.0
        assert precision_at_k([], ["a"], k=3) == 0.0

    def test_recall_at_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
        assert recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5
        assert recall_at_k(["a"], [], k=3) == 0.0

    def test_mrr(self) -> None:
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
        assert mean_reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
        assert mean_reciprocal_rank(["a", "b", "c"], ["d"]) == 0.0

    def test_average_score(self) -> None:
        results = [
            SearchResult("a", 0.8, 1),
            SearchResult("b", 0.6, 2),
        ]
        assert average_score(results) == 0.7
        assert average_score([]) == 0.0


class TestBenchmarkRunner:
    def test_run_produces_report(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        pipeline = RAGPipeline(embedder, store, expander)
        pipeline.ingest(TEST_DOCUMENTS)
        runner = BenchmarkRunner(pipeline)
        # Use only first 2 queries for speed
        report = runner.run(BENCHMARK_QUERIES[:2], top_k=3)
        assert len(report.comparisons) == 2
        assert report.model_name == "all-MiniLM-L6-v2"

    def test_to_json_valid(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        pipeline = RAGPipeline(embedder, store, expander)
        pipeline.ingest(TEST_DOCUMENTS)
        runner = BenchmarkRunner(pipeline)
        report = runner.run(BENCHMARK_QUERIES[:1], top_k=3)
        json_str = runner.to_json(report)
        parsed = json.loads(json_str)
        assert "comparisons" in parsed
        assert "summary" in parsed

    def test_to_table_nonempty(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        pipeline = RAGPipeline(embedder, store, expander)
        pipeline.ingest(TEST_DOCUMENTS)
        runner = BenchmarkRunner(pipeline)
        report = runner.run(BENCHMARK_QUERIES[:1], top_k=3)
        table = runner.to_table(report)
        assert len(table) > 0
        assert "Strategy" in table

    def test_metrics_in_range(
        self,
        embedder: LocalEmbeddingModel,
        store: FAISSVectorStore,
        expander: LocalQueryExpander,
    ) -> None:
        pipeline = RAGPipeline(embedder, store, expander)
        pipeline.ingest(TEST_DOCUMENTS)
        runner = BenchmarkRunner(pipeline)
        report = runner.run(BENCHMARK_QUERIES[:1], top_k=3)
        comp = report.comparisons[0]
        for s in [comp.strategy_a, comp.strategy_b]:
            assert 0.0 <= s.precision_at_3 <= 1.0
            assert 0.0 <= s.recall_at_3 <= 1.0
            assert 0.0 <= s.mrr <= 1.0
            assert s.latency_ms > 0
