from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BenchmarkQuery:
    query: str
    description: str
    expected_relevant_ids: list[str]
    difficulty: Literal["simple", "moderate", "complex"]


BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="How does the system handle peak load?",
        description="Tests scaling + load balancing retrieval with indirect phrasing",
        expected_relevant_ids=["doc_scaling", "doc_load_balancing", "doc_rate_limiting"],
        difficulty="moderate",
    ),
    BenchmarkQuery(
        query="What happens when a downstream service becomes unresponsive?",
        description="Tests fault tolerance retrieval without using exact terms",
        expected_relevant_ids=["doc_circuit_breaker", "doc_service_mesh"],
        difficulty="complex",
    ),
    BenchmarkQuery(
        query="Explain the caching strategy and how stale data is avoided",
        description="Direct terminology match — should be easy for both strategies",
        expected_relevant_ids=["doc_caching"],
        difficulty="simple",
    ),
    BenchmarkQuery(
        query="How do you deploy new versions without downtime?",
        description="Tests deployment concepts without exact terms like 'blue-green'",
        expected_relevant_ids=["doc_deployments", "doc_load_balancing"],
        difficulty="complex",
    ),
    BenchmarkQuery(
        query="What observability tools are used to detect performance degradation?",
        description="Tests monitoring/observability retrieval",
        expected_relevant_ids=["doc_observability", "doc_circuit_breaker"],
        difficulty="moderate",
    ),
]
