from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from tabulate import tabulate

from src.retrieval.pipeline import RAGPipeline

from .metrics import average_score, mean_reciprocal_rank, precision_at_k, recall_at_k
from .queries import BenchmarkQuery


@dataclass
class StrategyBenchmark:
    strategy_name: str
    query: str
    expanded_query: str | None
    top_results: list[dict]
    precision_at_3: float
    recall_at_3: float
    mrr: float
    avg_similarity: float
    latency_ms: float


@dataclass
class QueryComparison:
    query: str
    difficulty: str
    description: str
    strategy_a: StrategyBenchmark
    strategy_b: StrategyBenchmark


@dataclass
class BenchmarkReport:
    timestamp: str
    model_name: str
    num_documents: int
    similarity_metric: str
    comparisons: list[QueryComparison]
    summary: dict


class BenchmarkRunner:
    def __init__(self, pipeline: RAGPipeline, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._pipeline = pipeline
        self._model_name = model_name

    def _evaluate_strategy(
        self,
        query: str,
        strategy_key: str,
        benchmark_query: BenchmarkQuery,
        top_k: int,
    ) -> StrategyBenchmark:
        result = self._pipeline.search(query, strategy=strategy_key, top_k=top_k)
        retrieved_ids = [r.doc_id for r in result.results]
        relevant_ids = benchmark_query.expected_relevant_ids

        return StrategyBenchmark(
            strategy_name=result.strategy_name,
            query=result.query,
            expanded_query=result.expanded_query,
            top_results=[
                {"doc_id": r.doc_id, "score": round(r.score, 4), "rank": r.rank}
                for r in result.results
            ],
            precision_at_3=round(precision_at_k(retrieved_ids, relevant_ids, top_k), 4),
            recall_at_3=round(recall_at_k(retrieved_ids, relevant_ids, top_k), 4),
            mrr=round(mean_reciprocal_rank(retrieved_ids, relevant_ids), 4),
            avg_similarity=round(average_score(result.results), 4),
            latency_ms=round(result.latency_ms, 2),
        )

    def run(
        self, queries: list[BenchmarkQuery], top_k: int = 3
    ) -> BenchmarkReport:
        comparisons: list[QueryComparison] = []

        for bq in queries:
            strategy_a = self._evaluate_strategy(bq.query, "A", bq, top_k)
            strategy_b = self._evaluate_strategy(bq.query, "B", bq, top_k)
            comparisons.append(
                QueryComparison(
                    query=bq.query,
                    difficulty=bq.difficulty,
                    description=bq.description,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                )
            )

        summary = self._compute_summary(comparisons)

        return BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self._model_name,
            num_documents=self._pipeline._store.count,
            similarity_metric="cosine (via IndexFlatIP on L2-normalized vectors)",
            comparisons=comparisons,
            summary=summary,
        )

    def _compute_summary(self, comparisons: list[QueryComparison]) -> dict:
        n = len(comparisons)
        a_p = sum(c.strategy_a.precision_at_3 for c in comparisons) / n
        b_p = sum(c.strategy_b.precision_at_3 for c in comparisons) / n
        a_r = sum(c.strategy_a.recall_at_3 for c in comparisons) / n
        b_r = sum(c.strategy_b.recall_at_3 for c in comparisons) / n
        a_mrr = sum(c.strategy_a.mrr for c in comparisons) / n
        b_mrr = sum(c.strategy_b.mrr for c in comparisons) / n
        a_sim = sum(c.strategy_a.avg_similarity for c in comparisons) / n
        b_sim = sum(c.strategy_b.avg_similarity for c in comparisons) / n
        a_lat = sum(c.strategy_a.latency_ms for c in comparisons) / n
        b_lat = sum(c.strategy_b.latency_ms for c in comparisons) / n

        return {
            "strategy_a_avg": {
                "precision_at_3": round(a_p, 4),
                "recall_at_3": round(a_r, 4),
                "mrr": round(a_mrr, 4),
                "avg_similarity": round(a_sim, 4),
                "avg_latency_ms": round(a_lat, 2),
            },
            "strategy_b_avg": {
                "precision_at_3": round(b_p, 4),
                "recall_at_3": round(b_r, 4),
                "mrr": round(b_mrr, 4),
                "avg_similarity": round(b_sim, 4),
                "avg_latency_ms": round(b_lat, 2),
            },
        }

    def to_json(self, report: BenchmarkReport) -> str:
        def serialize(obj: object) -> object:
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)

        return json.dumps(asdict(report), indent=2, default=serialize)

    def to_table(self, report: BenchmarkReport) -> str:
        rows = []
        for comp in report.comparisons:
            a = comp.strategy_a
            b = comp.strategy_b
            top1_a = a.top_results[0]["doc_id"] if a.top_results else "—"
            top1_b = b.top_results[0]["doc_id"] if b.top_results else "—"
            score_a = a.top_results[0]["score"] if a.top_results else 0
            score_b = b.top_results[0]["score"] if b.top_results else 0

            rows.append([
                comp.query[:55],
                comp.difficulty,
                "A (Raw)",
                top1_a,
                f"{score_a:.4f}",
                f"{a.precision_at_3:.2f}",
                f"{a.recall_at_3:.2f}",
                f"{a.mrr:.2f}",
                f"{a.latency_ms:.1f}ms",
            ])
            rows.append([
                "",
                "",
                "B (AI)",
                top1_b,
                f"{score_b:.4f}",
                f"{b.precision_at_3:.2f}",
                f"{b.recall_at_3:.2f}",
                f"{b.mrr:.2f}",
                f"{b.latency_ms:.1f}ms",
            ])
        headers = [
            "Query", "Difficulty", "Strategy", "Top-1 Doc",
            "Score", "P@3", "R@3", "MRR", "Latency",
        ]
        return tabulate(rows, headers=headers, tablefmt="grid")
