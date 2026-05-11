"""Entry point: run the RAG benchmark comparison."""
from __future__ import annotations

import json
from pathlib import Path

from data.corpus import CORPUS_DOCUMENTS
from src.benchmark.queries import BENCHMARK_QUERIES
from src.benchmark.runner import BenchmarkRunner
from src.embedding.local_embedder import LocalEmbeddingModel
from src.retrieval.pipeline import RAGPipeline
from src.retrieval.query_expander import LocalQueryExpander
from src.storage.faiss_store import FAISSVectorStore


def main() -> None:
    print("Initializing embedding model...")
    embedder = LocalEmbeddingModel()
    store = FAISSVectorStore(dimension=embedder.dimension)
    expander = LocalQueryExpander()
    pipeline = RAGPipeline(embedder=embedder, store=store, expander=expander)

    print(f"Ingesting {len(CORPUS_DOCUMENTS)} documents...")
    count = pipeline.ingest(CORPUS_DOCUMENTS)
    print(f"Ingested {count} documents ({embedder.dimension}-dim embeddings)\n")

    runner = BenchmarkRunner(pipeline)
    report = runner.run(BENCHMARK_QUERIES, top_k=3)

    print("=" * 80)
    print("BENCHMARK RESULTS: Strategy A (Raw Vector Search) vs Strategy B (AI-Enhanced)")
    print("=" * 80)
    print()
    print(runner.to_table(report))
    print()

    print("\n--- Aggregate Summary ---")
    summary = report.summary
    print(f"  Strategy A avg: P@3={summary['strategy_a_avg']['precision_at_3']:.4f}  "
          f"R@3={summary['strategy_a_avg']['recall_at_3']:.4f}  "
          f"MRR={summary['strategy_a_avg']['mrr']:.4f}  "
          f"Latency={summary['strategy_a_avg']['avg_latency_ms']:.1f}ms")
    print(f"  Strategy B avg: P@3={summary['strategy_b_avg']['precision_at_3']:.4f}  "
          f"R@3={summary['strategy_b_avg']['recall_at_3']:.4f}  "
          f"MRR={summary['strategy_b_avg']['mrr']:.4f}  "
          f"Latency={summary['strategy_b_avg']['avg_latency_ms']:.1f}ms")

    json_path = Path("benchmark_output.json")
    json_path.write_text(runner.to_json(report))
    print(f"\nJSON report saved to {json_path}")


if __name__ == "__main__":
    main()
