from __future__ import annotations

from src.storage.base import SearchResult


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for doc_id in relevant_ids if doc_id in top_k)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> float:
    """1 / rank of first relevant result. Returns 0 if no relevant result found."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def average_score(results: list[SearchResult]) -> float:
    """Mean cosine similarity score across results."""
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)
