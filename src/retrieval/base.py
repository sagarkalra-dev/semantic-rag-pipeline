from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.storage.base import SearchResult


@dataclass
class RetrievalResult:
    query: str
    strategy_name: str
    results: list[SearchResult]
    expanded_query: str | None
    latency_ms: float


class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
