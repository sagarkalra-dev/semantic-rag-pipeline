from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchResult:
    doc_id: str
    score: float
    rank: int


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, doc_ids: list[str], embeddings: NDArray[np.float32]) -> None:
        ...

    @abstractmethod
    def search(
        self, query_embedding: NDArray[np.float32], top_k: int = 5
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        ...
