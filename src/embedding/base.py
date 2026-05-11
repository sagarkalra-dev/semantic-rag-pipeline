from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingResult:
    """Mirrors vertexai.language_models.TextEmbedding interface."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def to_numpy(self) -> NDArray[np.float32]:
        return np.array(self.values, dtype=np.float32)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[EmbeddingResult]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...
