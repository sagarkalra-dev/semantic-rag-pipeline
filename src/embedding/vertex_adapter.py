from __future__ import annotations

from .base import BaseEmbeddingModel, EmbeddingResult


class VertexEmbeddingAdapter(BaseEmbeddingModel):
    """
    Production adapter for Vertex AI TextEmbeddingModel (legacy SDK).

    NOTE: vertexai.language_models was deprecated on June 24, 2025 and will be
    removed on June 24, 2026. This adapter exists for mock testing as required
    by the assessment spec. For new production code, use GenAIEmbeddingAdapter
    below which uses the recommended google-genai SDK.

    Usage (requires google-cloud-aiplatform):
        import vertexai
        vertexai.init(project="my-project", location="us-central1")
        model = VertexEmbeddingAdapter(model_name="text-embedding-005")
    """

    def __init__(self, model_name: str = "text-embedding-005") -> None:
        from vertexai.language_models import TextEmbeddingModel

        self._model = TextEmbeddingModel.from_pretrained(model_name)
        self._dimension = 768

    def get_embeddings(self, texts: list[str]) -> list[EmbeddingResult]:
        vertex_embeddings = self._model.get_embeddings(texts)
        return [EmbeddingResult(values=e.values) for e in vertex_embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class GenAIEmbeddingAdapter(BaseEmbeddingModel):
    """
    Production adapter using the recommended Google Gen AI SDK (google-genai).

    This is the forward-looking replacement for VertexEmbeddingAdapter.

    Usage (requires google-genai):
        pip install google-genai
        model = GenAIEmbeddingAdapter(project="my-project", location="us-central1")
    """

    def __init__(
        self,
        project: str = "my-project",
        location: str = "us-central1",
        model_name: str = "gemini-embedding-001",
    ) -> None:
        from google import genai

        self._client = genai.Client(
            vertexai=True, project=project, location=location
        )
        self._model_name = model_name
        self._dimension = 768

    def get_embeddings(self, texts: list[str]) -> list[EmbeddingResult]:
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        return [
            EmbeddingResult(values=emb.values) for emb in result.embeddings
        ]

    @property
    def dimension(self) -> int:
        return self._dimension
