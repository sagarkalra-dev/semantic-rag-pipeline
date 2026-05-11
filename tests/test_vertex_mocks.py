"""
Tests demonstrating how to mock the Vertex AI SDK for CI/CD without GCP credentials.

These tests use unittest.mock to inject fake modules into sys.modules,
allowing imports of vertexai.language_models and vertexai.generative_models
to resolve against mocks rather than requiring google-cloud-aiplatform.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


class TestVertexEmbeddingModelMock:
    def test_mock_text_embedding_model(self) -> None:
        """Mock vertexai.language_models.TextEmbeddingModel.from_pretrained."""
        mock_vertexai = MagicMock()
        mock_language_models = MagicMock()

        mock_model_instance = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 384
        mock_model_instance.get_embeddings.return_value = [mock_embedding]
        mock_language_models.TextEmbeddingModel.from_pretrained.return_value = (
            mock_model_instance
        )
        mock_vertexai.language_models = mock_language_models

        with patch.dict(sys.modules, {
            "vertexai": mock_vertexai,
            "vertexai.language_models": mock_language_models,
        }):
            # Force reimport to pick up the mocked module
            if "src.embedding.vertex_adapter" in sys.modules:
                del sys.modules["src.embedding.vertex_adapter"]
            from src.embedding.vertex_adapter import VertexEmbeddingAdapter

            adapter = VertexEmbeddingAdapter(model_name="text-embedding-005")
            results = adapter.get_embeddings(["test query"])

            assert len(results) == 1
            assert len(results[0].values) == 384
            mock_model_instance.get_embeddings.assert_called_once_with(
                ["test query"]
            )

    def test_mock_batch_embeddings(self) -> None:
        """Mock batch embedding of multiple texts."""
        mock_vertexai = MagicMock()
        mock_language_models = MagicMock()

        mock_model_instance = MagicMock()
        mock_emb_1 = MagicMock()
        mock_emb_1.values = [0.2] * 384
        mock_emb_2 = MagicMock()
        mock_emb_2.values = [0.3] * 384
        mock_model_instance.get_embeddings.return_value = [mock_emb_1, mock_emb_2]
        mock_language_models.TextEmbeddingModel.from_pretrained.return_value = (
            mock_model_instance
        )
        mock_vertexai.language_models = mock_language_models

        with patch.dict(sys.modules, {
            "vertexai": mock_vertexai,
            "vertexai.language_models": mock_language_models,
        }):
            if "src.embedding.vertex_adapter" in sys.modules:
                del sys.modules["src.embedding.vertex_adapter"]
            from src.embedding.vertex_adapter import VertexEmbeddingAdapter

            adapter = VertexEmbeddingAdapter()
            results = adapter.get_embeddings(["text one", "text two"])

            assert len(results) == 2
            assert results[0].values == [0.2] * 384
            assert results[1].values == [0.3] * 384


class TestVertexGenerativeModelMock:
    def test_mock_generative_model(self) -> None:
        """Mock vertexai.generative_models.GenerativeModel for query expansion."""
        mock_vertexai = MagicMock()
        mock_generative_models = MagicMock()

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "How does horizontal scaling and autoscaling handle increased "
            "traffic load and request spikes in distributed systems?"
        )
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_models.GenerativeModel.return_value = mock_model_instance
        mock_vertexai.generative_models = mock_generative_models

        with patch.dict(sys.modules, {
            "vertexai": mock_vertexai,
            "vertexai.generative_models": mock_generative_models,
        }):
            if "src.retrieval.query_expander" in sys.modules:
                del sys.modules["src.retrieval.query_expander"]
            from src.retrieval.query_expander import VertexQueryExpander

            expander = VertexQueryExpander(model_name="gemini-2.5-flash-001")
            result = expander.expand("How does scaling work?")

            assert "horizontal scaling" in result.lower()
            mock_model_instance.generate_content.assert_called_once_with(
                "How does scaling work?"
            )

    def test_mock_system_instruction_passed(self) -> None:
        """Verify that system_instruction is passed to GenerativeModel."""
        mock_vertexai = MagicMock()
        mock_generative_models = MagicMock()
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "expanded query"
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_models.GenerativeModel.return_value = mock_model_instance
        mock_vertexai.generative_models = mock_generative_models

        with patch.dict(sys.modules, {
            "vertexai": mock_vertexai,
            "vertexai.generative_models": mock_generative_models,
        }):
            if "src.retrieval.query_expander" in sys.modules:
                del sys.modules["src.retrieval.query_expander"]
            from src.retrieval.query_expander import VertexQueryExpander

            VertexQueryExpander(model_name="gemini-2.5-flash-001")

            call_kwargs = mock_generative_models.GenerativeModel.call_args
            assert "system_instruction" in call_kwargs.kwargs
