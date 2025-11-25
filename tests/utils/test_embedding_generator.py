"""
Tests for the embedding generator utility.
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from agentic_nav.utils.embedding_generator import batch_embed_documents


def create_mock_response(embeddings_list):
    """Helper function to create properly formatted mock response."""
    mock_response = Mock()
    mock_data = [{"embedding": emb} for emb in embeddings_list]
    mock_response.data = mock_data
    mock_response.__getitem__ = lambda self, key: mock_data if key == "data" else None
    return mock_response


class TestBatchEmbedDocuments:
    """Test the batch_embed_documents function."""

    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_basic(self, mock_embedding_fn):
        """Test basic embedding generation functionality."""
        # Mock embedding response
        mock_embedding_fn.return_value = create_mock_response([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])

        texts = ["first document", "second document"]
        result = batch_embed_documents(
            texts=texts,
            batch_size=2,
            embedding_model="test-model",
            api_base="http://localhost:11434"
        )

        # Verify embedding_fn was called correctly
        mock_embedding_fn.assert_called_once_with(
            model="test-model",
            input=texts,
            api_base="http://localhost:11434",
            num_ctx=2048
        )

        # Verify result is numpy array with correct shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

        # Verify the embeddings are normalized (unit vectors)
        # The function normalizes embeddings, so we check the direction is correct
        expected_0_normalized = np.array([0.1, 0.2, 0.3]) / np.linalg.norm([0.1, 0.2, 0.3])
        expected_1_normalized = np.array([0.4, 0.5, 0.6]) / np.linalg.norm([0.4, 0.5, 0.6])

        np.testing.assert_allclose(result[0], expected_0_normalized, rtol=1e-5)
        np.testing.assert_allclose(result[1], expected_1_normalized, rtol=1e-5)

    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_with_batching(self, mock_embedding_fn):
        """Test embedding with multiple batches."""
        # Mock responses for each batch
        mock_embedding_fn.side_effect = [
            create_mock_response([[0.1, 0.2]]),
            create_mock_response([[0.3, 0.4]])
        ]

        texts = ["doc1", "doc2"]
        result = batch_embed_documents(
            texts=texts,
            batch_size=1,  # Force multiple batches
            embedding_model="test-model",
            api_base="http://localhost:11434"
        )

        # Verify embedding_fn was called twice
        assert mock_embedding_fn.call_count == 2

        # Check first call
        first_call = mock_embedding_fn.call_args_list[0]
        assert first_call[1]['input'] == ["doc1"]

        # Check second call
        second_call = mock_embedding_fn.call_args_list[1]
        assert second_call[1]['input'] == ["doc2"]

        # Verify combined result (embeddings are normalized)
        assert result.shape == (2, 2)
        expected_0_normalized = np.array([0.1, 0.2]) / np.linalg.norm([0.1, 0.2])
        expected_1_normalized = np.array([0.3, 0.4]) / np.linalg.norm([0.3, 0.4])
        np.testing.assert_allclose(result[0], expected_0_normalized, rtol=1e-5)
        np.testing.assert_allclose(result[1], expected_1_normalized, rtol=1e-5)

    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_with_none_values(self, mock_embedding_fn, caplog):
        """Test handling of None values in input texts."""
        mock_embedding_fn.return_value = create_mock_response([
            [0.1, 0.2],
            [0.3, 0.4]
        ])

        texts = ["valid document", None]

        batch_embed_documents(texts=texts, batch_size=2, api_base="http://localhost:11434")  # Process both items in single batch

        # Verify warning was logged
        assert "WARNING: Detected documents with 'None' values" in caplog.text

        # Verify None was replaced with empty string
        call_args = mock_embedding_fn.call_args
        assert call_args[1]['input'] == ["valid document", ""]

    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_error_fallback(self, mock_embedding_fn, caplog):
        """Test fallback to single sample processing on batch error."""
        # First call (batch) raises exception
        mock_embedding_fn.side_effect = [
            Exception("Batch failed"),
            # Individual calls succeed
            create_mock_response([[0.1, 0.2]]),
            create_mock_response([[0.3, 0.4]])
        ]

        texts = ["doc1", "doc2"]
        result = batch_embed_documents(texts=texts, batch_size=2, api_base="http://localhost:11434")

        # Verify error was logged
        assert "Error during embedding batch" in caplog.text
        assert "Falling back to single sample processing" in caplog.text

        # Verify fallback individual calls were made
        assert mock_embedding_fn.call_count == 3  # 1 failed batch + 2 individual

        # Verify result is still correct
        assert result.shape == (2, 2)

    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_bad_request_fallback(self, mock_embedding_fn, caplog):
        """Test handling of BadRequestError during individual processing."""
        from litellm import BadRequestError

        # Batch call fails, individual calls have mixed results
        mock_embedding_fn.side_effect = [
            Exception("Batch failed"),
            BadRequestError("Bad request", model="test-model", llm_provider="test"),  # First individual fails
            create_mock_response([[0.3, 0.4]])  # Second succeeds
        ]

        texts = ["problematic doc", "good doc"]
        result = batch_embed_documents(texts=texts, batch_size=2, api_base="http://localhost:11434")

        # Should handle BadRequestError gracefully
        assert "Encountered error processing paper" in caplog.text

        # Result should only have the successful embedding (failed one is skipped)
        assert result.shape == (1, 2)
        expected_normalized = np.array([0.3, 0.4]) / np.linalg.norm([0.3, 0.4])
        np.testing.assert_allclose(result[0], expected_normalized, rtol=1e-5)

    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_default_params(self, mock_embedding_fn):
        """Test function with default parameters."""
        mock_embedding_fn.return_value = create_mock_response([[0.1, 0.2, 0.3]])

        batch_embed_documents(texts=["test doc"])

        # Verify default parameters were used
        call_args = mock_embedding_fn.call_args
        assert call_args[1]['model'] == "nomic-ai/nomic-embed-text-v1.5"
        assert call_args[1]['api_base'] == "hf_spaces_local"
        assert call_args[1]['num_ctx'] == 2048

    def test_batch_embed_documents_empty_input(self):
        """Test function with empty input list."""
        result = batch_embed_documents(texts=[])

        # Should return empty numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0

    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_single_document(self, mock_embedding_fn):
        """Test embedding of single document."""
        mock_embedding_fn.return_value = create_mock_response([[0.1, 0.2, 0.3, 0.4]])

        result = batch_embed_documents(texts=["single document"], api_base="http://localhost:11434")

        assert result.shape == (1, 4)
        expected_normalized = np.array([0.1, 0.2, 0.3, 0.4]) / np.linalg.norm([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(result[0], expected_normalized, rtol=1e-5)

    @patch('agentic_nav.utils.embedding_generator.tqdm')
    @patch('agentic_nav.utils.embedding_generator.embedding_fn')
    def test_batch_embed_documents_progress_bar(self, mock_embedding_fn, mock_tqdm):
        """Test that progress bar is used for batch processing."""
        mock_embedding_fn.return_value = create_mock_response([[0.1, 0.2]])

        # Mock tqdm to return the range as-is
        mock_tqdm.return_value = range(0, 2, 1)

        batch_embed_documents(texts=["doc1", "doc2"], batch_size=1, api_base="http://localhost:11434", show_progress=False)

        # Verify tqdm was called with the range and disable=True (since show_progress=False)
        mock_tqdm.assert_called_once_with(range(0, 2, 1), disable=True)
