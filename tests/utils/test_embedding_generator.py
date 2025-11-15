"""
Tests for the embedding generator utility.
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from llm_agents.utils.embedding_generator import batch_embed_documents


class TestBatchEmbedDocuments:
    """Test the batch_embed_documents function."""

    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_basic(self, mock_embedding):
        """Test basic embedding generation functionality."""
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_embedding.return_value = mock_response
        
        texts = ["first document", "second document"]
        result = batch_embed_documents(
            texts=texts,
            batch_size=2,
            embedding_model="test-model",
            api_base="http://test.com"
        )
        
        # Verify embedding was called correctly
        mock_embedding.assert_called_once_with(
            model="test-model",
            input=texts,
            api_base="http://test.com",
            num_ctx=2048
        )
        
        # Verify result is numpy array with correct shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.array_equal(result[0], [0.1, 0.2, 0.3])
        assert np.array_equal(result[1], [0.4, 0.5, 0.6])

    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_with_batching(self, mock_embedding):
        """Test embedding with multiple batches."""
        # Mock responses for each batch
        batch1_response = Mock()
        batch1_response.data = [Mock(embedding=[0.1, 0.2])]
        
        batch2_response = Mock()
        batch2_response.data = [Mock(embedding=[0.3, 0.4])]
        
        mock_embedding.side_effect = [batch1_response, batch2_response]
        
        texts = ["doc1", "doc2"]
        result = batch_embed_documents(
            texts=texts,
            batch_size=1,  # Force multiple batches
            embedding_model="test-model",
            api_base="http://test.com"
        )
        
        # Verify embedding was called twice
        assert mock_embedding.call_count == 2
        
        # Check first call
        first_call = mock_embedding.call_args_list[0]
        assert first_call[1]['input'] == ["doc1"]
        
        # Check second call  
        second_call = mock_embedding.call_args_list[1]
        assert second_call[1]['input'] == ["doc2"]
        
        # Verify combined result
        assert result.shape == (2, 2)
        assert np.array_equal(result[0], [0.1, 0.2])
        assert np.array_equal(result[1], [0.3, 0.4])

    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_with_none_values(self, mock_embedding, caplog):
        """Test handling of None values in input texts."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4])
        ]
        mock_embedding.return_value = mock_response
        
        texts = ["valid document", None]
        
        batch_embed_documents(texts=texts)
        
        # Verify warning was logged
        assert "WARNING: Detected documents with 'None' values" in caplog.text
        
        # Verify None was replaced with empty string
        call_args = mock_embedding.call_args
        assert call_args[1]['input'] == ["valid document", ""]

    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_error_fallback(self, mock_embedding, caplog):
        """Test fallback to single sample processing on batch error."""
        # First call (batch) raises exception
        mock_embedding.side_effect = [
            Exception("Batch failed"),
            # Individual calls succeed
            Mock(data=[Mock(embedding=[0.1, 0.2])]),
            Mock(data=[Mock(embedding=[0.3, 0.4])])
        ]
        
        texts = ["doc1", "doc2"]
        result = batch_embed_documents(texts=texts, batch_size=2)
        
        # Verify error was logged
        assert "Error during embedding batch" in caplog.text
        assert "Falling back to single sample processing" in caplog.text
        
        # Verify fallback individual calls were made
        assert mock_embedding.call_count == 3  # 1 failed batch + 2 individual
        
        # Verify result is still correct
        assert result.shape == (2, 2)

    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_bad_request_fallback(self, mock_embedding, caplog):
        """Test handling of BadRequestError during individual processing."""
        from litellm import BadRequestError
        
        # Batch call fails, individual calls have mixed results
        mock_embedding.side_effect = [
            Exception("Batch failed"),
            BadRequestError("Bad request", response=Mock()),  # First individual fails
            Mock(data=[Mock(embedding=[0.3, 0.4])])  # Second succeeds
        ]
        
        texts = ["problematic doc", "good doc"]
        result = batch_embed_documents(texts=texts, batch_size=2)
        
        # Should handle BadRequestError gracefully
        assert "Failed to embed document" in caplog.text
        
        # Result should have zeros for failed embedding and actual values for successful one
        assert result.shape == (2, 2)
        assert np.array_equal(result[0], [0.0, 0.0])  # Failed embedding
        assert np.array_equal(result[1], [0.3, 0.4])  # Successful embedding

    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_default_params(self, mock_embedding):
        """Test function with default parameters."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_embedding.return_value = mock_response
        
        batch_embed_documents(texts=["test doc"])
        
        # Verify default parameters were used
        call_args = mock_embedding.call_args
        assert call_args[1]['model'] == "ollama/nomic-embed-text"
        assert call_args[1]['api_base'] == "http://localhost:11435"
        assert call_args[1]['num_ctx'] == 2048

    def test_batch_embed_documents_empty_input(self):
        """Test function with empty input list."""
        result = batch_embed_documents(texts=[])
        
        # Should return empty numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0

    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_single_document(self, mock_embedding):
        """Test embedding of single document."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]
        mock_embedding.return_value = mock_response
        
        result = batch_embed_documents(texts=["single document"])
        
        assert result.shape == (1, 4)
        assert np.array_equal(result[0], [0.1, 0.2, 0.3, 0.4])

    @patch('llm_agents.utils.embedding_generator.tqdm')
    @patch('llm_agents.utils.embedding_generator.embedding')
    def test_batch_embed_documents_progress_bar(self, mock_embedding, mock_tqdm):
        """Test that progress bar is used for batch processing."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_embedding.return_value = mock_response
        
        # Mock tqdm to return the range as-is
        mock_tqdm.return_value = range(0, 2, 1)
        
        batch_embed_documents(texts=["doc1", "doc2"], batch_size=1)
        
        # Verify tqdm was called with the range
        mock_tqdm.assert_called_once_with(range(0, 2, 1))