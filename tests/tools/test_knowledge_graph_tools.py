"""
Tests for knowledge graph tool functions.
"""
import pytest
from unittest.mock import patch, Mock, MagicMock

from llm_agents.tools.knowledge_graph import (
    search_similar_papers,
    find_neighboring_papers, 
    traverse_graph
)


class TestSearchSimilarPapers:
    """Test the search_similar_papers function."""

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    def test_search_similar_papers_basic(self, mock_encode, mock_worker_class):
        """Test basic similarity search functionality."""
        # Mock the worker instance
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        
        # Mock search results
        mock_papers = [
            {"id": "paper1", "title": "Test Paper 1", "score": 0.95},
            {"id": "paper2", "title": "Test Paper 2", "score": 0.90}
        ]
        mock_worker.similarity_search.return_value = mock_papers
        
        # Mock encoding
        mock_encode.return_value = "encoded_papers"
        
        # Call the function
        result = search_similar_papers(
            user_query="machine learning",
            num_papers_to_return=5,
            min_similarity=0.8
        )
        
        # Verify worker was created with correct params
        mock_worker_class.assert_called_once_with(
            uri="bolt://localhost:7687",
            username="neo4j",
            password=None  # From test environment
        )
        
        # Verify search was called correctly
        mock_worker.similarity_search.assert_called_once_with(
            user_query="machine learning",
            top_k=5,
            min_similarity=0.8
        )
        
        # Verify encoding was applied
        mock_encode.assert_called_once_with(mock_papers)
        
        # Verify return value
        assert result == "encoded_papers"

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    def test_search_similar_papers_default_params(self, mock_encode, mock_worker_class):
        """Test search with default parameters."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        mock_worker.similarity_search.return_value = []
        mock_encode.return_value = "empty_results"
        
        result = search_similar_papers("test query")
        
        # Verify default parameters were used
        mock_worker.similarity_search.assert_called_once_with(
            user_query="test query",
            top_k=10,  # default
            min_similarity=None  # default
        )

    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://custom:7687',
        'NEO4J_USERNAME': 'custom_user',
        'NEO4J_PASSWORD': 'custom_pass'
    })
    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    def test_search_uses_environment_variables(self, mock_worker_class):
        """Test that function uses environment variables for Neo4j connection."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        mock_worker.similarity_search.return_value = []
        
        search_similar_papers("test")
        
        mock_worker_class.assert_called_once_with(
            uri="bolt://custom:7687",
            username="custom_user", 
            password="custom_pass"
        )


class TestFindNeighboringPapers:
    """Test the find_neighboring_papers function."""

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    @patch('llm_agents.tools.knowledge_graph.random.shuffle')
    def test_find_neighboring_papers_basic(self, mock_shuffle, mock_encode, mock_worker_class):
        """Test basic neighborhood search functionality."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        
        # Mock neighborhood search results
        mock_neighbors = {
            "similar_papers": [
                {"neighbor": {"id": "paper1", "title": "Similar Paper 1"}},
                {"neighbor": {"id": "paper2", "title": "Similar Paper 2"}}
            ]
        }
        mock_worker.neighborhood_search.return_value = mock_neighbors
        mock_encode.return_value = "encoded_neighbors"
        
        result = find_neighboring_papers(
            paper_id="test_paper_id",
            relationship_types=["SIMILAR_TO"],
            neighbor_entity="similar_papers",
            num_neighbors_to_return=5
        )
        
        # Verify worker creation
        mock_worker_class.assert_called_once_with(
            uri="bolt://localhost:7687",
            username="neo4j", 
            password=None
        )
        
        # Verify neighborhood search
        mock_worker.neighborhood_search.assert_called_once_with(
            paper_id="test_paper_id",
            relationship_types=["SIMILAR_TO"]
        )
        
        # Verify shuffle was called
        mock_shuffle.assert_called_once()
        
        # Verify encoding
        mock_encode.assert_called_once()
        
        assert result == "encoded_neighbors"

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    def test_find_neighboring_papers_string_relationship_type(self, mock_encode, mock_worker_class):
        """Test that string relationship_type is converted to list."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        mock_worker.neighborhood_search.return_value = {"similar_papers": []}
        mock_encode.return_value = "result"
        
        find_neighboring_papers(
            paper_id="test_id",
            relationship_types="SIMILAR_TO",  # String instead of list
            neighbor_entity="similar_papers"
        )
        
        # Should convert string to list
        mock_worker.neighborhood_search.assert_called_once_with(
            paper_id="test_id",
            relationship_types=["SIMILAR_TO"]
        )

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    def test_find_neighboring_papers_defaults(self, mock_encode, mock_worker_class):
        """Test function with default parameters."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        mock_worker.neighborhood_search.return_value = {"similar_papers": []}
        mock_encode.return_value = "result"
        
        find_neighboring_papers("test_id")
        
        # Verify defaults are used
        mock_worker.neighborhood_search.assert_called_once_with(
            paper_id="test_id",
            relationship_types=["SIMILAR_TO"]  # default
        )


class TestTraverseGraph:
    """Test the traverse_graph function."""

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    def test_traverse_graph_basic(self, mock_encode, mock_worker_class):
        """Test basic graph traversal functionality."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        
        mock_papers = [
            {"id": "paper1", "title": "Traversed Paper 1"},
            {"id": "paper2", "title": "Traversed Paper 2"}
        ]
        mock_worker.graph_traversal.return_value = mock_papers
        mock_encode.return_value = "encoded_traversal"
        
        result = traverse_graph(
            start_paper_id="start_id",
            n_hops=3,
            relationship_type="SIMILAR_TO",
            max_results=50,
            strategy="depth_first",
            max_branches=3,
            random_seed=123
        )
        
        # Verify worker creation
        mock_worker_class.assert_called_once_with(
            uri="bolt://localhost:7687",
            username="neo4j",
            password=None
        )
        
        # Verify traversal call
        mock_worker.graph_traversal.assert_called_once_with(
            start_paper_id="start_id",
            n_hops=3,
            relationship_type="SIMILAR_TO",
            max_results=50,
            strategy="depth_first",
            max_branches=3,
            random_seed=123
        )
        
        # Verify encoding
        mock_encode.assert_called_once_with(mock_papers)
        
        assert result == "encoded_traversal"

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    def test_traverse_graph_defaults(self, mock_encode, mock_worker_class):
        """Test graph traversal with default parameters."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        mock_worker.graph_traversal.return_value = []
        mock_encode.return_value = "result"
        
        traverse_graph("start_id")
        
        # Verify default parameters
        mock_worker.graph_traversal.assert_called_once_with(
            start_paper_id="start_id",
            n_hops=2,  # default
            relationship_type="BELONGS_TO_TOPIC",  # default
            max_results=30,  # default
            strategy="breadth_first_random",  # default
            max_branches=2,  # default
            random_seed=42  # default
        )

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    def test_traverse_graph_optional_none_values(self, mock_encode, mock_worker_class):
        """Test graph traversal with None values for optional parameters."""
        mock_worker = Mock()
        mock_worker_class.return_value = mock_worker
        mock_worker.graph_traversal.return_value = []
        mock_encode.return_value = "result"
        
        traverse_graph(
            start_paper_id="start_id",
            relationship_type=None,
            max_results=None,
            max_branches=None,
            random_seed=None
        )
        
        # Verify None values are passed through
        mock_worker.graph_traversal.assert_called_once_with(
            start_paper_id="start_id",
            n_hops=2,  # default
            relationship_type=None,
            max_results=None,
            strategy="breadth_first_random",  # default
            max_branches=None,
            random_seed=None
        )