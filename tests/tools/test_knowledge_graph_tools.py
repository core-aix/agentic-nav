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
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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
        
        # Verify worker was created with correct params (patched values)
        mock_worker_class.assert_called_once_with(
            uri='bolt://localhost:7687',
            username='neo4j',
            password='llm_agents'
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
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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

    @pytest.mark.no_auto_env
    def test_search_uses_environment_variables(self):
        """Test that function uses environment variables for Neo4j connection."""
        import sys
        import importlib

        # Store original modules to restore later
        original_modules = {}
        modules_to_clear = [
            'llm_agents.tools.knowledge_graph',
            'llm_agents.tools.knowledge_graph.retriever'
        ]

        for module in modules_to_clear:
            if module in sys.modules:
                original_modules[module] = sys.modules[module]
                del sys.modules[module]

        try:
            # Set custom environment variables FIRST, before any imports
            with patch.dict('os.environ', {
                'NEO4J_URI': 'bolt://custom:7687',
                'NEO4J_USERNAME': 'custom_user',
                'NEO4J_PASSWORD': 'custom_pass'
            }, clear=False), \
            patch('neo4j.GraphDatabase.driver') as mock_driver, \
            patch('toon_format.encode') as mock_encode, \
            patch('llm_agents.utils.embedding_generator.embedding') as mock_embedding:

                # Setup the Neo4j driver mock
                mock_driver_instance = Mock()
                mock_driver.return_value = mock_driver_instance
                mock_driver_instance.verify_connectivity.return_value = True

                # Setup session mock to support context manager protocol
                mock_session = Mock()
                mock_session.__enter__ = Mock(return_value=mock_session)
                mock_session.__exit__ = Mock(return_value=False)

                # Make session.run() return an iterable result (empty list)
                mock_result = Mock()
                mock_result.__iter__ = Mock(return_value=iter([]))  # Empty iterator
                mock_session.run.return_value = mock_result
                mock_driver_instance.session.return_value = mock_session

                # Mock embedding responses
                mock_response = Mock()
                mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
                mock_response.__getitem__ = lambda self, key: mock_response.data if key == "data" else None
                mock_embedding.return_value = mock_response

                # Now import the module - it will read from patched environment
                import llm_agents.tools.knowledge_graph
                importlib.reload(llm_agents.tools.knowledge_graph)
                from llm_agents.tools.knowledge_graph import search_similar_papers

                # Mock the search results
                mock_encode.return_value = "result"

                # Call the function
                result = search_similar_papers("test")

                # Verify the driver was called with the correct environment variables
                mock_driver.assert_called_with(
                    "bolt://custom:7687",
                    auth=("custom_user", "custom_pass")
                )

                # Verify the result
                assert result == "result"

        finally:
            # Restore original modules to avoid interfering with other tests
            for module, original_module in original_modules.items():
                sys.modules[module] = original_module


class TestFindNeighboringPapers:
    """Test the find_neighboring_papers function."""

    @patch('llm_agents.tools.knowledge_graph.Neo4jGraphWorker')
    @patch('llm_agents.tools.knowledge_graph.toon_encode')
    @patch('llm_agents.tools.knowledge_graph.random.shuffle')
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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
            uri='bolt://localhost:7687',
            username='neo4j', 
            password='llm_agents'
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
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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
            uri='bolt://localhost:7687',
            username='neo4j',
            password='llm_agents'
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
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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
    @patch.dict('os.environ', {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'llm_agents'
    })
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