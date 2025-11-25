"""
Tests for knowledge graph tool functions.
"""
import pytest
import sys
import importlib
from unittest.mock import patch, Mock, MagicMock


def reload_knowledge_graph_module():
    """Reload the knowledge graph module to pick up environment changes."""
    if 'agentic_nav.tools.knowledge_graph' in sys.modules:
        importlib.reload(sys.modules['agentic_nav.tools.knowledge_graph'])
    else:
        import agentic_nav.tools.knowledge_graph

    from agentic_nav.tools.knowledge_graph import (
        search_similar_papers,
        find_neighboring_papers,
        traverse_graph
    )
    return search_similar_papers, find_neighboring_papers, traverse_graph


class TestSearchSimilarPapers:
    """Test the search_similar_papers function."""

    # def test_search_similar_papers_basic(self):
    #     """Test basic similarity search functionality."""
    #     with patch.dict('os.environ', {
    #         'NEO4J_URI': 'bolt://localhost:7687',
    #         'NEO4J_USERNAME': 'neo4j',
    #         'NEO4J_PASSWORD': 'agentic_nav'
    #     }):
    #         # Reload module to pick up patched environment variables
    #         search_similar_papers, _, _ = reload_knowledge_graph_module()
    #
    #         with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
    #              patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode:
    #
    #             # Mock the worker instance
    #             mock_worker = Mock()
    #             mock_worker_class.return_value = mock_worker
    #
    #             # Mock search results
    #             mock_papers = [
    #                 {"id": "paper1", "title": "Test Paper 1", "score": 0.95},
    #                 {"id": "paper2", "title": "Test Paper 2", "score": 0.90}
    #             ]
    #             mock_worker.similarity_search.return_value = mock_papers
    #
    #             # Mock encoding
    #             mock_encode.return_value = "encoded_papers"
    #
    #             # Call the function
    #             result = search_similar_papers(
    #                 user_query="machine learning",
    #                 num_papers_to_return=5,
    #                 min_similarity=0.8
    #             )
    #
    #             # Verify worker was created with correct params (patched values)
    #             mock_worker_class.assert_called_once_with(
    #                 uri='bolt://localhost:7687',
    #                 username='neo4j',
    #                 password='agentic_nav'
    #             )
    #
    #             # Verify search was called correctly
    #             mock_worker.similarity_search.assert_called_once_with(
    #                 user_query="machine learning",
    #                 top_k=5,
    #                 min_similarity=0.8
    #             )
    #
    #             # Verify encoding was applied
    #             mock_encode.assert_called_once_with(mock_papers)
    #
    #             # Verify return value
    #             assert result == "encoded_papers"

    def test_search_similar_papers_default_params(self):
        """Test search with default parameters."""
        with patch.dict('os.environ', {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'agentic_nav'
        }):
            # Reload module to pick up patched environment variables
            search_similar_papers, _, _ = reload_knowledge_graph_module()

            with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
                 patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode:

                mock_worker = Mock()
                mock_worker_class.return_value = mock_worker
                mock_worker.similarity_search.return_value = []
                mock_encode.return_value = "empty_results"

                result = search_similar_papers("test query")

                # Verify default parameters were used
                mock_worker.similarity_search.assert_called_once_with(
                    user_query="test query",
                    top_k=50,  # default changed from 10 to 50
                    min_similarity=None,  # default
                    day=None,  # new parameter
                    timeslots=None  # new parameter
                )

    # @pytest.mark.no_auto_env
    # def test_search_uses_environment_variables(self):
    #     """Test that function uses environment variables for Neo4j connection."""
    #     with patch.dict('os.environ', {
    #         'NEO4J_URI': 'bolt://custom:7687',
    #         'NEO4J_USERNAME': 'custom_user',
    #         'NEO4J_PASSWORD': 'custom_pass'
    #     }):
    #         # Reload module to pick up patched environment variables
    #         search_similar_papers, _, _ = reload_knowledge_graph_module()
    #
    #         with patch('neo4j.GraphDatabase.driver') as mock_driver, \
    #              patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode, \
    #              patch('agentic_nav.utils.embedding_generator.embedding') as mock_embedding:
    #
    #             # Setup the Neo4j driver mock
    #             mock_driver_instance = Mock()
    #             mock_driver.return_value = mock_driver_instance
    #             mock_driver_instance.verify_connectivity.return_value = True
    #
    #             # Setup session mock to support context manager protocol
    #             mock_session = Mock()
    #             mock_session.__enter__ = Mock(return_value=mock_session)
    #             mock_session.__exit__ = Mock(return_value=False)
    #
    #             # Make session.run() return an iterable result (empty list)
    #             mock_result = Mock()
    #             mock_result.__iter__ = Mock(return_value=iter([]))  # Empty iterator
    #             mock_session.run.return_value = mock_result
    #             mock_driver_instance.session.return_value = mock_session
    #
    #             # Mock embedding responses
    #             mock_response = Mock()
    #             mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
    #             mock_response.__getitem__ = lambda self, key: mock_response.data if key == "data" else None
    #             mock_embedding.return_value = mock_response
    #
    #             # Mock the search results
    #             mock_encode.return_value = "result"
    #
    #             # Call the function
    #             result = search_similar_papers("test")
    #
    #             # Verify the driver was called with the correct environment variables
    #             mock_driver.assert_called_with(
    #                 "bolt://custom:7687",
    #                 auth=("custom_user", "custom_pass")
    #             )
    #
    #             # Verify the result
    #             assert result == "result"


class TestFindNeighboringPapers:
    """Test the find_neighboring_papers function."""

    # def test_find_neighboring_papers_basic(self):
    #     """Test basic neighborhood search functionality."""
    #     with patch.dict('os.environ', {
    #         'NEO4J_URI': 'bolt://localhost:7687',
    #         'NEO4J_USERNAME': 'neo4j',
    #         'NEO4J_PASSWORD': 'agentic_nav'
    #     }):
    #         # Reload module to pick up patched environment variables
    #         _, find_neighboring_papers, _ = reload_knowledge_graph_module()
    #
    #         with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
    #              patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode, \
    #              patch('agentic_nav.tools.knowledge_graph.random.shuffle') as mock_shuffle:
    #
    #             mock_worker = Mock()
    #             mock_worker_class.return_value = mock_worker
    #
    #             # Mock neighborhood search results
    #             mock_neighbors = {
    #                 "similar_papers": [
    #                     {"neighbor": {"id": "paper1", "title": "Similar Paper 1"}},
    #                     {"neighbor": {"id": "paper2", "title": "Similar Paper 2"}}
    #                 ]
    #             }
    #             mock_worker.neighborhood_search.return_value = mock_neighbors
    #             mock_encode.return_value = "encoded_neighbors"
    #
    #             result = find_neighboring_papers(
    #                 paper_id="test_paper_id",
    #                 relationship_types=["SIMILAR_TO"],
    #                 neighbor_entity="similar_papers",
    #                 num_neighbors_to_return=5
    #             )
    #
    #             # Verify worker creation
    #             mock_worker_class.assert_called_once_with(
    #                 uri='bolt://localhost:7687',
    #                 username='neo4j',
    #                 password='agentic_nav'
    #             )
    #
    #             # Verify neighborhood search
    #             mock_worker.neighborhood_search.assert_called_once_with(
    #                 paper_id="test_paper_id",
    #                 relationship_types=["SIMILAR_TO"]
    #             )
    #
    #             # Verify shuffle was called
    #             mock_shuffle.assert_called_once()
    #
    #             # Verify encoding
    #             mock_encode.assert_called_once()
    #
    #             assert result == "encoded_neighbors"

    def test_find_neighboring_papers_string_relationship_type(self):
        """Test that string relationship_type is converted to list."""
        with patch.dict('os.environ', {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'agentic_nav'
        }):
            # Reload module to pick up patched environment variables
            _, find_neighboring_papers, _ = reload_knowledge_graph_module()

            with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
                 patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode:

                mock_worker = Mock()
                mock_worker_class.return_value = mock_worker
                mock_worker.neighborhood_search.return_value = {"similar_papers": []}
                mock_encode.return_value = "result"

                find_neighboring_papers(
                    paper_id="test_id",
                    relationship_types="SIMILAR_TO"  # String instead of list
                )

                # Should convert string to list
                mock_worker.neighborhood_search.assert_called_once_with(
                    paper_id="test_id",
                    relationship_types=["SIMILAR_TO"],
                    min_similarity=0.75  # default
                )

    def test_find_neighboring_papers_defaults(self):
        """Test function with default parameters."""
        with patch.dict('os.environ', {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'agentic_nav'
        }):
            # Reload module to pick up patched environment variables
            _, find_neighboring_papers, _ = reload_knowledge_graph_module()

            with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
                 patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode:

                mock_worker = Mock()
                mock_worker_class.return_value = mock_worker
                mock_worker.neighborhood_search.return_value = {"similar_papers": []}
                mock_encode.return_value = "result"

                find_neighboring_papers("test_id")

                # Verify defaults are used
                mock_worker.neighborhood_search.assert_called_once_with(
                    paper_id="test_id",
                    relationship_types=["SIMILAR_TO"],  # default
                    min_similarity=0.75  # default
                )


class TestTraverseGraph:
    """Test the traverse_graph function."""

    # def test_traverse_graph_basic(self):
    #     """Test basic graph traversal functionality."""
    #     with patch.dict('os.environ', {
    #         'NEO4J_URI': 'bolt://localhost:7687',
    #         'NEO4J_USERNAME': 'neo4j',
    #         'NEO4J_PASSWORD': 'agentic_nav'
    #     }):
    #         # Reload module to pick up patched environment variables
    #         _, _, traverse_graph = reload_knowledge_graph_module()
    #
    #         with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
    #              patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode:
    #
    #             mock_worker = Mock()
    #             mock_worker_class.return_value = mock_worker
    #
    #             mock_papers = [
    #                 {"id": "paper1", "title": "Traversed Paper 1"},
    #                 {"id": "paper2", "title": "Traversed Paper 2"}
    #             ]
    #             mock_worker.graph_traversal.return_value = mock_papers
    #             mock_encode.return_value = "encoded_traversal"
    #
    #             result = traverse_graph(
    #                 start_paper_id="start_id",
    #                 n_hops=3,
    #                 relationship_type="SIMILAR_TO",
    #                 max_results=50,
    #                 strategy="depth_first",
    #                 max_branches=3,
    #                 random_seed=123
    #             )
    #
    #             # Verify worker creation
    #             mock_worker_class.assert_called_once_with(
    #                 uri='bolt://localhost:7687',
    #                 username='neo4j',
    #                 password='agentic_nav'
    #             )
    #
    #             # Verify traversal call
    #             mock_worker.graph_traversal.assert_called_once_with(
    #                 start_paper_id="start_id",
    #                 n_hops=3,
    #                 relationship_type="SIMILAR_TO",
    #                 max_results=50,
    #                 strategy="depth_first",
    #                 max_branches=3,
    #                 random_seed=123
    #             )
    #
    #             # Verify encoding
    #             mock_encode.assert_called_once_with(mock_papers)
    #
    #             assert result == "encoded_traversal"

    def test_traverse_graph_defaults(self):
        """Test graph traversal with default parameters."""
        with patch.dict('os.environ', {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'agentic_nav'
        }):
            # Reload module to pick up patched environment variables
            _, _, traverse_graph = reload_knowledge_graph_module()

            with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
                 patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode:

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

    def test_traverse_graph_optional_none_values(self):
        """Test graph traversal with None values for optional parameters."""
        with patch.dict('os.environ', {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'agentic_nav'
        }):
            # Reload module to pick up patched environment variables
            _, _, traverse_graph = reload_knowledge_graph_module()

            with patch('agentic_nav.tools.knowledge_graph.Neo4jGraphWorker') as mock_worker_class, \
                 patch('agentic_nav.tools.knowledge_graph.toon_encode') as mock_encode:

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
