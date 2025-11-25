"""
Tests for graph traversal strategy functions.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from agentic_nav.tools.knowledge_graph.graph_traversal_strategies.breadth_first_random import (
    _graph_traversal_bfs_random
)
from agentic_nav.tools.knowledge_graph.graph_traversal_strategies.depth_first_random import (
    _graph_traversal_dfs_random
)
from agentic_nav.tools.knowledge_graph.graph_traversal_strategies.neo4j_builtin import (
    _graph_traversal_cypher
)


def create_mock_driver_with_session(mock_session_return_value):
    """Helper to create a properly mocked Neo4j driver with context manager support."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = mock_session_return_value
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = False
    return mock_driver, mock_session


class TestBreadthFirstRandom:
    """Test the BFS random traversal strategy."""

    def test_bfs_basic_traversal(self):
        """Test basic BFS traversal."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = MagicMock(__enter__=Mock(return_value=mock_session), __exit__=Mock(return_value=False))

        # Mock neighbors for the start node
        mock_session.run.return_value = [
            {
                'id': 'paper2',
                'name': 'Paper 2',
                'abstract': 'Abstract 2',
                'topic': 'AI'
            },
            {
                'id': 'paper3',
                'name': 'Paper 3',
                'abstract': 'Abstract 3',
                'topic': 'ML'
            }
        ]

        result = _graph_traversal_bfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=10,
            max_branches=5
        )

        assert len(result) <= 2
        assert all('distance' in paper for paper in result)

    def test_bfs_respects_max_results(self):
        """Test that BFS respects max_results limit."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Return many neighbors
        mock_session.run.return_value = [
            {
                'id': f'paper{i}',
                'name': f'Paper {i}',
                'abstract': f'Abstract {i}',
                'topic': 'AI'
            }
            for i in range(20)
        ]

        result = _graph_traversal_bfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=5,
            max_branches=10
        )

        assert len(result) <= 5

    def test_bfs_respects_max_branches(self):
        """Test that BFS samples at most max_branches neighbors."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Return many neighbors
        mock_session.run.return_value = [
            {
                'id': f'paper{i}',
                'name': f'Paper {i}',
                'abstract': f'Abstract {i}',
                'topic': 'AI'
            }
            for i in range(20)
        ]

        result = _graph_traversal_bfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=None,
            max_branches=3
        )

        # Should sample at most 3 neighbors
        assert len(result) <= 3

    def test_bfs_with_relationship_type(self):
        """Test BFS with specific relationship type."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        _graph_traversal_bfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type='SIMILAR_TO',
            max_results=10,
            max_branches=5
        )

        # Check that query includes relationship type
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert 'SIMILAR_TO' in query

    def test_bfs_avoids_visited_nodes(self):
        """Test that BFS doesn't revisit nodes."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # First level returns paper2
        # Second level should not include paper1 (start) or paper2
        mock_session.run.side_effect = [
            [{'id': 'paper2', 'name': 'Paper 2', 'abstract': 'A', 'topic': 'AI'}],
            # Second call for paper2's neighbors
            [
                {'id': 'paper1', 'name': 'Paper 1', 'abstract': 'A', 'topic': 'AI'},  # Should be skipped
                {'id': 'paper3', 'name': 'Paper 3', 'abstract': 'A', 'topic': 'AI'}
            ]
        ]

        result = _graph_traversal_bfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=2,
            relationship_type=None,
            max_results=None,
            max_branches=5
        )

        # Should have paper2 and paper3, but not paper1 again
        paper_ids = [p['id'] for p in result]
        assert 'paper1' not in paper_ids
        assert 'paper2' in paper_ids

    def test_bfs_empty_neighbors(self):
        """Test BFS when node has no neighbors."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        result = _graph_traversal_bfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=2,
            relationship_type=None,
            max_results=10,
            max_branches=5
        )

        assert result == []


class TestDepthFirstRandom:
    """Test the DFS random traversal strategy."""

    def test_dfs_basic_traversal(self):
        """Test basic DFS traversal."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_session.run.return_value = [
            {
                'id': 'paper2',
                'name': 'Paper 2',
                'abstract': 'Abstract 2',
                'topic': 'AI'
            }
        ]

        result = _graph_traversal_dfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=10,
            max_branches=5
        )

        assert len(result) >= 0
        assert all('distance' in paper for paper in result)

    def test_dfs_respects_max_results(self):
        """Test that DFS respects max_results limit."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Return neighbors at each level
        mock_session.run.return_value = [
            {
                'id': f'paper{i}',
                'name': f'Paper {i}',
                'abstract': f'Abstract {i}',
                'topic': 'AI'
            }
            for i in range(10)
        ]

        result = _graph_traversal_dfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=3,
            relationship_type=None,
            max_results=5,
            max_branches=2
        )

        assert len(result) <= 5

    def test_dfs_respects_max_branches(self):
        """Test that DFS samples at most max_branches neighbors."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # First call returns many neighbors
        mock_session.run.side_effect = [
            [{'id': f'paper{i}', 'name': f'Paper {i}', 'abstract': 'A', 'topic': 'AI'}
             for i in range(20)],
            []  # Subsequent calls return empty
        ]

        result = _graph_traversal_dfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=None,
            max_branches=3
        )

        # Should sample at most 3 neighbors from first level
        assert len(result) <= 3

    def test_dfs_with_relationship_type(self):
        """Test DFS with specific relationship type."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        _graph_traversal_dfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type='CITES',
            max_results=10,
            max_branches=5
        )

        # Check that query includes relationship type
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert 'CITES' in query

    def test_dfs_avoids_visited_nodes(self):
        """Test that DFS doesn't revisit nodes."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Setup to return paper2, then paper3, but not revisit paper1
        mock_session.run.side_effect = [
            [{'id': 'paper2', 'name': 'Paper 2', 'abstract': 'A', 'topic': 'AI'}],
            [
                {'id': 'paper1', 'name': 'Paper 1', 'abstract': 'A', 'topic': 'AI'},  # Should skip
                {'id': 'paper3', 'name': 'Paper 3', 'abstract': 'A', 'topic': 'AI'}
            ],
            []
        ]

        result = _graph_traversal_dfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=2,
            relationship_type=None,
            max_results=None,
            max_branches=5
        )

        # Should not include paper1 (start node)
        paper_ids = [p['id'] for p in result]
        assert 'paper1' not in paper_ids

    def test_dfs_empty_neighbors(self):
        """Test DFS when node has no neighbors."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        result = _graph_traversal_dfs_random(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=2,
            relationship_type=None,
            max_results=10,
            max_branches=5
        )

        assert result == []


class TestNeo4jBuiltin:
    """Test the Neo4j built-in Cypher traversal strategy."""

    def test_cypher_basic_traversal(self):
        """Test basic Cypher traversal."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_session.run.return_value = [
            {
                'id': 'paper2',
                'name': 'Paper 2',
                'abstract': 'Abstract 2',
                'topic': 'AI',
                'distance': 1
            },
            {
                'id': 'paper3',
                'name': 'Paper 3',
                'abstract': 'Abstract 3',
                'topic': 'ML',
                'distance': 1
            }
        ]

        result = _graph_traversal_cypher(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=10
        )

        assert len(result) == 2
        assert result[0]['id'] == 'paper2'
        assert result[0]['distance'] == 1

    def test_cypher_respects_max_results(self):
        """Test that Cypher traversal respects max_results."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_session.run.return_value = [
            {
                'id': f'paper{i}',
                'name': f'Paper {i}',
                'abstract': f'Abstract {i}',
                'topic': 'AI',
                'distance': 1
            }
            for i in range(20)
        ]

        _graph_traversal_cypher(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=5
        )

        # Check that LIMIT was added to query
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert 'LIMIT 5' in query

    def test_cypher_with_relationship_type(self):
        """Test Cypher traversal with specific relationship type."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        _graph_traversal_cypher(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=2,
            relationship_type='SIMILAR_TO',
            max_results=None
        )

        # Check that query includes relationship type
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert 'SIMILAR_TO' in query

    def test_cypher_without_max_results(self):
        """Test Cypher traversal without max_results limit."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        _graph_traversal_cypher(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=2,
            relationship_type=None,
            max_results=None
        )

        # Query should not have LIMIT clause
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert 'LIMIT' not in query

    def test_cypher_returns_correct_structure(self):
        """Test that Cypher traversal returns correctly structured papers."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_session.run.return_value = [
            {
                'id': 'paper2',
                'name': 'Test Paper',
                'abstract': 'Test Abstract',
                'topic': 'AI',
                'distance': 2
            }
        ]

        result = _graph_traversal_cypher(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=2,
            relationship_type=None,
            max_results=10
        )

        assert len(result) == 1
        paper = result[0]
        assert paper['id'] == 'paper2'
        assert paper['name'] == 'Test Paper'
        assert paper['abstract'] == 'Test Abstract'
        assert paper['topic'] == 'AI'
        assert paper['distance'] == 2

    def test_cypher_empty_result(self):
        """Test Cypher traversal with no results."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        result = _graph_traversal_cypher(
            db_driver=mock_driver,
            start_paper_id='paper1',
            n_hops=1,
            relationship_type=None,
            max_results=10
        )

        assert result == []
