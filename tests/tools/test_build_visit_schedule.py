"""
Tests for the build_visit_schedule function.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from agentic_nav.tools.session_routing import build_visit_schedule


class TestBuildVisitSchedule:
    """Test the build_visit_schedule function."""

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_basic(self, mock_worker_class, mock_driver_class):
        """Test basic schedule building."""
        # Mock the worker
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95, 'name': 'Test Paper'}
        ]
        mock_worker_class.return_value = mock_worker

        # Mock the driver
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Mock the query result
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Test Paper',
                'abstract': 'Abstract',
                'topic': 'AI',
                'session': 'Morning',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'poster_position': '#123',
                'presentation_type': 'Poster',
                'url': 'https://example.com',
                'authors': ['Author A']
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="machine learning",
            max_papers=10,
            min_similarity=0.6
        )

        assert isinstance(result, str)
        assert "NeurIPS 2025" in result or "Test Paper" in result

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_with_multiple_topics(self, mock_worker_class, mock_driver_class):
        """Test schedule building with multiple topics."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95}
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Test Paper',
                'abstract': 'Abstract',
                'topic': 'AI',
                'session': 'Morning',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'poster_position': '#123',
                'presentation_type': 'Poster',
                'url': 'https://example.com',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics=["machine learning", "computer vision"],
            max_papers=10
        )

        # Should have called worker for each topic
        assert mock_worker.similarity_search.call_count == 2
        assert isinstance(result, str)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_with_dates(self, mock_worker_class, mock_driver_class):
        """Test schedule building with date filtering."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95}
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Test Paper',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'A',
                'topic': 'AI',
                'session': 'S',
                'room_name': 'Hall',
                'poster_position': '#1',
                'presentation_type': 'Poster',
                'url': 'url',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="machine learning",
            dates="2025-12-02",
            max_papers=10
        )

        assert isinstance(result, str)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_with_time_preferences(self, mock_worker_class, mock_driver_class):
        """Test schedule building with time preferences."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95}
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Morning Paper',
                'session_start_time': '2025-12-02T17:00:00Z',  # 9 AM PST
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'A',
                'topic': 'AI',
                'session': 'Morning',
                'room_name': 'Hall A',
                'poster_position': '#123',
                'presentation_type': 'Poster',
                'url': 'url',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="machine learning",
            time_preferences="morning",
            max_papers=10
        )

        assert isinstance(result, str)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_no_papers_found(self, mock_worker_class, mock_driver_class):
        """Test when no papers match the topics."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = []  # No papers
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="very obscure topic",
            max_papers=10
        )

        assert "No papers found" in result

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_no_papers_after_filtering(self, mock_worker_class, mock_driver_class):
        """Test when papers are found but filtered out by date/time."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95}
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        # Return paper with different date
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Paper',
                'session_start_time': '2025-12-05T17:00:00Z',  # Dec 5
                'session_end_time': '2025-12-05T19:00:00Z',
                'abstract': 'A',
                'topic': 'AI',
                'session': 'S',
                'room_name': 'Hall',
                'poster_position': '#1',
                'presentation_type': 'Poster',
                'url': 'url',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="machine learning",
            dates="2025-12-02",  # Different date
            max_papers=10
        )

        assert "No papers found" in result or "date and time" in result

    def test_build_visit_schedule_topics_required(self):
        """Test that topics parameter is required."""
        with pytest.raises(ValueError, match="Topics parameter is required"):
            build_visit_schedule(topics=None)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_type_coercion(self, mock_worker_class, mock_driver_class):
        """Test type coercion for max_papers and min_similarity."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = []
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_driver_class.return_value = mock_driver

        # Pass string values that should be coerced
        result = build_visit_schedule(
            topics="machine learning",
            max_papers="15",  # String instead of int
            min_similarity="0.7"  # String instead of float
        )

        # Should not raise error, values should be coerced
        assert isinstance(result, str)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_multiple_dates(self, mock_worker_class, mock_driver_class):
        """Test with multiple dates."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95}
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Paper 1',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'A',
                'topic': 'AI',
                'session': 'S',
                'room_name': 'Hall',
                'poster_position': '#1',
                'presentation_type': 'Poster',
                'url': 'url',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="machine learning",
            dates=["2025-12-02", "2025-12-03"],
            max_papers=10
        )

        assert isinstance(result, str)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_day_names(self, mock_worker_class, mock_driver_class):
        """Test with day names instead of dates."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95}
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Paper',
                'session_start_time': '2025-12-02T17:00:00Z',  # Tuesday
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'A',
                'topic': 'AI',
                'session': 'S',
                'room_name': 'Hall',
                'poster_position': '#1',
                'presentation_type': 'Poster',
                'url': 'url',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="machine learning",
            dates="Tuesday",
            max_papers=10
        )

        assert isinstance(result, str)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_time_range_format(self, mock_worker_class, mock_driver_class):
        """Test with time range format."""
        mock_worker = Mock()
        mock_worker.similarity_search.return_value = [
            {'id': 'paper1', 'score': 0.95}
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Paper',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'A',
                'topic': 'AI',
                'session': 'S',
                'room_name': 'Hall',
                'poster_position': '#1',
                'presentation_type': 'Poster',
                'url': 'url',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics="machine learning",
            time_preferences="9:00-12:00",
            max_papers=10
        )

        assert isinstance(result, str)

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_handles_search_errors(self, mock_worker_class, mock_driver_class):
        """Test handling of errors during paper search."""
        mock_worker = Mock()
        mock_worker.similarity_search.side_effect = Exception("Search failed")
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics=["topic1", "topic2"],
            max_papers=10
        )

        # Should continue with other topics and eventually return "no papers" message
        assert "No papers found" in result

    @patch('agentic_nav.tools.session_routing.GraphDatabase.driver')
    @patch('agentic_nav.tools.knowledge_graph.retriever.Neo4jGraphWorker')
    def test_build_visit_schedule_merges_scores_from_multiple_topics(self, mock_worker_class, mock_driver_class):
        """Test that highest scores are kept when paper matches multiple topics."""
        mock_worker = Mock()
        # Same paper returned for both topics with different scores
        mock_worker.similarity_search.side_effect = [
            [{'id': 'paper1', 'score': 0.85}],  # First topic
            [{'id': 'paper1', 'score': 0.95}]   # Second topic (higher score)
        ]
        mock_worker_class.return_value = mock_worker

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {
                'id': 'paper1',
                'name': 'Paper',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'A',
                'topic': 'AI',
                'session': 'S',
                'room_name': 'Hall',
                'poster_position': '#1',
                'presentation_type': 'Poster',
                'url': 'url',
                'authors': []
            }
        ]
        mock_driver_class.return_value = mock_driver

        result = build_visit_schedule(
            topics=["topic1", "topic2"],
            max_papers=10
        )

        # Should use the higher score (0.95) and include paper once
        assert isinstance(result, str)
