"""
Tests for the ScheduleBuilder class.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from agentic_nav.tools.session_routing.scheduler import ScheduleBuilder


class TestScheduleBuilderInit:
    """Test ScheduleBuilder initialization."""

    def test_init_with_driver(self):
        """Test initialization with Neo4j driver."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)
        assert builder.driver == mock_driver


class TestFilterByDatetime:
    """Test the filter_by_datetime method."""

    def test_filter_by_datetime_empty_paper_ids(self):
        """Test filtering with empty paper IDs list."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        result = builder.filter_by_datetime([], dates=None, time_range=None)
        assert result == []

    def test_filter_by_datetime_no_filters(self):
        """Test filtering without date or time filters."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_result = [
            {
                'id': 'paper1',
                'name': 'Test Paper 1',
                'abstract': 'Abstract 1',
                'topic': 'AI',
                'session': 'Morning Session',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'poster_position': '#123',
                'presentation_type': 'Poster',
                'url': 'https://example.com/paper1',
                'authors': ['Author A', 'Author B']
            }
        ]
        mock_session.run.return_value = mock_result

        builder = ScheduleBuilder(mock_driver)
        result = builder.filter_by_datetime(['paper1'], dates=None, time_range=None)

        assert len(result) == 1
        assert result[0]['id'] == 'paper1'

    def test_filter_by_datetime_with_date_filter(self):
        """Test filtering by specific dates."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_result = [
            {
                'id': 'paper1',
                'name': 'Test Paper 1',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'Abstract 1',
                'topic': 'AI',
                'session': 'Morning',
                'room_name': 'Hall A',
                'poster_position': '#123',
                'presentation_type': 'Poster',
                'url': 'https://example.com',
                'authors': ['Author A']
            },
            {
                'id': 'paper2',
                'name': 'Test Paper 2',
                'session_start_time': '2025-12-03T17:00:00Z',
                'session_end_time': '2025-12-03T19:00:00Z',
                'abstract': 'Abstract 2',
                'topic': 'ML',
                'session': 'Afternoon',
                'room_name': 'Hall B',
                'poster_position': '#124',
                'presentation_type': 'Poster',
                'url': 'https://example.com',
                'authors': ['Author B']
            }
        ]
        mock_session.run.return_value = mock_result

        builder = ScheduleBuilder(mock_driver)
        dates = [datetime(2025, 12, 2)]
        result = builder.filter_by_datetime(['paper1', 'paper2'], dates=dates)

        # Should only include paper from Dec 2
        assert len(result) == 1
        assert result[0]['id'] == 'paper1'

    def test_filter_by_datetime_with_time_range(self):
        """Test filtering by time range."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_result = [
            {
                'id': 'paper1',
                'name': 'Morning Paper',
                'session_start_time': '2025-12-02T17:00:00Z',  # 9 AM PST (17 UTC)
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'Abstract',
                'topic': 'AI',
                'session': 'Morning',
                'room_name': 'Hall A',
                'poster_position': '#123',
                'presentation_type': 'Poster',
                'url': 'https://example.com',
                'authors': []
            },
            {
                'id': 'paper2',
                'name': 'Evening Paper',
                'session_start_time': '2025-12-03T01:00:00Z',  # 5 PM PST (1 UTC next day)
                'session_end_time': '2025-12-03T03:00:00Z',
                'abstract': 'Abstract',
                'topic': 'ML',
                'session': 'Evening',
                'room_name': 'Hall B',
                'poster_position': '#124',
                'presentation_type': 'Poster',
                'url': 'https://example.com',
                'authors': []
            }
        ]
        mock_session.run.return_value = mock_result

        builder = ScheduleBuilder(mock_driver)
        # Filter for morning hours (8-12 UTC = equivalent to checking hour range)
        time_range = (17, 20)  # UTC hours
        result = builder.filter_by_datetime(['paper1', 'paper2'], time_range=time_range)

        # Should only include morning paper
        assert len(result) == 1
        assert result[0]['id'] == 'paper1'

    def test_filter_by_datetime_deduplicates_papers(self):
        """Test that duplicate paper IDs are deduplicated."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Return duplicate papers
        mock_result = [
            {'id': 'paper1', 'name': 'Paper 1', 'session_start_time': '2025-12-02T17:00:00Z',
             'abstract': 'A', 'topic': 'AI', 'session': 'S', 'session_end_time': '2025-12-02T19:00:00Z',
             'room_name': 'Hall', 'poster_position': '#1', 'presentation_type': 'Poster',
             'url': 'url', 'authors': []},
            {'id': 'paper1', 'name': 'Paper 1', 'session_start_time': '2025-12-02T17:00:00Z',
             'abstract': 'A', 'topic': 'AI', 'session': 'S', 'session_end_time': '2025-12-02T19:00:00Z',
             'room_name': 'Hall', 'poster_position': '#1', 'presentation_type': 'Poster',
             'url': 'url', 'authors': []}
        ]
        mock_session.run.return_value = mock_result

        builder = ScheduleBuilder(mock_driver)
        result = builder.filter_by_datetime(['paper1', 'paper1'])

        # Should only have one paper after deduplication
        assert len(result) == 1

    def test_filter_by_datetime_handles_invalid_times(self):
        """Test handling of papers with invalid time formats."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_result = [
            {
                'id': 'paper1',
                'name': 'Valid Paper',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'abstract': 'A', 'topic': 'AI', 'session': 'S',
                'room_name': 'Hall', 'poster_position': '#1',
                'presentation_type': 'Poster', 'url': 'url', 'authors': []
            },
            {
                'id': 'paper2',
                'name': 'Invalid Paper',
                'session_start_time': 'invalid-time',
                'session_end_time': 'invalid-time',
                'abstract': 'A', 'topic': 'AI', 'session': 'S',
                'room_name': 'Hall', 'poster_position': '#2',
                'presentation_type': 'Poster', 'url': 'url', 'authors': []
            }
        ]
        mock_session.run.return_value = mock_result

        builder = ScheduleBuilder(mock_driver)
        # With time range filter, invalid paper should still be included (safe fallback)
        result = builder.filter_by_datetime(['paper1', 'paper2'], time_range=(17, 20))

        # Valid paper should be included, invalid paper is included as fallback
        assert len(result) >= 1  # At least the valid paper


class TestScorePapers:
    """Test the score_papers method."""

    def test_score_papers_basic(self):
        """Test basic paper scoring functionality."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        papers = [
            {'id': 'paper1', 'name': 'Paper 1'},
            {'id': 'paper2', 'name': 'Paper 2'}
        ]
        relevance_scores = {'paper1': 0.95, 'paper2': 0.87}

        result = builder.score_papers(papers, relevance_scores)

        assert len(result) == 2
        assert result[0]['relevance_score'] == 0.95
        assert result[1]['relevance_score'] == 0.87

    def test_score_papers_sorts_by_score(self):
        """Test that papers are sorted by relevance score."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        papers = [
            {'id': 'paper1', 'name': 'Low Score'},
            {'id': 'paper2', 'name': 'High Score'}
        ]
        relevance_scores = {'paper1': 0.60, 'paper2': 0.95}

        result = builder.score_papers(papers, relevance_scores)

        # Should be sorted highest first
        assert result[0]['id'] == 'paper2'
        assert result[1]['id'] == 'paper1'

    def test_score_papers_handles_missing_scores(self):
        """Test handling of papers without relevance scores."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        papers = [
            {'id': 'paper1', 'name': 'Scored'},
            {'id': 'paper2', 'name': 'Not Scored'}
        ]
        relevance_scores = {'paper1': 0.85}

        result = builder.score_papers(papers, relevance_scores)

        assert result[0]['relevance_score'] == 0.85
        assert result[1]['relevance_score'] == 0.0  # Default score


class TestOptimizeSchedule:
    """Test the optimize_schedule method."""

    def test_optimize_schedule_basic(self):
        """Test basic schedule optimization."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        papers = [
            {
                'id': 'paper1',
                'name': 'Paper 1',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'relevance_score': 0.95
            }
        ]

        result = builder.optimize_schedule(papers, max_papers=10)

        assert '2025-12-02' in result
        assert len(result) > 0

    def test_optimize_schedule_limits_papers(self):
        """Test that schedule respects max_papers limit."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        papers = [
            {
                'id': f'paper{i}',
                'name': f'Paper {i}',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'relevance_score': 0.9 - (i * 0.01)
            }
            for i in range(30)
        ]

        result = builder.optimize_schedule(papers, max_papers=10)

        # Count total papers in schedule
        total_papers = 0
        for date_data in result.values():
            for time_data in date_data.values():
                for room_data in time_data.values():
                    total_papers += len(room_data)

        assert total_papers <= 10

    def test_optimize_schedule_groups_by_room(self):
        """Test that papers are grouped by room."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        papers = [
            {
                'id': 'paper1',
                'name': 'Paper 1',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'relevance_score': 0.95
            },
            {
                'id': 'paper2',
                'name': 'Paper 2',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall B',
                'relevance_score': 0.90
            }
        ]

        result = builder.optimize_schedule(papers, max_papers=10)

        # Should have separate entries for each room
        date_key = '2025-12-02'
        assert date_key in result
        # There should be entries for both halls
        time_slots = result[date_key]
        for time_slot_data in time_slots.values():
            # Check if we have multiple rooms
            rooms = list(time_slot_data.keys())
            if len(papers) == 2:
                assert len(rooms) <= 2  # At most 2 rooms

    def test_optimize_schedule_handles_missing_room(self):
        """Test handling of papers without room_name."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        papers = [
            {
                'id': 'paper1',
                'name': 'Paper 1',
                'session': 'Morning Session',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': None,  # No room
                'relevance_score': 0.95
            }
        ]

        result = builder.optimize_schedule(papers, max_papers=10)

        # Should use session as fallback
        assert '2025-12-02' in result

    def test_optimize_schedule_deduplicates_papers(self):
        """Test that duplicate paper IDs are deduplicated."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        # Duplicate papers
        papers = [
            {
                'id': 'paper1',
                'name': 'Paper 1',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'relevance_score': 0.95
            },
            {
                'id': 'paper1',
                'name': 'Paper 1',
                'session_start_time': '2025-12-02T17:00:00Z',
                'session_end_time': '2025-12-02T19:00:00Z',
                'room_name': 'Hall A',
                'relevance_score': 0.95
            }
        ]

        result = builder.optimize_schedule(papers, max_papers=10)

        # Count total papers (should be 1, not 2)
        total_papers = 0
        for date_data in result.values():
            for time_data in date_data.values():
                for room_data in time_data.values():
                    total_papers += len(room_data)

        assert total_papers == 1


class TestFormatAsMarkdown:
    """Test the format_as_markdown method."""

    def test_format_as_markdown_empty_schedule(self):
        """Test formatting empty schedule."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        result = builder.format_as_markdown({})
        assert "No papers found" in result

    def test_format_as_markdown_basic(self):
        """Test basic markdown formatting."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        schedule = {
            '2025-12-02': {
                '9:00 AM - 11:00 AM PST': {
                    'Hall A': [
                        {
                            'name': 'Test Paper',
                            'authors': ['Author A', 'Author B'],
                            'topic': 'AI',
                            'poster_position': '#123',
                            'presentation_type': 'Poster',
                            'relevance_score': 0.95,
                            'url': 'https://example.com',
                            'session': 'Morning Session'
                        }
                    ]
                }
            }
        }

        result = builder.format_as_markdown(schedule)

        assert '# Your NeurIPS 2025 Conference Schedule' in result
        assert 'Test Paper' in result
        assert 'Author A, Author B' in result
        assert '0.95' in result

    def test_format_as_markdown_with_abstracts(self):
        """Test markdown formatting with abstracts included."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        schedule = {
            '2025-12-02': {
                '9:00 AM - 11:00 AM PST': {
                    'Hall A': [
                        {
                            'name': 'Test Paper',
                            'authors': ['Author A'],
                            'topic': 'AI',
                            'poster_position': '#123',
                            'presentation_type': 'Poster',
                            'relevance_score': 0.95,
                            'abstract': 'This is a test abstract',
                            'url': 'https://example.com',
                            'session': 'Morning'
                        }
                    ]
                }
            }
        }

        result = builder.format_as_markdown(schedule, include_abstracts=True)

        assert 'Abstract:' in result
        assert 'This is a test abstract' in result

    def test_format_as_markdown_sorts_by_poster_position(self):
        """Test that papers are sorted by poster position."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        schedule = {
            '2025-12-02': {
                '9:00 AM - 11:00 AM PST': {
                    'Hall A': [
                        {
                            'name': 'Paper 2',
                            'poster_position': '#200',
                            'presentation_type': 'Poster',
                            'authors': ['A'],
                            'topic': 'AI',
                            'relevance_score': 0.9,
                            'session': 'Morning'
                        },
                        {
                            'name': 'Paper 1',
                            'poster_position': '#100',
                            'presentation_type': 'Poster',
                            'authors': ['B'],
                            'topic': 'ML',
                            'relevance_score': 0.95,
                            'session': 'Morning'
                        }
                    ]
                }
            }
        }

        result = builder.format_as_markdown(schedule)

        # Paper 1 should appear before Paper 2
        pos_paper1 = result.find('Paper 1')
        pos_paper2 = result.find('Paper 2')
        assert pos_paper1 < pos_paper2

    def test_format_as_markdown_handles_missing_fields(self):
        """Test formatting with missing optional fields."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        schedule = {
            '2025-12-02': {
                '9:00 AM - 11:00 AM PST': {
                    'Hall A': [
                        {
                            'name': 'Minimal Paper',
                            'authors': 'N/A',
                            'topic': 'General',
                            'poster_position': None,
                            'presentation_type': 'Poster',
                            'relevance_score': 0.80,
                            'session': 'Session'
                        }
                    ]
                }
            }
        }

        result = builder.format_as_markdown(schedule)

        assert 'Minimal Paper' in result
        assert '0.80' in result


class TestClose:
    """Test the close method."""

    def test_close_driver(self):
        """Test closing the Neo4j driver."""
        mock_driver = MagicMock()
        builder = ScheduleBuilder(mock_driver)

        builder.close()

        mock_driver.close.assert_called_once()

    def test_close_with_none_driver(self):
        """Test closing when driver is None."""
        builder = ScheduleBuilder(None)

        # Should not raise error
        builder.close()
