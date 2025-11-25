"""
Tests for session routing utility functions.
"""
import pytest
from datetime import datetime

from agentic_nav.tools.session_routing.utils import (
    convert_utc_to_local,
    parse_date_input,
    parse_time_preference,
    format_time_slot,
    format_date_header,
    cluster_papers_by_room
)


class TestConvertUtcToLocal:
    """Test the convert_utc_to_local function."""

    def test_convert_utc_to_local_basic(self):
        """Test basic UTC to local time conversion."""
        result = convert_utc_to_local("2025-12-02T17:00:00Z")
        assert result == "9:00 AM PST"

    def test_convert_utc_to_local_afternoon(self):
        """Test conversion for afternoon time."""
        result = convert_utc_to_local("2025-12-02T20:00:00Z")
        assert result == "12:00 PM PST"

    def test_convert_utc_to_local_evening(self):
        """Test conversion for evening time."""
        result = convert_utc_to_local("2025-12-02T01:00:00Z")
        assert result == "5:00 PM PST"  # Previous day

    def test_convert_utc_to_local_with_minutes(self):
        """Test conversion preserves minutes."""
        result = convert_utc_to_local("2025-12-02T17:30:00Z")
        assert result == "9:30 AM PST"

    def test_convert_utc_to_local_custom_offset(self):
        """Test conversion with custom timezone offset."""
        result = convert_utc_to_local("2025-12-02T17:00:00Z", timezone_offset=-5)
        assert result == "12:00 PM PST"  # EST offset

    def test_convert_utc_to_local_without_z(self):
        """Test conversion without Z suffix."""
        result = convert_utc_to_local("2025-12-02T17:00:00")
        assert result == "9:00 AM PST"

    def test_convert_utc_to_local_invalid_format(self):
        """Test error handling for invalid time format."""
        with pytest.raises(ValueError):
            convert_utc_to_local("invalid-time")

    def test_convert_utc_to_local_midnight(self):
        """Test conversion for midnight."""
        result = convert_utc_to_local("2025-12-02T08:00:00Z")
        assert result == "12:00 AM PST"

    def test_convert_utc_to_local_noon(self):
        """Test conversion for noon."""
        result = convert_utc_to_local("2025-12-02T20:00:00Z")
        assert result == "12:00 PM PST"


class TestParseDateInput:
    """Test the parse_date_input function."""

    def test_parse_date_input_iso_format(self):
        """Test parsing ISO format date."""
        result = parse_date_input("2025-12-02")
        assert result == datetime(2025, 12, 2, 0, 0)

    def test_parse_date_input_day_names(self):
        """Test parsing day names for conference dates."""
        # Conference starts on Tuesday, Dec 2, 2025
        tuesday_result = parse_date_input("tuesday")
        assert tuesday_result == datetime(2025, 12, 2)

        wednesday_result = parse_date_input("wednesday")
        assert wednesday_result == datetime(2025, 12, 3)

    def test_parse_date_input_case_insensitive(self):
        """Test that day names are case insensitive."""
        result1 = parse_date_input("Tuesday")
        result2 = parse_date_input("TUESDAY")
        result3 = parse_date_input("tuesday")

        assert result1 == result2 == result3

    def test_parse_date_input_with_whitespace(self):
        """Test handling of leading/trailing whitespace."""
        result = parse_date_input("  2025-12-02  ")
        assert result == datetime(2025, 12, 2, 0, 0)

    def test_parse_date_input_empty_string(self):
        """Test handling of empty string."""
        result = parse_date_input("")
        assert result is None

    def test_parse_date_input_none(self):
        """Test handling of None input."""
        result = parse_date_input(None)
        assert result is None

    def test_parse_date_input_invalid_format(self):
        """Test handling of invalid date format."""
        result = parse_date_input("invalid-date")
        assert result is None

    def test_parse_date_input_various_formats(self):
        """Test parsing various date formats."""
        # ISO format
        result1 = parse_date_input("2025-12-02")
        assert result1 == datetime(2025, 12, 2)

        # MM/DD/YYYY format
        result2 = parse_date_input("12/02/2025")
        assert result2 == datetime(2025, 12, 2)


class TestParseTimePreference:
    """Test the parse_time_preference function."""

    def test_parse_time_preference_morning(self):
        """Test parsing 'morning' preset."""
        result = parse_time_preference("morning")
        assert result == (8, 12)

    def test_parse_time_preference_afternoon(self):
        """Test parsing 'afternoon' preset."""
        result = parse_time_preference("afternoon")
        assert result == (12, 17)

    def test_parse_time_preference_evening(self):
        """Test parsing 'evening' preset."""
        result = parse_time_preference("evening")
        assert result == (17, 21)

    def test_parse_time_preference_early(self):
        """Test parsing 'early' preset."""
        result = parse_time_preference("early")
        assert result == (8, 10)

    def test_parse_time_preference_late(self):
        """Test parsing 'late' preset."""
        result = parse_time_preference("late")
        assert result == (19, 21)

    def test_parse_time_preference_range_with_colon(self):
        """Test parsing time range with colons."""
        result = parse_time_preference("9:00-12:00")
        assert result == (9, 12)

    def test_parse_time_preference_range_without_colon(self):
        """Test parsing simple hour range."""
        result = parse_time_preference("9-12")
        assert result == (9, 12)

    def test_parse_time_preference_range_with_space(self):
        """Test parsing time range with space separator."""
        result = parse_time_preference("9:00 - 15:00")
        assert result == (9, 15)

    def test_parse_time_preference_case_insensitive(self):
        """Test that presets are case insensitive."""
        result1 = parse_time_preference("Morning")
        result2 = parse_time_preference("MORNING")
        result3 = parse_time_preference("morning")

        assert result1 == result2 == result3

    def test_parse_time_preference_with_whitespace(self):
        """Test handling of whitespace."""
        result = parse_time_preference("  morning  ")
        assert result == (8, 12)

    def test_parse_time_preference_empty_string(self):
        """Test handling of empty string."""
        result = parse_time_preference("")
        assert result is None

    def test_parse_time_preference_none(self):
        """Test handling of None input."""
        result = parse_time_preference(None)
        assert result is None

    def test_parse_time_preference_invalid_format(self):
        """Test handling of invalid format."""
        result = parse_time_preference("invalid-time")
        assert result is None


class TestFormatTimeSlot:
    """Test the format_time_slot function."""

    def test_format_time_slot_basic(self):
        """Test basic time slot formatting."""
        result = format_time_slot(
            "2025-12-02T17:00:00Z",
            "2025-12-02T19:00:00Z"
        )
        assert result == "9:00 AM - 11:00 AM PST"

    def test_format_time_slot_cross_meridiem(self):
        """Test formatting across AM/PM boundary."""
        result = format_time_slot(
            "2025-12-02T19:00:00Z",
            "2025-12-02T21:00:00Z"
        )
        assert result == "11:00 AM - 1:00 PM PST"

    def test_format_time_slot_with_minutes(self):
        """Test formatting with non-zero minutes."""
        result = format_time_slot(
            "2025-12-02T17:30:00Z",
            "2025-12-02T19:30:00Z"
        )
        assert result == "9:30 AM - 11:30 AM PST"

    def test_format_time_slot_invalid_times(self):
        """Test handling of invalid time strings."""
        result = format_time_slot("invalid", "invalid")
        assert result == "invalid - invalid"


class TestFormatDateHeader:
    """Test the format_date_header function."""

    def test_format_date_header_iso_string(self):
        """Test formatting from ISO date string."""
        result = format_date_header("2025-12-02")
        assert result == "Tuesday, December 02, 2025"

    def test_format_date_header_datetime_object(self):
        """Test formatting from datetime object."""
        dt = datetime(2025, 12, 2)
        result = format_date_header(dt)
        assert result == "Tuesday, December 02, 2025"

    def test_format_date_header_with_time(self):
        """Test formatting from ISO string with time."""
        result = format_date_header("2025-12-02T10:00:00")
        assert result == "Tuesday, December 02, 2025"

    def test_format_date_header_invalid_format(self):
        """Test handling of invalid date format."""
        result = format_date_header("invalid-date")
        assert result == "invalid-date"

    def test_format_date_header_different_dates(self):
        """Test formatting for different dates."""
        result1 = format_date_header("2025-12-03")
        assert "Wednesday" in result1

        result2 = format_date_header("2025-12-04")
        assert "Thursday" in result2


class TestClusterPapersByRoom:
    """Test the cluster_papers_by_room function."""

    def test_cluster_papers_by_room_basic(self):
        """Test basic paper clustering by room."""
        papers = [
            {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 1'},
            {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 2'},
            {'session': 'Morning', 'room_name': 'Hall B', 'name': 'Paper 3'},
        ]

        result = cluster_papers_by_room(papers)

        assert 'Morning' in result
        assert 'Hall A' in result['Morning']
        assert 'Hall B' in result['Morning']
        assert len(result['Morning']['Hall A']) == 2
        assert len(result['Morning']['Hall B']) == 1

    def test_cluster_papers_by_room_multiple_sessions(self):
        """Test clustering across multiple time slots."""
        papers = [
            {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 1'},
            {'session': 'Afternoon', 'room_name': 'Hall A', 'name': 'Paper 2'},
        ]

        result = cluster_papers_by_room(papers)

        assert 'Morning' in result
        assert 'Afternoon' in result
        assert len(result['Morning']['Hall A']) == 1
        assert len(result['Afternoon']['Hall A']) == 1

    def test_cluster_papers_by_room_missing_fields(self):
        """Test handling of papers with missing fields."""
        papers = [
            {'name': 'Paper 1'},  # Missing session and room_name
            {'session': 'Morning', 'name': 'Paper 2'},  # Missing room_name
        ]

        result = cluster_papers_by_room(papers)

        assert 'Unknown Session' in result
        assert 'Morning' in result
        assert 'Unknown Room' in result['Unknown Session']
        assert 'Unknown Room' in result['Morning']

    def test_cluster_papers_by_room_empty_list(self):
        """Test clustering empty paper list."""
        papers = []
        result = cluster_papers_by_room(papers)
        assert result == {}

    def test_cluster_papers_by_room_preserves_paper_data(self):
        """Test that paper data is preserved in clusters."""
        papers = [
            {
                'session': 'Morning',
                'room_name': 'Hall A',
                'name': 'Paper 1',
                'authors': ['Author A', 'Author B'],
                'id': 'paper_1'
            }
        ]

        result = cluster_papers_by_room(papers)

        paper = result['Morning']['Hall A'][0]
        assert paper['name'] == 'Paper 1'
        assert paper['authors'] == ['Author A', 'Author B']
        assert paper['id'] == 'paper_1'

    def test_cluster_papers_by_room_custom_key(self):
        """Test clustering with custom time slot key."""
        papers = [
            {'time_slot': 'Slot 1', 'room_name': 'Hall A', 'name': 'Paper 1'},
            {'time_slot': 'Slot 2', 'room_name': 'Hall A', 'name': 'Paper 2'},
        ]

        result = cluster_papers_by_room(papers, time_slot_key='time_slot')

        assert 'Slot 1' in result
        assert 'Slot 2' in result

    def test_cluster_papers_by_room_maintains_order(self):
        """Test that papers maintain their order within clusters."""
        papers = [
            {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 1'},
            {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 2'},
            {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 3'},
        ]

        result = cluster_papers_by_room(papers)

        hall_a_papers = result['Morning']['Hall A']
        assert hall_a_papers[0]['name'] == 'Paper 1'
        assert hall_a_papers[1]['name'] == 'Paper 2'
        assert hall_a_papers[2]['name'] == 'Paper 3'
