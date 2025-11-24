"""
Utility functions for session routing and schedule building.

This module provides helper functions for time zone conversion,
date parsing, and formatting schedule outputs.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import re


def convert_utc_to_local(utc_time_str: str, timezone_offset: int = -8) -> str:
    """
    Convert UTC time string to local conference time.

    Args:
        utc_time_str: ISO format UTC time string (e.g., "2025-12-02T17:00:00Z")
        timezone_offset: Hours offset from UTC (default: -8 for PST/Mexico City)

    Returns:
        Local time string in format "9:00 AM PST"

    Raises:
        ValueError: If time string cannot be parsed

    Example:
        >>> convert_utc_to_local("2025-12-02T17:00:00Z")
        "9:00 AM PST"
    """
    try:
        # Handle various UTC time formats
        utc_time_str = utc_time_str.strip()
        if utc_time_str.endswith('Z'):
            utc_time_str = utc_time_str[:-1]
        elif '+' in utc_time_str or utc_time_str.count('-') > 2:
            # Has timezone info, extract just the datetime part
            utc_time_str = utc_time_str.split('+')[0].split('T')[0] + 'T' + utc_time_str.split('T')[1].split('+')[0].split('-')[0]

        # Parse the UTC time
        if 'T' in utc_time_str:
            utc_dt = datetime.fromisoformat(utc_time_str)
        else:
            # Try parsing without T separator
            utc_dt = datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S")

        # Apply timezone offset
        local_dt = utc_dt + timedelta(hours=timezone_offset)

        # Format as human-readable time
        hour = local_dt.hour
        minute = local_dt.minute
        am_pm = "AM" if hour < 12 else "PM"
        hour_12 = hour if hour <= 12 else hour - 12
        hour_12 = 12 if hour_12 == 0 else hour_12

        if minute == 0:
            time_str = f"{hour_12}:00 {am_pm} PST"
        else:
            time_str = f"{hour_12}:{minute:02d} {am_pm} PST"

        return time_str
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Could not parse time string '{utc_time_str}': {e}")


def parse_date_input(date_str: str) -> Optional[datetime]:
    """
    Parse flexible date input formats.

    Supports:
    - ISO format: "2025-12-02"
    - Day names: "Monday", "Tuesday", etc.
    - Relative: "today", "tomorrow"

    Args:
        date_str: Date string in various formats

    Returns:
        Datetime object or None if parsing fails

    Example:
        >>> parse_date_input("2025-12-02")
        datetime.datetime(2025, 12, 2, 0, 0)
    """
    if not date_str:
        return None

    date_str = date_str.strip().lower()

    # Try ISO format first
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        pass

    # Try common date formats
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Handle day names (for NeurIPS 2025: Dec 2-7, 2025)
    conference_start = datetime(2025, 12, 2)  # Tuesday
    day_mapping = {
        'monday': conference_start - timedelta(days=1),
        'tuesday': conference_start,
        'wednesday': conference_start + timedelta(days=1),
        'thursday': conference_start + timedelta(days=2),
        'friday': conference_start + timedelta(days=3),
        'saturday': conference_start + timedelta(days=4),
        'sunday': conference_start + timedelta(days=5),
    }

    if date_str in day_mapping:
        return day_mapping[date_str]

    return None


def parse_time_preference(time_pref: str) -> Optional[Tuple[int, int]]:
    """
    Parse time preference string into hour range.

    Args:
        time_pref: Time preference like "morning", "afternoon", "9:00-12:00"

    Returns:
        Tuple of (start_hour, end_hour) in 24-hour format, or None

    Example:
        >>> parse_time_preference("morning")
        (8, 12)
        >>> parse_time_preference("9:00-15:00")
        (9, 15)
    """
    if not time_pref:
        return None

    time_pref = time_pref.strip().lower()

    # Predefined time slots
    presets = {
        'morning': (8, 12),
        'afternoon': (12, 17),
        'evening': (17, 21),
        'early': (8, 10),
        'late': (19, 21),
    }

    if time_pref in presets:
        return presets[time_pref]

    # Parse time range format: "9:00-12:00" or "09:00-12:00" or "9-12"
    range_pattern = r'(\d{1,2})(?::(\d{2}))?[\s\-]+(\d{1,2})(?::(\d{2}))?'
    match = re.match(range_pattern, time_pref)

    if match:
        start_hour = int(match.group(1))
        end_hour = int(match.group(3))
        return (start_hour, end_hour)

    return None


def format_time_slot(start_time: str, end_time: str) -> str:
    """
    Format time slot for display.

    Args:
        start_time: Start time in UTC format
        end_time: End time in UTC format

    Returns:
        Formatted time range string

    Example:
        >>> format_time_slot("2025-12-02T17:00:00Z", "2025-12-02T19:00:00Z")
        "9:00 AM - 11:00 AM PST"
    """
    try:
        start_local = convert_utc_to_local(start_time)
        end_local = convert_utc_to_local(end_time)

        # Remove PST from start time if both are same timezone
        if start_local.endswith(' PST') and end_local.endswith(' PST'):
            start_local = start_local[:-4]

        return f"{start_local} - {end_local}"
    except ValueError:
        return f"{start_time} - {end_time}"


def format_date_header(date_str: str) -> str:
    """
    Format date for section headers.

    Args:
        date_str: Date string (ISO format or datetime)

    Returns:
        Formatted date like "Tuesday, December 2, 2025"

    Example:
        >>> format_date_header("2025-12-02")
        "Tuesday, December 2, 2025"
    """
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.split('T')[0])
        else:
            dt = date_str

        return dt.strftime("%A, %B %d, %Y")
    except (ValueError, AttributeError):
        return str(date_str)


def cluster_papers_by_room(papers: list, time_slot_key: str = 'session') -> dict:
    """
    Group papers by room within their time slots.

    Args:
        papers: List of paper dictionaries with room_name and session info
        time_slot_key: Key to group by time slots (default: 'session')

    Returns:
        Nested dict: {time_slot: {room_name: [papers]}}

    Example:
        >>> papers = [
        ...     {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 1'},
        ...     {'session': 'Morning', 'room_name': 'Hall A', 'name': 'Paper 2'},
        ... ]
        >>> cluster_papers_by_room(papers)
        {'Morning': {'Hall A': [...]}}
    """
    clustered = {}

    for paper in papers:
        time_slot = paper.get(time_slot_key, 'Unknown Session')
        room = paper.get('room_name', 'Unknown Room')

        if time_slot not in clustered:
            clustered[time_slot] = {}

        if room not in clustered[time_slot]:
            clustered[time_slot][room] = []

        clustered[time_slot][room].append(paper)

    return clustered
