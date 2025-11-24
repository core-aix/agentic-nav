"""
Schedule builder for NeurIPS 2025 conference paper sessions.

This module provides the ScheduleBuilder class that handles filtering,
scoring, and organizing papers into optimized visiting schedules.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import neo4j

from agentic_nav.tools.session_routing.utils import (
    convert_utc_to_local,
    format_time_slot,
    format_date_header,
    cluster_papers_by_room,
    parse_time_preference
)


class ScheduleBuilder:
    """
    Build optimized conference visiting schedules.

    This class handles filtering papers by date/time, scoring by relevance,
    clustering by room location, and formatting the final schedule.
    """

    def __init__(self, neo4j_driver: neo4j.Driver):
        """
        Initialize the schedule builder.

        Args:
            neo4j_driver: Neo4j database driver for querying papers
        """
        self.driver = neo4j_driver

    def filter_by_datetime(
        self,
        paper_ids: List[str],
        dates: Optional[List[datetime]] = None,
        time_range: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter papers by date and time preferences.

        Args:
            paper_ids: List of paper IDs to filter
            dates: List of conference dates to include (None = all dates)
            time_range: Tuple of (start_hour, end_hour) in UTC (None = all times)

        Returns:
            List of paper dictionaries with full details including session times

        Example:
            >>> builder.filter_by_datetime(['paper1', 'paper2'], dates=[datetime(2025,12,2)])
        """
        if not paper_ids:
            return []

        # Deduplicate paper_ids to ensure we only query each paper once
        unique_paper_ids = list(set(paper_ids))

        # Build Cypher query to get full paper details including authors via relationship
        # Relationship is IS_AUTHOR_OF (uppercase) and author property is 'fullname'
        query = """
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        OPTIONAL MATCH (a:Author)-[:IS_AUTHOR_OF]-(p)
        WITH p, collect(a.fullname) as authors
        RETURN DISTINCT p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic,
               p.session as session,
               p.session_start_time as session_start_time,
               p.session_end_time as session_end_time,
               p.room_name as room_name,
               p.poster_position as poster_position,
               p.presentation_type as presentation_type,
               p.url as url,
               authors
        """

        with self.driver.session() as session:
            result = session.run(query, paper_ids=unique_paper_ids)
            papers = [dict(record) for record in result]

        # Deduplicate papers by ID (just in case)
        seen_ids = set()
        unique_papers = []
        for paper in papers:
            paper_id = paper.get('id')
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)

        papers = unique_papers

        # Filter by date if specified
        if dates:
            date_strs = [d.strftime("%Y-%m-%d") for d in dates]
            papers = [
                p for p in papers
                if p.get('session_start_time') and
                   any(date_str in p['session_start_time'] for date_str in date_strs)
            ]

        # Filter by time range if specified (convert UTC time range)
        if time_range:
            start_hour, end_hour = time_range
            filtered_papers = []

            for paper in papers:
                try:
                    start_time_str = paper.get('session_start_time', '')
                    if not start_time_str:
                        continue

                    # Parse UTC time
                    if 'T' in start_time_str:
                        dt = datetime.fromisoformat(start_time_str.replace('Z', ''))
                    else:
                        continue

                    # Check if paper session falls within time range (UTC)
                    if start_hour <= dt.hour < end_hour:
                        filtered_papers.append(paper)

                except (ValueError, AttributeError):
                    # If we can't parse time, include the paper to be safe
                    filtered_papers.append(paper)

            papers = filtered_papers

        return papers

    def score_papers(
        self,
        papers: List[Dict[str, Any]],
        relevance_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Add relevance scores to papers.

        Args:
            papers: List of paper dictionaries
            relevance_scores: Dict mapping paper_id to relevance score

        Returns:
            Papers with added 'relevance_score' field, sorted by score descending

        Example:
            >>> builder.score_papers(papers, {'paper1': 0.95, 'paper2': 0.87})
        """
        scored_papers = []

        for paper in papers:
            paper_id = paper.get('id')
            score = relevance_scores.get(paper_id, 0.0)

            paper_with_score = paper.copy()
            paper_with_score['relevance_score'] = score
            scored_papers.append(paper_with_score)

        # Sort by relevance score (highest first)
        scored_papers.sort(key=lambda p: p['relevance_score'], reverse=True)

        return scored_papers

    def optimize_schedule(
        self,
        papers: List[Dict[str, Any]],
        max_papers: int = 20
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Optimize schedule by grouping papers chronologically and by room.

        Args:
            papers: List of scored paper dictionaries
            max_papers: Maximum number of papers to include

        Returns:
            Nested dict: {date: {time_slot: {room: [papers]}}}

        Example:
            >>> schedule = builder.optimize_schedule(papers, max_papers=15)
        """
        # Deduplicate papers by ID first
        seen_ids = set()
        unique_papers = []
        for paper in papers:
            paper_id = paper.get('id')
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)

        # Limit to top papers by relevance
        top_papers = unique_papers[:max_papers]

        # Group by date and time
        schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for paper in top_papers:
            try:
                start_time = paper.get('session_start_time', '')
                if not start_time:
                    continue

                # Extract date
                date_str = start_time.split('T')[0]

                # Create time slot key
                end_time = paper.get('session_end_time', '')
                time_slot = format_time_slot(start_time, end_time) if end_time else start_time

                # Get room (handle None values, fallback to session for Mexico City papers)
                room = paper.get('room_name')
                if not room:
                    # Use session as fallback (e.g., for Mexico City papers)
                    room = paper.get('session') or 'N/A'

                # Add to schedule
                schedule[date_str][time_slot][room].append(paper)

            except (ValueError, AttributeError, IndexError):
                # Skip papers with invalid time data
                continue

        return schedule

    def format_as_markdown(
        self,
        schedule: Dict[str, Dict[str, List[Dict[str, Any]]]],
        include_abstracts: bool = False
    ) -> str:
        """
        Format schedule as structured markdown.

        Args:
            schedule: Nested schedule dictionary
            include_abstracts: Whether to include paper abstracts (default: False)

        Returns:
            Formatted markdown string with format:
            "Date (MM dd, yyyy) - Time Slot - Session Name - Location"

        Example:
            >>> markdown = builder.format_as_markdown(schedule)
        """
        if not schedule:
            return "No papers found matching your criteria."

        output = ["# Your NeurIPS 2025 Conference Schedule\n"]

        # Flatten schedule into list of blocks for better formatting
        schedule_blocks = []

        for date_str in sorted(schedule.keys()):
            time_slots = schedule[date_str]

            for time_slot in sorted(time_slots.keys()):
                rooms = time_slots[time_slot]

                for room_or_session in sorted(rooms.keys()):
                    papers_in_block = rooms[room_or_session]

                    # Sort papers by poster position ID (numerically)
                    def poster_sort_key(paper):
                        poster_pos = paper.get('poster_position')
                        if not poster_pos:
                            return float('inf')  # Put papers without position at end

                        # Remove '#' prefix if present
                        if isinstance(poster_pos, str) and poster_pos.startswith('#'):
                            poster_pos = poster_pos[1:]

                        # Convert to integer for numerical sorting
                        try:
                            return int(poster_pos)
                        except (ValueError, TypeError):
                            return float('inf')  # Put invalid positions at end

                    papers_in_block.sort(key=poster_sort_key)

                    schedule_blocks.append({
                        'date': date_str,
                        'time_slot': time_slot,
                        'room_or_session': room_or_session,
                        'papers': papers_in_block
                    })

        # Format each schedule block
        total_papers = 0
        for block in schedule_blocks:
            date_str = block['date']
            time_slot = block['time_slot']
            room_or_session = block['room_or_session']
            papers = block['papers']

            total_papers += len(papers)

            # Get session and location from first paper (all papers in block share these)
            if papers:
                first_paper = papers[0]
                session_name = first_paper.get('session', 'N/A')
                actual_room = first_paper.get('room_name')

                # Determine location: use room if available, otherwise indicate session-based location
                if actual_room:
                    location = actual_room
                else:
                    location = "Mexico City"  # Papers without room are from Mexico City

            else:
                session_name = room_or_session
                location = room_or_session

            # Format date as "Month DD, YYYY"
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(date_str)
                formatted_date = dt.strftime("%B %d, %Y")
            except:
                formatted_date = date_str

            # Create a comprehensive header
            header = f"## {formatted_date} - {time_slot} - {session_name} - {location}\n"
            output.append(f"\n{header}")

            # List papers in this block
            for paper in papers:
                title = paper.get('name', 'Untitled')
                poster_pos = paper.get('poster_position', 'N/A')
                # TODO: This needs to be the distance between the user input query and the paper embedding, i.e.,
                #   compare encoded user_input with "embedding" in database.
                relevance = paper.get('relevance_score', 0)
                topic = paper.get('topic', 'General')
                pres_type = paper.get('presentation_type', 'Poster')
                authors = paper.get('authors', 'N/A')

                # Format authors for display
                if isinstance(authors, list):
                    authors_str = ', '.join(authors) if authors else 'N/A'
                elif authors and authors != 'N/A':
                    authors_str = str(authors)
                else:
                    authors_str = 'N/A'

                # Format paper entry
                output.append(f"- **{pres_type} {poster_pos.replace('#', '') if poster_pos is not None else ''}** | {title}")
                output.append(f"  - Authors: {authors_str}")
                output.append(f"  - Topic: {topic}")

                # Add paper URL if available
                paper_url = paper.get('url')
                if paper_url:
                    output.append(f"  - URL: {paper_url}")

                output.append(f"  - Relevance: {relevance:.2f}")

                if include_abstracts and paper.get('abstract'):
                    abstract = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
                    output.append(f"  - Abstract: {abstract}")

                output.append("")  # Blank line between papers

        # Add summary footer
        output.append(f"\n---\n**Total Papers in Schedule: {total_papers}**")

        return "\n".join(output)

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
