"""
Session routing tool for building personalized conference visiting schedules.

This tool helps NeurIPS 2025 conference attendees create optimized schedules
for visiting poster sessions based on their research interests, preferred dates,
and time slots.
"""

import os
from typing import Union, List, Optional
from neo4j import GraphDatabase

from llm_agents.tools.knowledge_graph import search_similar_papers
from llm_agents.tools.session_routing.scheduler import ScheduleBuilder
from llm_agents.tools.session_routing.utils import parse_date_input, parse_time_preference


# Environment variables for Neo4j connection
NEO4J_DB_URI = os.getenv("NEO4J_DB_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


def build_visit_schedule(
    topics: Union[str, List[str]],
    dates: Union[str, List[str]] = None,
    time_preferences: str = None,
    max_papers: int = 20,
    min_similarity: float = 0.6
) -> str:
    """
    Build a personalized visiting schedule for NeurIPS 2025 conference poster sessions.

    This tool helps you create an optimized schedule by:
    1. Finding papers relevant to your research interests (topics)
    2. Filtering by your preferred dates and time slots
    3. Scoring papers by relevance to your topics
    4. Clustering papers by room location to minimize walking
    5. Organizing chronologically for easy navigation

    The schedule includes paper titles, locations, poster positions, and relevance scores.

    Args:
        topics: Research topic(s) of interest. Can be a single topic string or a list of topics.
                Examples: "transformer architectures", ["reinforcement learning", "multi-agent systems"]
        dates: Conference date(s) to include. Can be:
               - ISO format: "2025-12-02" or ["2025-12-02", "2025-12-03"]
               - Day names: "Tuesday", "Wednesday"
               - None (default): include all conference days (Dec 2-7, 2025)
        time_preferences: Preferred time slot(s). Can be:
                         - Preset: "morning" (8am-12pm), "afternoon" (12pm-5pm), "evening" (5pm-9pm)
                         - Range: "9:00-12:00" or "14-17"
                         - None (default): include all time slots
        max_papers: Maximum number of papers to include in schedule (default: 20)
        min_similarity: Minimum similarity score for paper relevance (0.0-1.0, default: 0.6)

    Returns:
        Formatted markdown schedule organized by date, time slot, and room location.
        All times are displayed in conference local time (PST/UTC-8).

    Restrictions:
        - Requires Neo4j database connection (NEO4J_DB_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        - Requires Paper nodes with session timing and location fields
        - Conference dates: December 2-7, 2025 in San Diego/Mexico City (UTC-8)

    Notes:
        - Papers are scored by similarity to your topics using embedding search
        - Schedule optimizes for both relevance and room clustering
        - Time zones are automatically converted from UTC to PST
        - Poster positions help you quickly locate papers in exhibition halls

    Raises:
        ValueError: If topics is empty or dates cannot be parsed
        Exception: If Neo4j connection fails

    Example:
        >>> build_visit_schedule(
        ...     topics=["machine learning", "computer vision"],
        ...     dates="2025-12-02",
        ...     time_preferences="morning",
        ...     max_papers=15
        ... )
        # Your NeurIPS 2025 Conference Schedule

        ## Tuesday, December 2, 2025

        ### 9:00 AM - 11:00 AM PST

        **Hall A**
        - **Poster #123** | Attention Mechanisms in Vision Transformers
          - Authors: John Doe, Jane Doe, et al.
          - Topic: Computer Vision
          - Relevance: 0.92
        ...
    """
    # Type coercion for parameters that may come as strings from LLM tool calls
    if isinstance(topics, str):
        # If topics is a single string, treat as one topic
        topics = [topics]
    elif topics is None:
        raise ValueError("Topics parameter is required. Please provide at least one research topic.")

    if max_papers is not None and not isinstance(max_papers, int):
        max_papers = int(max_papers)

    if min_similarity is not None and not isinstance(min_similarity, float):
        min_similarity = float(min_similarity)

    # Parse dates
    parsed_dates = None
    if dates:
        if isinstance(dates, str):
            dates = [dates]

        parsed_dates = []
        for date_str in dates:
            parsed = parse_date_input(date_str)
            if parsed:
                parsed_dates.append(parsed)

        if not parsed_dates:
            parsed_dates = None  # Fall back to all dates if parsing fails

    # Parse time preferences (convert to UTC for database query)
    time_range = None
    if time_preferences:
        local_time_range = parse_time_preference(time_preferences)
        if local_time_range:
            # Convert PST to UTC (add 8 hours)
            start_utc = (local_time_range[0] + 8) % 24
            end_utc = (local_time_range[1] + 8) % 24
            time_range = (start_utc, end_utc)

    # Step 1: Search for papers matching each topic using existing tool
    all_paper_ids = set()
    relevance_scores = {}

    for topic in topics:
        try:

            from llm_agents.tools.knowledge_graph.retriever import Neo4jGraphWorker

            worker = Neo4jGraphWorker(
                uri=NEO4J_DB_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            )

            papers = worker.similarity_search(
                user_query=topic,
                top_k=max_papers * 2,
                min_similarity=min_similarity
            )

            worker.close()

            # Extract paper IDs and scores
            for paper in papers:
                paper_id = paper.get('id')
                score = paper.get('score', 0.0)

                if paper_id:
                    all_paper_ids.add(paper_id)
                    # Keep highest score if paper matches multiple topics
                    if paper_id not in relevance_scores or score > relevance_scores[paper_id]:
                        relevance_scores[paper_id] = score

        except Exception as e:
            # If search fails for one topic, continue with others
            continue

    if not all_paper_ids:
        return "No papers found matching your topics. Try broadening your search criteria or adjusting the minimum similarity threshold."

    # Step 2: Initialize schedule builder
    driver = GraphDatabase.driver(NEO4J_DB_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    builder = ScheduleBuilder(driver)

    try:
        # Step 3: Filter papers by date and time
        filtered_papers = builder.filter_by_datetime(
            paper_ids=list(all_paper_ids),
            dates=parsed_dates,
            time_range=time_range
        )

        if not filtered_papers:
            return "No papers found matching your date and time preferences. Try expanding your time range or selecting different dates."

        # Step 4: Score papers by relevance
        scored_papers = builder.score_papers(filtered_papers, relevance_scores)

        # Step 5: Optimize schedule (chronological + room clustering)
        schedule = builder.optimize_schedule(scored_papers, max_papers=max_papers)

        # Step 6: Format as markdown
        markdown_output = builder.format_as_markdown(schedule, include_abstracts=False)

        return markdown_output

    finally:
        builder.close()


__all__ = ['build_visit_schedule']

if __name__ == "__main__":
    print(build_visit_schedule(topics=["federated learning"], max_papers=200, dates=["Wednesday"]))