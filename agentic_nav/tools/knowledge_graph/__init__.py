"""
This file defines the tools that can be made available to an agent.
The idea is to put the actual functions into wrappers that provide LLM-friendly and token efficient outputs.
"""
import os
import random

from toon_format import encode as toon_encode
from typing import List, Optional, Union

from agentic_nav.tools.knowledge_graph.retriever import Neo4jGraphWorker, LOGGER

NEO4J_DB_URI = os.environ.get("NEO4J_DB_URI", "bolt://neo4j_db:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")


def search_similar_papers(
        user_query: str,
        num_papers_to_return: int = 50,
        min_similarity: float = None,
        day: str = None,
        timeslots: List[str] = None
) -> str:
    """
    Search for research papers semantically similar to a user's natural language query.

    This function performs vector similarity search against a Neo4j knowledge graph database
    to find papers that match the semantic meaning of the user's query. It serves as the
    entry point for paper discovery workflows and is typically followed by neighborhood
    or graph traversal searches for deeper exploration.

    Args:
        user_query (str): Natural language query describing the research topic or interest.
            The query is embedded and compared against paper embeddings in the database.
        num_papers_to_return (int, optional): Maximum number of papers to return, ranked by
            similarity score. Defaults to 50.
        min_similarity (float, optional): Minimum similarity threshold for returned papers.
            Defaults to None (no filtering). Should be a value between 0.0 and 1.0, where
            higher values indicate stricter similarity requirements.
        day (str, optional): Conference day as a date string in ISO format (e.g., "2024-12-10").
            When provided, only papers scheduled on this day will be searched. Defaults to None
            (no day filtering).
        timeslots (List[str], optional): List of time ranges to filter papers by their session
            times. Each timeslot should be formatted as "HH:MM:SS-HH:MM:SS" (e.g.,
            ["09:00:00-12:00:00", "14:00:00-17:00:00"]). Papers with session start times
            falling within any of these ranges will be included. Defaults to None (no time filtering).

    Returns:
        str: A token-efficient formatted string representation of papers matching the query,
            encoded using the toon_encode function. Papers are typically ordered by
            descending similarity score.

    Restrictions:
        - Requires a running Neo4j database instance at bolt://localhost:7687 with credentials
          (username: "neo4j", password: "llm_agents")
        - The database must have pre-computed embeddings for papers to enable similarity search
        - The database must have a vector index configured for efficient similarity queries
        - Currently creates a new database connection for each function call, which may not be
          optimal for concurrent usage (see TODO note)

    Notes:
        - This function is designed as the initial step in a multi-stage paper discovery workflow
        - Results can be further explored using find_neighboring_papers() or traverse_graph()
        - When day and/or timeslots are provided, the database filters papers by their session
          times BEFORE performing vector similarity search for better performance
        - TODO: The Neo4jGraphWorker should be wrapped in a session to better handle
          concurrent connections and connection pooling

    Raises:
        Connection errors if Neo4j database is not accessible
        ValueError if min_similarity is outside the valid range [0.0, 1.0]
        ValueError if day is not in valid ISO date format (YYYY-MM-DD)
        ValueError if timeslots are not properly formatted
        Embedding errors if the query cannot be properly embedded

    Example:
        >>> # Basic similarity search
        >>> papers = search_similar_papers(
        ...     user_query="federated learning for privacy-preserving machine learning",
        ...     num_papers_to_return=15
        ... )
        >>>
        >>> # Search with similarity threshold
        >>> highly_relevant_papers = search_similar_papers(
        ...     user_query="transformer architectures for NLP",
        ...     num_papers_to_return=20,
        ...     min_similarity=0.75
        ... )
        >>>
        >>> # Search for papers on a specific day and time
        >>> morning_papers = search_similar_papers(
        ...     user_query="computer vision applications",
        ...     num_papers_to_return=50,
        ...     day="2024-12-10",
        ...     timeslots=["09:00:00-12:00:00"]
        ... )
        >>>
        >>> # Search across multiple timeslots on a specific day
        >>> daytime_papers = search_similar_papers(
        ...     user_query="reinforcement learning",
        ...     num_papers_to_return=25,
        ...     day="2024-12-11",
        ...     timeslots=["09:00:00-12:00:00", "14:00:00-17:00:00"]
        ... )
    """
    # Type coercion for parameters that may come as strings from LLM tool calls
    if num_papers_to_return is not None and not isinstance(num_papers_to_return, int):
        num_papers_to_return = int(num_papers_to_return)
    if min_similarity is not None and not isinstance(min_similarity, float):
        min_similarity = float(min_similarity)

    # Handle timeslots - ensure it's a list or None
    if timeslots is not None and isinstance(timeslots, str):
        # If a single string is provided, wrap it in a list
        timeslots = [timeslots]

    worker = Neo4jGraphWorker(
        uri=NEO4J_DB_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

    # Fetch papers with optional day and time filtering
    papers = worker.similarity_search(
        user_query=user_query,
        top_k=num_papers_to_return,
        min_similarity=min_similarity,
        day=day,
        timeslots=timeslots
    )

    # Format outputs to be more token efficient
    formatted_papers = toon_encode(papers)

    return formatted_papers


def find_neighboring_papers(
        paper_id: str,
        relationship_types: Union[List[str], str] = ["SIMILAR_TO"],
        num_neighbors_to_return: int = 10,
        min_similarity: float = 0.75
) -> str:
    """
    [Your existing docstring]
    """
    # Type coercion for parameters that may come as strings from LLM tool calls
    if num_neighbors_to_return is not None and not isinstance(num_neighbors_to_return, int):
        num_neighbors_to_return = int(num_neighbors_to_return)

    if type(relationship_types) is str:
        relationship_types = [relationship_types]

    worker = Neo4jGraphWorker(
        uri=NEO4J_DB_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

    neighbors = worker.neighborhood_search(
        paper_id=paper_id,
        relationship_types=relationship_types,
        min_similarity=min_similarity,
    )

    # Flatten all neighbors from all relationship types into one list
    relevant_neighbors = []
    for rel_type, neighbor_list in neighbors.items():
        # neighbor_list is a list of paper dicts, extend to flatten
        relevant_neighbors.extend(neighbor_list)

    # Constrain and shuffle neighbors for more diverse responses
    random.shuffle(relevant_neighbors)

    if num_neighbors_to_return is not None and isinstance(num_neighbors_to_return, int):
        relevant_neighbors = relevant_neighbors[:num_neighbors_to_return]

    # Format outputs to be more token efficient
    formatted_neighbors = toon_encode(relevant_neighbors)

    return formatted_neighbors


def traverse_graph(
        start_paper_id: str,
        n_hops: int = 2,
        relationship_type: Optional[str] = "BELONGS_TO_TOPIC",
        max_results: Optional[int] = 30,
        strategy: str = "breadth_first_random",
        max_branches: Optional[int] = 2,
        random_seed: Optional[int] = 42
) -> str:
    """
    Traverse a Neo4j knowledge graph to discover related research papers through various relationship types.

    This function performs exploratory graph traversal starting from a seed paper to find potentially
    interesting related papers. It is designed to be used after an initial similarity search, allowing
    users to discover papers through different connection paths (topics, authors, similarity).

    Args:
        start_paper_id (str): The unique identifier of the starting paper node in the graph. neo4j UUID.
        n_hops (int, optional): Number of relationship hops to traverse from the starting paper.
            Defaults to 2. Higher values explore further but may return less relevant results.
        relationship_type (str, optional): Types of relationships to follow during traversal.
            Defaults to "BELONGS_TO_TOPIC".
            Valid options: ["SIMILAR_TO", "AUTHORED_BY", "BELONGS_TO_TOPIC"]
        max_results (int, optional): Maximum number of papers to return. Defaults to 30.
        strategy (str, optional): Graph traversal strategy to use. Defaults to "breadth_first_random".
            Valid options: ["breadth_first", "depth_first", "breadth_first_random", "depth_first_random"]
        max_branches (int, optional): Maximum number of branches to follow at each node during traversal.
            Defaults to 2. Controls the breadth of exploration at each step.
        random_seed (int, optional): Seed for random number generation in randomized strategies.
            Defaults to 42. Ensures reproducible results when using random strategies.

    Returns:
        str: A formatted string representation of discovered papers, encoded using the toon_encode function.

    Restrictions:
        - Requires a running Neo4j database instance at bolt://localhost:7687 with credentials
          (username: "neo4j", password: "llm_agents")
        - Should be used after an initial similarity search as part of an exploratory workflow
        - The start_paper_id must exist in the Neo4j graph database
        - Only the three specified relationship types are supported
        - Only the four specified traversal strategies are supported
        - Random strategies require random_seed for reproducibility

    Raises:
        Connection errors if Neo4j database is not accessible
        ValueError if invalid relationship_types or strategy are provided

    Example:
        >>> related_papers = traverse_graph(
        ...     start_paper_id="paper_12345",
        ...     n_hops=3,
        ...     relationship_type="SIMILAR_TO",
        ...     max_results=50,
        ...     strategy="breadth_first_random"
        ... )
    """
    # Type coercion for parameters that may come as strings from LLM tool calls
    if n_hops is not None and not isinstance(n_hops, int):
        n_hops = int(n_hops)
    if max_results is not None and not isinstance(max_results, int):
        max_results = int(max_results)
    if max_branches is not None and not isinstance(max_branches, int):
        max_branches = int(max_branches)
    if random_seed is not None and not isinstance(random_seed, int):
        random_seed = int(random_seed)

    worker = Neo4jGraphWorker(
        uri=NEO4J_DB_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

    papers = worker.graph_traversal(
        start_paper_id=start_paper_id,
        n_hops=n_hops,
        relationship_type=relationship_type,
        max_results=max_results,
        strategy=strategy,
        max_branches=max_branches,
        random_seed=random_seed
    )

    formatted_neighbors = toon_encode(papers)

    return formatted_neighbors
