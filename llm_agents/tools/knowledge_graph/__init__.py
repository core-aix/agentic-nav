"""
This file defines the tools that can be made available to an agent.
The idea is to put the actual functions into wrappers that provide LLM-friendly and token efficient outputs.
"""
import random

from toon_format import encode as toon_encode
from typing import List, Optional

from llm_agents.tools.knowledge_graph.retriever import Neo4jGraphWorker


def search_similar_papers(
    user_query: str,
    num_papers_to_return: int = 10,
    min_similarity: float = None
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
            similarity score. Defaults to 10.
        min_similarity (float, optional): Minimum similarity threshold for returned papers.
            Defaults to None (no filtering). Should be a value between 0.0 and 1.0, where
            higher values indicate stricter similarity requirements.

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
        - TODO: The Neo4jGraphWorker should be wrapped in a session to better handle
          concurrent connections and connection pooling

    Raises:
        Connection errors if Neo4j database is not accessible
        ValueError if min_similarity is outside the valid range [0.0, 1.0]
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
    """
    worker = Neo4jGraphWorker(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="llm_agents"
    )

    # Fetch papers
    papers = worker.similarity_search(
        user_query=user_query,
        top_k=num_papers_to_return,
        min_similarity=min_similarity
    )

    # Format outputs to be more token efficient
    formatted_papers = toon_encode(papers)

    return formatted_papers


def find_neighboring_papers(
    paper_id: str,
    relationship_types: List[str] = ["SIMILAR_TO"],
    neighbor_entity: str = "similar_papers",
    num_neighbors_to_return: int = 10
) -> str:
    """
    Retrieve immediate neighboring entities of a specific paper from the Neo4j knowledge graph.

    This function performs a one-hop neighborhood search to find entities directly connected to
    a target paper. It is designed to be used after an initial similarity search when users want
    to explore specific relationships (similar papers, authors, or topics) for a paper of interest.

    Args:
        paper_id (str): The unique identifier of the target paper node in the graph. neo4j UUID.
        relationship_types (List[str], optional): Types of relationships to query.
            Defaults to ["SIMILAR_TO"].
            Valid options: ["SIMILAR_TO", "AUTHORED_BY", "BELONGS_TO_TOPIC"]
        neighbor_entity (str, optional): The type of neighboring entity to return.
            Defaults to "similar_papers".
            Valid options: ["similar_papers", "authors", "topics", "raw_results"]
        num_neighbors_to_return (int, optional): Maximum number of neighbors to return.
            Defaults to 10. Results are randomly shuffled before truncation to provide diversity.

    Returns:
        str: A token-efficient formatted string representation of neighboring entities,
            encoded using the toon_encode function.

    Restrictions:
        - Requires a running Neo4j database instance at bolt://localhost:7687 with credentials
          (username: "neo4j", password: "llm_agents")
        - Should be used after an initial similarity search as part of a focused exploration workflow
        - The paper_id must exist in the Neo4j graph database
        - Only performs one-hop searches (direct neighbors only)
        - Only the three specified relationship types are supported
        - Only the four specified neighbor entity types are supported
        - The neighbor_entity parameter must match the relationship_types used
          (e.g., "similar_papers" with "SIMILAR_TO", "authors" with "AUTHORED_BY")

    Notes:
        - Results are randomly shuffled to provide diverse recommendations across multiple calls
        - The function extracts only the "neighbor" data from the returned results
        - There is a potential bug: the type check `type(relevant_neighbors) is int` should likely be
          `type(num_neighbors_to_return) is int` for proper list truncation

    Raises:
        Connection errors if Neo4j database is not accessible
        KeyError if neighbor_entity doesn't exist in the returned neighbors dictionary
        ValueError if invalid relationship_types or neighbor_entity are provided

    Example:
        >>> similar_papers = find_neighboring_papers(
        ...     paper_id="<UUID>",
        ...     relationship_types=["SIMILAR_TO"],
        ...     neighbor_entity="similar_papers",
        ...     num_neighbors_to_return=5
        ... )
        >>>
        >>> authors = find_neighboring_papers(
        ...     paper_id="<UUID>",
        ...     relationship_types=["AUTHORED_BY"],
        ...     neighbor_entity="authors",
        ...     num_neighbors_to_return=3
        ... )
    """

    if type(relationship_types) is str:
        relationship_types = [relationship_types]

    worker = Neo4jGraphWorker(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="llm_agents"
    )

    neighbors = worker.neighborhood_search(
        paper_id=paper_id,
        relationship_types=relationship_types,
    )

    relevant_neighbors = []
    for neighbor in neighbors[neighbor_entity]:
        relevant_neighbors.append({
            **neighbor["neighbor"]
        })

    # Constrain and shuffle neighbors for more diverse responses
    random.shuffle(relevant_neighbors)

    if num_neighbors_to_return is not None and type(relevant_neighbors) is int:
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

    print(f"PAPER ID: {start_paper_id}")

    if type(relationship_types) is str:
        relationship_types = [relationship_types]

    worker = Neo4jGraphWorker(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="llm_agents"
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
