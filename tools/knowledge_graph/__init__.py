"""
This file defines the tools that can be made available to an agent.
The idea is to put the actual functions into wrappers that provide LLM-friendly and token efficient outputs.
"""
import random

from toon import encode as toon_encode

from .retriever import Neo4jGraphWorker

from typing import List


def search_similar_papers(
    user_query: str,
    num_papers_to_return: int = 10,
    min_similarity: float = None
) -> str:

    # TODO: The Neo4j worker should be wrapped in a session to better handle concurrent connections.
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
):
    """
    This functions requires to first run a similarity search and ask the user whether they want to find neighboring
    papers for a paper they are interested in.
    Available relationship types: ['SIMILAR_TO', 'AUTHORED_BY', 'BELONGS_TO_TOPIC']
    Available related entities: ['similar_papers', 'authors', 'topics', 'raw_results']
    """
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


