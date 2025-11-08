import neo4j
from typing import List, Dict, Any, Optional, Set
from collections import deque
import random


def _graph_traversal_bfs_random(
    db_driver: neo4j.Driver,
    start_paper_id: str,
    n_hops: int,
    relationship_type: Optional[str],
    max_results: Optional[int],
    max_branches: int
) -> List[Dict[str, Any]]:
    """
    BFS traversal with random neighbor sampling.
    Explores level by level, randomly sampling neighbors at each level.
    """
    with db_driver.session() as session:
        visited: Set[str] = {start_paper_id}
        queue = deque([(start_paper_id, 0)])  # (paper_id, distance)
        papers = []

        # Build relationship type filter
        if relationship_type:
            rel_filter = f":{':'.join([relationship_type])}"
        else:
            rel_filter = ""

        while queue:
            if max_results and type(max_results) is int and len(papers) >= max_results:
                break

            current_id, distance = queue.popleft()

            # Stop if we've reached max depth
            if distance >= n_hops:
                continue

            # Query to get all neighbors
            query = f"""
            MATCH (p:Paper {{id: $paper_id}})-[r{rel_filter}]->(neighbor:Paper)
            RETURN neighbor.id as id,
                   neighbor.name as name,
                   neighbor.abstract as abstract,
                   neighbor.topic as topic
            """

            result = session.run(query, paper_id=current_id)
            neighbors = list(result)

            # Randomly sample neighbors
            if neighbors:
                sampled_neighbors = random.sample(
                    neighbors,
                    min(max_branches, len(neighbors))
                )

                for record in sampled_neighbors:
                    neighbor_id = record['id']

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)

                        paper = {
                            'id': neighbor_id,
                            'name': record['name'],
                            'abstract': record['abstract'],
                            'topic': record['topic'],
                            'distance': distance + 1
                        }
                        papers.append(paper)

                        # Add to queue for next level
                        queue.append((neighbor_id, distance + 1))

                        if max_results and type(max_results) is int and len(papers) >= max_results:
                            break

        return papers