from typing import List, Dict, Any, Optional, Set
import random

import neo4j


def _graph_traversal_dfs_random(
    db_driver: neo4j.Driver,
    start_paper_id: str,
    n_hops: int,
    relationship_types: Optional[List[str]],
    max_results: Optional[int],
    max_branches: int
) -> List[Dict[str, Any]]:
    """
    DFS traversal with random neighbor sampling.
    Explores deeply along random branches before backtracking.
    """
    with db_driver.session() as session:
        visited: Set[str] = {start_paper_id}
        papers = []

        # Build relationship type filter
        if relationship_types:
            rel_filter = f":{':'.join(relationship_types)}"
        else:
            rel_filter = ""

        def dfs_traverse(paper_id: str, distance: int):
            """Recursive DFS helper"""
            if max_results and len(papers) >= max_results:
                return

            if distance >= n_hops:
                return

            # Query to get all neighbors
            query = f"""
            MATCH (p:Paper {{id: $paper_id}})-[r{rel_filter}]->(neighbor:Paper)
            RETURN neighbor.id as id,
                   neighbor.name as name,
                   neighbor.abstract as abstract,
                   neighbor.topic as topic
            """

            result = session.run(query, paper_id=paper_id)
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
                        if max_results and len(papers) >= max_results:
                            return

                        visited.add(neighbor_id)

                        paper = {
                            'id': neighbor_id,
                            'name': record['name'],
                            'abstract': record['abstract'],
                            'topic': record['topic'],
                            'distance': distance + 1
                        }
                        papers.append(paper)

                        # Recursively explore this branch
                        dfs_traverse(neighbor_id, distance + 1)

        dfs_traverse(start_paper_id, 0)
        return papers