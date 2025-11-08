from typing import List, Dict, Any, Optional

import neo4j


_DB_GRAPH_TRAVERSAL_QUERY = lambda rel_filter, n_hops: f"""
    MATCH path = (start:Paper)-[{rel_filter}*1..{n_hops}]-(related:Paper)
    WHERE start.id IN $start_paper_ids
    AND related.id <> start.id
    WITH related, min(length(path)) as min_distance
    RETURN DISTINCT related.id as id,
           related.name as name,
           related.abstract as abstract,
           related.topic as topic,
           min_distance as distance
    ORDER BY min_distance, related.name
    """


def _graph_traversal_cypher(
    db_driver: neo4j.Driver,
    start_paper_id: str,
    n_hops: int,
    relationship_type: Optional[str],
    max_results: Optional[int]
) -> List[Dict[str, Any]]:
    """Original Cypher-based traversal (BFS/DFS handled by Neo4j)"""
    with db_driver.session() as session:
        if relationship_type:
            rel_filter = f":{':'.join([relationship_type])}"
        else:
            rel_filter = ""

        query = _DB_GRAPH_TRAVERSAL_QUERY(rel_filter=rel_filter, n_hops=n_hops)
        if max_results:
            query += f" LIMIT {max_results}"

        result = session.run(query, start_paper_ids=[start_paper_id])
        papers = []
        for record in result:
            paper = {
                'id': record['id'],
                'name': record['name'],
                'abstract': record['abstract'],
                'topic': record['topic'],
                'distance': record['distance']
            }
            papers.append(paper)

        return papers