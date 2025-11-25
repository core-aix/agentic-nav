import logging
import numpy as np
import random
import os

from neo4j import GraphDatabase
from pathlib import Path

from typing import List, Dict, Any, Optional

from agentic_nav.tools.knowledge_graph.graph_traversal_strategies import (
    TraversalStrategy,
    _graph_traversal_dfs_random,
    _graph_traversal_cypher,
    _graph_traversal_bfs_random
)

from agentic_nav.utils.embedding_generator import batch_embed_documents


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOGGER = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "ollama/nomic-embed-text")
EMBEDDING_MODEL_API_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE", "http://localhost:11435")
NEO4J_DB_URI = os.environ.get("NEO4J_DB_URI", "bolt://neo4j_db:7687")
NEO4J_DB_NODE_RETURN_LIMIT = int(os.environ.get("NEO4J_DB_NODE_RETURN_LIMIT", 200))


class Neo4jGraphWorker:
    """Search and traversal operations for Neo4j paper knowledge graph."""

    _DB_SIMILARITY_SEARCH_QUERY = """
        MATCH (node:Paper)
        WHERE ($day IS NULL OR node.session_start_time IS NOT NULL)
        WITH node
        WHERE ($day IS NULL OR date(datetime(node.session_start_time)).dayOfWeek = $day)
        AND ($time_ranges IS NULL OR 
             any(range IN $time_ranges WHERE 
                 time(datetime(node.session_start_time)) >= time(range.start) 
                 AND time(datetime(node.session_start_time)) <= time(range.end)))
        WITH collect(node) as filtered_nodes
        CALL db.index.vector.queryNodes('paper_embeddings', $top_k, $query_embedding)
        YIELD node, score
        WHERE node IN filtered_nodes OR ($day IS NULL AND $time_ranges IS NULL)
        RETURN node.id as id,
               node.name as name,
               node.abstract as abstract,
               node.topic as topic,
               node.paper_url as paper_url, 
               node.session as session,
               node.session_start_time as session_start_time,
               node.session_end_time as session_end_time,
               node.presentation_type as presentation_type,
               node.room_name as room_name,
               node.project_url as project_url,
               node.poster_position as poster_position,
               node.sourceid as sourceid,
               node.virtualsite_url as virtualsite_url,
               node.decision as decision,
               [(a:Author)-[:IS_AUTHOR_OF]->(node) | a] as authors,
               score
        ORDER BY score DESC
        LIMIT $limit
        """

    _DB_NEIGHBORHOOD_SEARCH_QUERY = """
        MATCH (p:Paper)-[r]-(neighbor)
        WHERE p.id IN $paper_ids 
          AND type(r) IN $allowed_rel_types
          AND 'Paper' IN labels(neighbor)
          AND (type(r) <> 'SIMILAR_TO' OR r.similarity >= $min_similarity)
        RETURN neighbor.id as id,
               neighbor.name as name,
               neighbor.abstract as abstract,
               neighbor.topic as topic,
               neighbor.paper_url as paper_url, 
               neighbor.session as session,
               neighbor.session_start_time as session_start_time,
               neighbor.session_end_time as session_end_time,
               neighbor.presentation_type as presentation_type,
               neighbor.room_name as room_name,
               neighbor.project_url as project_url,
               neighbor.poster_position as poster_position,
               neighbor.sourceid as sourceid,
               neighbor.virtualsite_url as virtualsite_url,
               neighbor.decision as decision,
               [(a:Author)-[:IS_AUTHOR_OF]->(neighbor) | a] as authors,
               p.id as source_paper_id,
               type(r) as relationship_type, 
               CASE WHEN type(r) = 'SIMILAR_TO' THEN r.similarity ELSE null END as similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

    # Find the DB query for graph traversal in the graph_traversal sub-folder.
    _DB_PAPERS_BY_AUTHOR = """
        MATCH (a:Author)-[:IS_AUTHOR_OF]->(p:Paper)
        WHERE a.fullname = $author_name
        WITH p, collect(DISTINCT a) as all_authors
        RETURN p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic,
               p.paper_url as paper_url,
               p.decision as decision,
               p.session as session,
               p.session_start_time as session_start_time,
               p.session_end_time as session_end_time,
               p.presentation_type as presentation_type,
               p.room_name as room_name,
               p.project_url as project_url,
               p.poster_position as poster_position,
               p.sourceid as sourceid,
               p.virtualsite_url as virtualsite_url,
               all_authors as authors
        ORDER BY p.name
        LIMIT $limit
        """

    _DB_PAPERS_BY_AUTHOR_FUZZY = """
        MATCH (a:Author)-[:IS_AUTHOR_OF]->(p:Paper)
        WHERE toLower(a.fullname) CONTAINS toLower($author_name)
        WITH p, collect(DISTINCT a) as all_authors
        RETURN p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic,
               p.paper_url as paper_url,
               p.decision as decision,
               p.session as session,
               p.session_start_time as session_start_time,
               p.session_end_time as session_end_time,
               p.presentation_type as presentation_type,
               p.room_name as room_name,
               p.project_url as project_url,
               p.poster_position as poster_position,
               p.sourceid as sourceid,
               p.virtualsite_url as virtualsite_url,
               all_authors as authors
        ORDER BY p.name
        LIMIT $limit
        """

    _DB_PAPERS_BY_TOPIC = """
        MATCH (p:Paper)-[:BELONGS_TO_TOPIC]->(t:Topic {name: $topic_name})
        RETURN p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic,
               p.paper_url as paper_url,
               p.decision as decision,
               p.session as session,
               p.session_start_time as session_start_time,
               p.session_end_time as session_end_time,
               p.presentation_type as presentation_type,
               p.room_name as room_name,
               p.project_url as project_url,
               p.poster_position as poster_position,
               p.sourceid as sourceid,
               p.virtualsite_url as virtualsite_url,
               [(a:Author)-[:IS_AUTHOR_OF]->(p) | a] as authors
        ORDER BY p.name
        LIMIT $limit
        """

    _DB_PAPERS_BY_TOPIC_AND_SUBTOPIC = """
        MATCH (t:Topic {name: $topic_name})
        OPTIONAL MATCH (subtopic:Topic)-[:SUBTOPIC_OF*]->(t)
        WITH t, collect(DISTINCT subtopic) + t as all_topics
        UNWIND all_topics as topic
        MATCH (p:Paper)-[:BELONGS_TO_TOPIC]->(topic)
        WITH DISTINCT p
        RETURN p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic,
               p.paper_url as paper_url,
               p.decision as decision,
               p.session as session,
               p.session_start_time as session_start_time,
               p.session_end_time as session_end_time,
               p.presentation_type as presentation_type,
               p.room_name as room_name,
               p.project_url as project_url,
               p.poster_position as poster_position,
               p.sourceid as sourceid,
               p.virtualsite_url as virtualsite_url,
               [(a:Author)-[:IS_AUTHOR_OF]->(p) | a] as authors
        ORDER BY p.name
        LIMIT $limit
        """

    def __init__(
            self,
            uri: str = NEO4J_DB_URI,
            username: str = "neo4j",
            password: str = "password"
    ):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        LOGGER.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    @staticmethod
    def embed_user_query(
        text: str,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        api_base: str = EMBEDDING_MODEL_API_BASE
    ):
        emb = batch_embed_documents(
            texts=[text],
            batch_size=1,
            api_base=api_base,
            embedding_model=embedding_model
        ).tolist()[0]

        return emb

    def similarity_search(
            self,
            user_query: str,
            day: Optional[str] = None,
            timeslots: Optional[List[str]] = None,
            top_k: int = 5,
            min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on paper embeddings.

        Args:
            user_query: User query (str)
            day: Conference day as date string (e.g., "2024-12-10") or None
            timeslots: List of time ranges as strings (e.g., ["09:00:00-12:00:00"]) or None
            top_k: Number of top results to return
            min_similarity: Optional minimum similarity threshold (0-1)

        Returns:
            List of dictionaries containing paper information and similarity scores
        """

        # Generate text embedding
        query_embedding = self.embed_user_query(
            text=user_query
        )

        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Parse day and timeslots for the query
        day_filter = None
        time_ranges = []

        if day:
            # Convert date string to day of week (1=Monday, 7=Sunday)
            from datetime import datetime
            date_obj = datetime.strptime(day, "%Y-%m-%d")
            day_filter = date_obj.isoweekday()

        if timeslots:
            # Parse timeslot ranges (e.g., "09:00:00-12:00:00")
            for slot in timeslots:
                if '-' in slot:
                    start, end = slot.split('-')
                    time_ranges.append({'start': start.strip(), 'end': end.strip()})
                else:
                    # If no range, assume it's a single time point with some buffer
                    time_ranges.append({'start': slot.strip(), 'end': slot.strip()})

        with self.driver.session() as session:
            result = session.run(
                self._DB_SIMILARITY_SEARCH_QUERY,
                query_embedding=query_embedding,
                top_k=top_k,
                limit=NEO4J_DB_NODE_RETURN_LIMIT,
                day=day_filter,
                time_ranges=time_ranges if time_ranges else None
            )
            papers = []
            for record in result:
                paper = {
                    'id': record['id'],
                    'name': record['name'],
                    'abstract': record['abstract'],
                    'topic': record['topic'],
                    'similarity_score': record['score'],
                    'paper_url': record['paper_url'],
                    'decision': record['decision'],
                    'session': record['session'],
                    'session_start_time': record['session_start_time'],
                    'session_end_time': record['session_end_time'],
                    'presentation_type': record['presentation_type'],
                    'room_name': record['room_name'],
                    'github_url': record['project_url'],
                    'poster_position': record['poster_position'],
                    'sourceid': record['sourceid'],
                    'virtualsite_url': record['virtualsite_url'],
                    'authors': [a['fullname'] for a in record['authors']]
                }

                # Apply minimum similarity filter if specified
                if min_similarity is None or paper['similarity_score'] >= min_similarity:
                    # IMPORTANT: We don't return the similarity as the model has high affinity to scores like that...
                    del paper["similarity_score"]
                    papers.append(paper)

            return papers

    def neighborhood_search(
            self,
            paper_id: str,
            relationship_types: List[str] = ["SIMILAR_TO"],
            min_similarity: float = 0.7
    ) -> Dict[str, Any]:
        """
        Find immediate neighbors of given paper nodes.

        Args:
            paper_id: Paper ID to find neighbors for
            relationship_types: Optional list of relationship types to filter
                               (e.g., ['SIMILAR_TO', 'IS_AUTHOR_OF', 'BELONGS_TO_TOPIC', 'SUBTOPIC_OF'])
            min_similarity (float): A minimum similarity score in the range of 0 - 1. Often a good value is 0.75 or 0.8.


        Returns:
            Dictionary with neighbors grouped by relationship type
        """
        allowed_rel_types = ['SIMILAR_TO', 'IS_AUTHOR_OF', 'BELONGS_TO_TOPIC', 'SUBTOPIC_OF']
        for rel_type in relationship_types:
            if rel_type not in allowed_rel_types:
                raise ValueError(f"Unsupported relationship type: {rel_type}. Supported relationship types: {allowed_rel_types}")

        with self.driver.session() as session:
            result = session.run(
                self._DB_NEIGHBORHOOD_SEARCH_QUERY,
                paper_ids=[paper_id],
                allowed_rel_types=relationship_types,
                min_similarity=min_similarity,
                limit=NEO4J_DB_NODE_RETURN_LIMIT
            )

            # Organize results by relationship type
            neighbors = {}

            for record in result:
                # Use the dict() object in Record to manipulate the data. Records are immutable.
                record = record.data()
                rel_type = record["relationship_type"]
                if rel_type not in neighbors.keys():
                    neighbors[rel_type] = []
                else:
                    if "similarity" in record.keys():
                        # IMPORTANT: We don't return the similarity as the model has high affinity to scores like that...
                        del record["similarity"]

                    neighbors[rel_type].append(record)

            return neighbors

    def graph_traversal(
        self,
        start_paper_id: str,
        n_hops: int = 2,
        relationship_type: Optional[str] = None,
        max_results: Optional[int] = None,
        strategy: str = "breadth_first_random",
        max_branches: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Traverse the graph for n hops from starting paper nodes.

        Args:
            start_paper_id: Paper ID to start traversal from
            n_hops: Number of hops to traverse (1-5 recommended)
            relationship_type: Optional list of relationship types to traverse
            max_results: Optional maximum number of results to return
            strategy: Traversal strategy (breadth_first, depth_first, breadth_first_random, depth_first_random)
            max_branches: Maximum number of random neighbors to explore per node (only for random strategies)
            random_seed: Optional seed for reproducible random sampling

        Returns:
            List of papers found through traversal with distance information
        """
        if random_seed is not None:
            random.seed(random_seed)

        # Use original Cypher-based approach for non-random strategies
        if strategy in ["breadth_first", "depth_first"]:
            LOGGER.debug(f"Doing a graph traversal with neo4j's built-in strategy")
            return _graph_traversal_cypher(
                self.driver,
                start_paper_id,
                n_hops,
                relationship_type,
                max_results
            )

        # Use Python-based traversal for random strategies
        elif strategy == "breadth_first_random":
            LOGGER.debug(f"Doing a graph traversal with a random sampling breadth first strategy")
            return _graph_traversal_bfs_random(
                self.driver,
                start_paper_id,
                n_hops,
                relationship_type,
                max_results,
                max_branches or 3
            )

        elif strategy == "depth_first_random":
            LOGGER.debug(f"Doing a graph traversal with a random sampling depth first strategy")
            return _graph_traversal_dfs_random(
                self.driver,
                start_paper_id,
                n_hops,
                relationship_type,
                max_results,
                max_branches or 3
            )
        
        else:
            raise ValueError(f"Unsupported traversal strategy: {strategy}. "
                           f"Supported strategies: breadth_first, depth_first, breadth_first_random, depth_first_random")

    def search_papers_by_author(
            self,
            author_name: str,
            fuzzy: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find all papers by a specific author.

        Args:
            author_name: Author name or partial name
            fuzzy: Whether to use fuzzy matching (CONTAINS vs exact match)

        Returns:
            List of papers by the author
        """
        with self.driver.session() as session:
            if fuzzy:
                query = self._DB_PAPERS_BY_AUTHOR_FUZZY
            else:
                query = self._DB_PAPERS_BY_AUTHOR

            result = session.run(query, author_name=author_name)

            papers = []
            for record in result:
                paper = {
                    'id': record['id'],
                    'name': record['name'],
                    'abstract': record['abstract'],
                    'topic': record['topic'],
                    'author_name': record['author_name'],
                    'paper_url': record['paper_url'],
                    'decision': record['decision'],
                    'session': record['session'],
                    'session_start_time': record['session_start_time'],
                    'session_end_time': record['session_end_time'],
                    'presentation_type': record['presentation_type'],
                    'room_name': record['room_name'],
                    'github_url': record['project_url'],
                    'poster_position': record['poster_position'],
                    'sourceid': record['sourceid'],
                    'virtualsite_url': record['virtualsite_url'],
                }
                papers.append(paper)

            return papers

    def search_papers_by_topic(
            self,
            topic_name: str,
            include_subtopics: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find all papers in a specific topic.

        Args:
            topic_name: Topic name
            include_subtopics: Whether to include papers from subtopics

        Returns:
            List of papers in the topic
        """
        with self.driver.session() as session:
            if include_subtopics:
                # Find topic and all its subtopics
                query = self._DB_PAPERS_BY_TOPIC_AND_SUBTOPIC
            else:
                query = self._DB_PAPERS_BY_TOPIC

            result = session.run(query, topic_name=topic_name, limit=NEO4J_DB_NODE_RETURN_LIMIT)

            papers = []
            for record in result:
                paper = {
                    'id': record['id'],
                    'name': record['name'],
                    'abstract': record['abstract'],
                    'topic': record['topic'],
                    'paper_url': record['paper_url'],
                    'decision': record['decision'],
                    'session': record['session'],
                    'session_start_time': record['session_start_time'],
                    'session_end_time': record['session_end_time'],
                    'presentation_type': record['presentation_type'],
                    'room_name': record['room_name'],
                    'github_url': record['project_url'],
                    'poster_position': record['poster_position'],
                    'sourceid': record['sourceid'],
                    'virtualsite_url': record['virtualsite_url'],
                }
                papers.append(paper)

            return papers

    def get_collaboration_network(
            self,
            author_name: str,
            n_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Find collaboration network: authors who co-authored papers.

        Args:
            author_name: Starting author name
            n_hops: Degrees of separation to explore

        Returns:
            Dictionary with collaborators and shared papers
        """
        with self.driver.session() as session:
            query = f"""
                MATCH (a1:Author)
                WHERE toLower(a1.fullname) CONTAINS toLower($author_name)
                MATCH path = (a1)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
                WHERE a1 <> a2
                WITH a1, a2, collect(DISTINCT p) as shared_papers, length(path) as distance
                RETURN a1.fullname as source_author,
                       a2.fullname as collaborator,
                       a2.institution as institution,
                       [p IN shared_papers | {{id: p.id, name: p.name}}] as papers,
                       size(shared_papers) as paper_count
                ORDER BY paper_count DESC
            """

            result = session.run(query, author_name=author_name)

            collaborations = []
            for record in result:
                collab = {
                    'source_author': record['source_author'],
                    'collaborator': record['collaborator'],
                    'institution': record['institution'],
                    'shared_papers': record['papers'],
                    'paper_count': record['paper_count']
                }
                collaborations.append(collab)

            return {
                'author': author_name,
                'collaborators': collaborations,
                'total_collaborators': len(collaborations)
            }


# Test
if __name__ == "__main__":
    # Initialize searcher
    searcher = Neo4jGraphWorker(
        uri=NEO4J_DB_URI,
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD")
    )

    try:
        # Example 1: Similarity search
        print("\n" + "=" * 60)
        print("Example 1: Similarity Search")
        print("=" * 60)
        user_query = "Reinforcement learning"
        similar_papers = searcher.similarity_search(user_query, top_k=30)
        for i, paper in enumerate(similar_papers, 1):
            print(f"\n{i}. {paper['name']}")
            print(f"   Topic: {paper['topic']}")
            # print(f"   Similarity: {paper['similarity_score']:.4f}")

        # Example 2: Neighborhood search
        if similar_papers:
            print("\n" + "=" * 60)
            print("Example 2: Neighborhood Search")
            print("=" * 60)
            paper_id = similar_papers[0]['id']
            neighbors = searcher.neighborhood_search(paper_id, min_similarity=0.75)
            print(f"\nNeighbors of: {similar_papers[0]['name']}")
            for rel_type, neighbors in neighbors.items():
                print(f"    \n{rel_type.upper()} RELATIONSHIPS:")
                for neighbor in neighbors:
                    print(f"      - {neighbor['name']}")  # (similarity: {neighbor['similarity']:.4f})

        # Example 3: Graph traversal
        print("\n" + "=" * 60)
        print("Example 3: Graph Traversal (2 hops)")
        print("=" * 60)
        if similar_papers:
            paper_ids = similar_papers[0]['id']
            related = searcher.graph_traversal(paper_ids, n_hops=2)
            print(f"\nFound {len(related)} related papers through traversal")
            for paper in related[:5]:  # Show first 5
                print(f"  - {paper['name']} (distance: {paper['distance']})")

    finally:
        searcher.close()
