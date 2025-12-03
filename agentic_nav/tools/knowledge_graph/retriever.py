import logging
import numpy as np
import random
import os
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

from neo4j import GraphDatabase
from pathlib import Path

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
    """Search and traversal operations for Neo4j paper knowledge graph - OPTIMIZED."""

    # Optimized: Reduced property fetching, streamlined UNWIND logic
    _DB_SIMILARITY_SEARCH_QUERY = """
        CALL db.index.vector.queryNodes('paper_embeddings', $top_k, $query_embedding)
        YIELD node, score
        WHERE ($day IS NULL OR node.session_start_time IS NOT NULL)
          AND ($day IS NULL OR date(datetime(node.session_start_time)).dayOfWeek = $day)
          AND ($time_ranges IS NULL OR 
               any(range IN $time_ranges WHERE 
                   time(datetime(node.session_start_time)) >= time(range.start) 
                   AND time(datetime(node.session_start_time)) <= time(range.end)))
        
        // Deduplicate matched nodes and keep highest score
        WITH node, max(score) as score
        ORDER BY score DESC
        LIMIT $top_k
        
        // Fetch pair only once per matched paper
        OPTIONAL MATCH (node)-[:ORAL_POSTER_PAIR]-(pair:Paper)
        WHERE elementId(node) < elementId(pair)
        
        // Collect unique papers
        WITH node, pair, score
        UNWIND (CASE WHEN pair IS NULL THEN [node] ELSE [node, pair] END) as paper
        
        // Deduplicate by paper.id to ensure uniqueness
        WITH paper, max(score) as score
        
        // Get authors ordered by author_order
        OPTIONAL MATCH (a:Author)-[r:IS_AUTHOR_OF]->(paper)
        WITH paper, score, a, r
        ORDER BY r.author_order
        WITH paper, score, collect(a.fullname) as authors
        
        RETURN paper.id as id,
               paper.name as name,
               paper.abstract as abstract,
               paper.topic as topic,
               paper.paper_url as paper_url, 
               paper.session as session,
               paper.session_start_time as session_start_time,
               paper.session_end_time as session_end_time,
               paper.presentation_type as presentation_type,
               paper.presentation_category as presentation_category,
               paper.room_name as room_name,
               paper.project_url as project_url,
               paper.poster_position as poster_position,
               paper.sourceid as sourceid,
               paper.virtualsite_url as virtualsite_url,
               paper.decisions as decisions,
               authors,
               score
        ORDER BY score DESC
        LIMIT $limit
        """

    # Optimized: More efficient author list comprehension
    _DB_NEIGHBORHOOD_SEARCH_QUERY = """
        MATCH (p:Paper {id: $paper_id})-[r]-(neighbor:Paper)
        WHERE type(r) IN $allowed_rel_types
          AND (type(r) <> 'SIMILAR_TO' OR r.similarity >= $min_similarity)
        
        // Deduplicate neighbors (same neighbor might be found via different relationship types)
        WITH neighbor, p, max(CASE WHEN type(r) = 'SIMILAR_TO' THEN r.similarity ELSE 0 END) as similarity,
             collect(DISTINCT type(r)) as rel_types
        
        // For simplicity, use the first relationship type if multiple exist
        WITH neighbor, p, similarity, rel_types[0] as relationship_type
        
        // Fetch pair only once per neighbor
        OPTIONAL MATCH (neighbor)-[:ORAL_POSTER_PAIR]-(pair:Paper)
        WHERE elementId(neighbor) < elementId(neighbor)  // Only expand from one direction
        
        WITH neighbor, pair, p, relationship_type, similarity
        UNWIND CASE WHEN pair IS NULL THEN [neighbor] ELSE [neighbor, pair] END as result_paper
        
        RETURN result_paper.id as id,
               result_paper.name as name,
               result_paper.abstract as abstract,
               result_paper.topic as topic,
               result_paper.paper_url as paper_url, 
               result_paper.session as session,
               result_paper.session_start_time as session_start_time,
               result_paper.session_end_time as session_end_time,
               result_paper.presentation_type as presentation_type,
               result_paper.presentation_category as presentation_category,
               result_paper.room_name as room_name,
               result_paper.project_url as project_url,
               result_paper.poster_position as poster_position,
               result_paper.sourceid as sourceid,
               result_paper.virtualsite_url as virtualsite_url,
               result_paper.decisions as decisions,
               [(a:Author)-[:IS_AUTHOR_OF]->(result_paper) | a.fullname] as authors,
               p.id as source_paper_id,
               relationship_type, 
               CASE WHEN relationship_type = 'SIMILAR_TO' THEN similarity ELSE null END as similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

    _DB_PAPERS_BY_AUTHOR = """
        MATCH (a:Author {fullname: $author_name})-[:IS_AUTHOR_OF]->(p:Paper)
        
        // Collect papers first to prevent duplicates
        WITH collect(DISTINCT p) as papers
        UNWIND papers as p
        
        OPTIONAL MATCH (p)-[:ORAL_POSTER_PAIR]-(pair:Paper)
        WHERE elementId(p) < elementId(pair)  // Only expand from one direction
        
        WITH p, pair
        UNWIND CASE WHEN pair IS NULL THEN [p] ELSE [p, pair] END as paper
        
        RETURN paper.id as id,
               paper.name as name,
               paper.abstract as abstract,
               paper.topic as topic,
               paper.paper_url as paper_url,
               paper.decisions as decisions,
               paper.session as session,
               paper.session_start_time as session_start_time,
               paper.session_end_time as session_end_time,
               paper.presentation_type as presentation_type,
               paper.presentation_category as presentation_category,
               paper.room_name as room_name,
               paper.project_url as project_url,
               paper.poster_position as poster_position,
               paper.sourceid as sourceid,
               paper.virtualsite_url as virtualsite_url,
               [(a:Author)-[:IS_AUTHOR_OF]->(paper) | a.fullname] as authors
        ORDER BY paper.name
        LIMIT $limit
        """

    _DB_PAPERS_BY_AUTHOR_FUZZY = """
        MATCH (a:Author)-[:IS_AUTHOR_OF]->(p:Paper)
        WHERE toLower(a.fullname) CONTAINS toLower($author_name)
        
        // Collect papers first to prevent duplicates
        WITH collect(DISTINCT p) as papers
        UNWIND papers as p
        
        OPTIONAL MATCH (p)-[:ORAL_POSTER_PAIR]-(pair:Paper)
        WHERE elementId(p) < elementId(pair) // Only expand from one direction
        
        WITH p, pair
        UNWIND CASE WHEN pair IS NULL THEN [p] ELSE [p, pair] END as paper
        
        RETURN paper.id as id,
               paper.name as name,
               paper.abstract as abstract,
               paper.topic as topic,
               paper.paper_url as paper_url,
               paper.decisions as decisions,
               paper.session as session,
               paper.session_start_time as session_start_time,
               paper.session_end_time as session_end_time,
               paper.presentation_type as presentation_type,
               paper.presentation_category as presentation_category,
               paper.room_name as room_name,
               paper.project_url as project_url,
               paper.poster_position as poster_position,
               paper.sourceid as sourceid,
               paper.virtualsite_url as virtualsite_url,
               [(a:Author)-[:IS_AUTHOR_OF]->(paper) | a.fullname] as authors
        ORDER BY paper.name
        LIMIT $limit
        """

    _DB_PAPERS_BY_TOPIC = """
        MATCH (t:Topic {name: $topic_name})<-[:BELONGS_TO_TOPIC]-(p:Paper)
        
        // Collect papers first to prevent duplicates
        WITH collect(DISTINCT p) as papers
        UNWIND papers as p
        
        OPTIONAL MATCH (p)-[:ORAL_POSTER_PAIR]-(pair:Paper)
        WHERE elementId(p) < elementId(pair)  // Only expand from one direction
        
        WITH p, pair
        UNWIND CASE WHEN pair IS NULL THEN [p] ELSE [p, pair] END as paper
        
        RETURN paper.id as id,
               paper.name as name,
               paper.abstract as abstract,
               paper.topic as topic,
               paper.paper_url as paper_url,
               paper.decisions as decisions,
               paper.session as session,
               paper.session_start_time as session_start_time,
               paper.session_end_time as session_end_time,
               paper.presentation_type as presentation_type,
               paper.presentation_category as presentation_category,
               paper.room_name as room_name,
               paper.project_url as project_url,
               paper.poster_position as poster_position,
               paper.sourceid as sourceid,
               paper.virtualsite_url as virtualsite_url,
               [(a:Author)-[:IS_AUTHOR_OF]->(paper) | a.fullname] as authors
        ORDER BY paper.name
        LIMIT $limit
        """

    _DB_PAPERS_BY_TOPIC_AND_SUBTOPIC = """
        MATCH (t:Topic {name: $topic_name})
        OPTIONAL MATCH (subtopic:Topic)-[:SUBTOPIC_OF*]->(t)
        WITH collect(DISTINCT subtopic) + t as all_topics
        UNWIND all_topics as topic
        MATCH (topic)<-[:BELONGS_TO_TOPIC]-(p:Paper)
        
        // Collect papers to prevent duplicates from multiple topic paths
        WITH collect(DISTINCT p) as papers
        UNWIND papers as p
        
        OPTIONAL MATCH (p)-[:ORAL_POSTER_PAIR]-(pair:Paper)
        WHERE elementId(p) < elementId(pair)  // Only expand from one direction
        
        WITH p, pair
        UNWIND CASE WHEN pair IS NULL THEN [p] ELSE [p, pair] END as paper
        
        RETURN paper.id as id,
               paper.name as name,
               paper.abstract as abstract,
               paper.topic as topic,
               paper.paper_url as paper_url,
               paper.decisions as decisions,
               paper.session as session,
               paper.session_start_time as session_start_time,
               paper.session_end_time as session_end_time,
               paper.presentation_type as presentation_type,
               paper.presentation_category as presentation_category,
               paper.room_name as room_name,
               paper.project_url as project_url,
               paper.poster_position as poster_position,
               paper.sourceid as sourceid,
               paper.virtualsite_url as virtualsite_url,
               [(a:Author)-[:IS_AUTHOR_OF]->(paper) | a.fullname] as authors
        ORDER BY paper.name
        LIMIT $limit
        """

    def __init__(
            self,
            uri: str = NEO4J_DB_URI,
            username: str = "neo4j",
            password: str = "password",
            max_connection_lifetime: int = 3600,
            max_connection_pool_size: int = 50,
            connection_acquisition_timeout: int = 60
    ):
        """
        Initialize Neo4j connection with optimized settings.

        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password
            max_connection_lifetime: Max lifetime of connections in seconds (default: 3600)
            max_connection_pool_size: Max number of connections in pool (default: 50)
            connection_acquisition_timeout: Timeout for acquiring connection (default: 60s)
        """
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=connection_acquisition_timeout
        )
        self.driver.verify_connectivity()
        LOGGER.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    @staticmethod
    def _link_oral_poster_pairs(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Link Oral and Poster pairs by adding cross-references.
        OPTIMIZED: Single pass with early exits.

        Args:
            papers: List of paper dictionaries from database query

        Returns:
            List of papers with Oral-Poster pairs linked via new fields
        """
        if not papers:
            return papers

        # Build lookup dictionaries by sourceid in a single pass
        oral_map = {}  # abs(sourceid) -> paper record
        poster_map = {}  # sourceid -> paper record

        for paper in papers:
            sourceid = paper.get('sourceid')
            if sourceid is None:
                continue

            if sourceid < 0:
                oral_map[abs(sourceid)] = paper
            else:
                poster_map[sourceid] = paper

        # Add cross-references - only for papers that have pairs
        for abs_sourceid, oral in oral_map.items():
            poster = poster_map.get(abs_sourceid)
            if not poster:
                continue

            # Link them together
            oral['has_poster'] = True
            oral['poster_id'] = poster['id']
            oral['poster_session'] = poster.get('session')
            oral['poster_session_start_time'] = poster.get('session_start_time')
            oral['poster_session_end_time'] = poster.get('session_end_time')
            oral['poster_room_name'] = poster.get('room_name')
            oral['poster_position'] = poster.get('poster_position')

            # Replace Oral's paper_url with Poster's paper_url (OpenReview link)
            if poster.get('paper_url'):
                oral['paper_url'] = poster['paper_url']

            poster['has_oral'] = True
            poster['oral_id'] = oral['id']
            poster['oral_session'] = oral.get('session')
            poster['oral_session_start_time'] = oral.get('session_start_time')
            poster['oral_session_end_time'] = oral.get('session_end_time')
            poster['oral_room_name'] = oral.get('room_name')

        return papers

    @staticmethod
    def embed_user_query(
            text: str,
            embedding_model: str = EMBEDDING_MODEL_NAME,
            api_base: str = EMBEDDING_MODEL_API_BASE
    ) -> List[float]:
        """Generate embedding for user query. Returns a list of floats."""
        emb = batch_embed_documents(
            texts=[text],
            batch_size=1,
            api_base=api_base,
            embedding_model=embedding_model
        )

        # Convert to list if numpy array
        if isinstance(emb, np.ndarray):
            return emb.tolist()[0]
        return emb[0]

    @staticmethod
    def _parse_day_filter(day: Optional[str]) -> Optional[int]:
        """Parse day string to day of week integer."""
        if not day:
            return None
        from datetime import datetime
        date_obj = datetime.strptime(day, "%Y-%m-%d")
        return date_obj.isoweekday()

    @staticmethod
    def _parse_timeslots(timeslots: Optional[List[str]]) -> Optional[List[Dict[str, str]]]:
        """Parse timeslot strings to time ranges."""
        if not timeslots:
            return None

        time_ranges = []
        for slot in timeslots:
            if '-' in slot:
                start, end = slot.split('-', 1)
                time_ranges.append({'start': start.strip(), 'end': end.strip()})
            else:
                time_ranges.append({'start': slot.strip(), 'end': slot.strip()})
        return time_ranges

    @staticmethod
    def _build_paper_dict(record) -> Dict[str, Any]:
        """
        Build paper dictionary from Neo4j record.
        OPTIMIZED: Centralized dict construction.
        """
        return {
            'id': record['id'],
            'name': record['name'],
            'abstract': record['abstract'],
            'topic': record['topic'],
            'paper_url': record['paper_url'],
            'decisions': record['decisions'],
            'session': record['session'],
            'session_start_time': record['session_start_time'],
            'session_end_time': record['session_end_time'],
            'presentation_type': record['presentation_type'],
            'presentation_category': record['presentation_category'],
            'room_name': record['room_name'],
            'github_url': record['project_url'],
            'poster_position': record['poster_position'],
            'sourceid': record['sourceid'],
            'virtualsite_url': record['virtualsite_url'],
            'authors': record['authors']  # Already processed in Cypher
        }

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
        # Generate embedding
        query_embedding = self.embed_user_query(user_query)

        # Parse filters
        day_filter = self._parse_day_filter(day)
        time_ranges = self._parse_timeslots(timeslots)

        with self.driver.session() as session:
            result = session.run(
                self._DB_SIMILARITY_SEARCH_QUERY,
                query_embedding=query_embedding,
                top_k=top_k,
                limit=NEO4J_DB_NODE_RETURN_LIMIT,
                day=day_filter,
                time_ranges=time_ranges
            )

            papers = []
            for record in result:
                score = record['score']

                # Apply minimum similarity filter early
                if min_similarity is not None and score < min_similarity:
                    continue

                paper = self._build_paper_dict(record)
                papers.append(paper)

            # Link oral-poster pairs
            return self._link_oral_poster_pairs(papers)

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
            relationship_types: List of relationship types to filter
            min_similarity: Minimum similarity score (0-1)

        Returns:
            Dictionary with neighbors grouped by relationship type
        """
        allowed_rel_types = ['SIMILAR_TO', 'IS_AUTHOR_OF', 'BELONGS_TO_TOPIC', 'SUBTOPIC_OF', 'ORAL_POSTER_PAIR']

        # Validate relationship types
        invalid_types = set(relationship_types) - set(allowed_rel_types)
        if invalid_types:
            raise ValueError(
                f"Unsupported relationship type(s): {invalid_types}. "
                f"Supported types: {allowed_rel_types}"
            )

        with self.driver.session() as session:
            result = session.run(
                self._DB_NEIGHBORHOOD_SEARCH_QUERY,
                paper_id=paper_id,
                allowed_rel_types=relationship_types,
                min_similarity=min_similarity,
                limit=NEO4J_DB_NODE_RETURN_LIMIT
            )

            # Organize results by relationship type
            neighbors = {}
            for record in result:
                rel_type = record["relationship_type"]

                if rel_type not in neighbors:
                    neighbors[rel_type] = []

                paper = self._build_paper_dict(record)
                neighbors[rel_type].append(paper)

            # Link oral-poster pairs in each relationship type
            for rel_type in neighbors:
                neighbors[rel_type] = self._link_oral_poster_pairs(neighbors[rel_type])

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
            relationship_type: Optional relationship type to traverse
            max_results: Optional maximum number of results to return
            strategy: Traversal strategy (breadth_first, depth_first, breadth_first_random, depth_first_random)
            max_branches: Maximum number of random neighbors per node (only for random strategies)
            random_seed: Optional seed for reproducible random sampling

        Returns:
            List of papers found through traversal with distance information
        """
        if random_seed is not None:
            random.seed(random_seed)

        # Use Cypher-based approach for non-random strategies
        if strategy in ["breadth_first", "depth_first"]:
            LOGGER.debug("Using Cypher-based traversal strategy")
            papers = _graph_traversal_cypher(
                self.driver,
                start_paper_id,
                n_hops,
                relationship_type,
                max_results
            )
        elif strategy == "breadth_first_random":
            LOGGER.debug("Using BFS random sampling strategy")
            papers = _graph_traversal_bfs_random(
                self.driver,
                start_paper_id,
                n_hops,
                relationship_type,
                max_results,
                max_branches or 3
            )
        elif strategy == "depth_first_random":
            LOGGER.debug("Using DFS random sampling strategy")
            papers = _graph_traversal_dfs_random(
                self.driver,
                start_paper_id,
                n_hops,
                relationship_type,
                max_results,
                max_branches or 3
            )
        else:
            raise ValueError(
                f"Unsupported traversal strategy: {strategy}. "
                f"Supported: breadth_first, depth_first, breadth_first_random, depth_first_random"
            )

        # Link oral-poster pairs
        return self._link_oral_poster_pairs(papers)

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
        query = self._DB_PAPERS_BY_AUTHOR_FUZZY if fuzzy else self._DB_PAPERS_BY_AUTHOR

        with self.driver.session() as session:
            result = session.run(
                query,
                author_name=author_name,
                limit=NEO4J_DB_NODE_RETURN_LIMIT
            )

            papers = [self._build_paper_dict(record) for record in result]
            return self._link_oral_poster_pairs(papers)

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
        query = (self._DB_PAPERS_BY_TOPIC_AND_SUBTOPIC if include_subtopics
                 else self._DB_PAPERS_BY_TOPIC)

        with self.driver.session() as session:
            result = session.run(
                query,
                topic_name=topic_name,
                limit=NEO4J_DB_NODE_RETURN_LIMIT
            )

            papers = [self._build_paper_dict(record) for record in result]
            return self._link_oral_poster_pairs(papers)

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
            query = """
                MATCH (a1:Author)-[:IS_AUTHOR_OF]->(p:Paper)<-[:IS_AUTHOR_OF]-(a2:Author)
                WHERE toLower(a1.fullname) CONTAINS toLower($author_name)
                  AND a1 <> a2
                WITH a1, a2, collect(DISTINCT p) as shared_papers
                RETURN a1.fullname as source_author,
                       a2.fullname as collaborator,
                       a2.institution as institution,
                       [p IN shared_papers | {id: p.id, name: p.name}] as papers,
                       size(shared_papers) as paper_count
                ORDER BY paper_count DESC
            """

            result = session.run(query, author_name=author_name)

            collaborations = [
                {
                    'source_author': record['source_author'],
                    'collaborator': record['collaborator'],
                    'institution': record['institution'],
                    'shared_papers': record['papers'],
                    'paper_count': record['paper_count']
                }
                for record in result
            ]

            return {
                'author': author_name,
                'collaborators': collaborations,
                'total_collaborators': len(collaborations)
            }


# Test
if __name__ == "__main__":
    worker = Neo4jGraphWorker(
        uri=NEO4J_DB_URI,
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD")
    )

    try:
        # Example 1: Similarity search
        print("\n" + "=" * 60)
        print("Example 1: Similarity Search")
        print("=" * 60)
        user_query = "Synthetic humans and cameras in motion"
        similar_papers = worker.similarity_search(user_query, top_k=30)
        for i, paper in enumerate(similar_papers, 1):
            print(f"\n{i}. {paper['name']}")
            print(f"   Topic: {paper['topic']}")
            print(f"   Presentation: {paper['presentation_type']}")

        # Example 2: Neighborhood search
        if similar_papers:
            print("\n" + "=" * 60)
            print("Example 2: Neighborhood Search")
            print("=" * 60)
            paper_id = similar_papers[0]['id']
            neighbors = worker.neighborhood_search(paper_id, min_similarity=0.75)
            print(f"\nNeighbors of: {similar_papers[0]['name']}")
            for rel_type, neighbor_list in neighbors.items():
                print(f"\n{rel_type.upper()} RELATIONSHIPS:")
                for neighbor in neighbor_list:
                    print(f"  - {neighbor['name']}")

        # Example 3: Graph traversal
        print("\n" + "=" * 60)
        print("Example 3: Graph Traversal (2 hops)")
        print("=" * 60)
        if similar_papers:
            paper_id = similar_papers[0]['id']
            related = worker.graph_traversal(paper_id, n_hops=2)
            print(f"\nFound {len(related)} related papers through traversal")
            for paper in related[:5]:
                print(f"  - {paper['name']} (distance: {paper['distance']})")

    finally:
        worker.close()
