import numpy as np

from neo4j import GraphDatabase
from pathlib import Path

from utils.embedding_generator import batch_embed_documents

from typing import List, Dict, Any, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.parent.parent


class Neo4jGraphWorker:
    """Search and traversal operations for Neo4j paper knowledge graph."""

    _DB_SIMILARITY_SEARCH_QUERY = """
        CALL db.index.vector.queryNodes('paper_embeddings', $top_k, $query_embedding)
        YIELD node, score
        RETURN node.id as id,
               node.name as name,
               node.abstract as abstract,
               node.topic as topic,
               score
        ORDER BY score DESC
        """

    _DB_NEIGHBORHOOD_SEARCH_QUERY = staticmethod(lambda rel_filter: f"""
        MATCH (p:Paper)
        WHERE p.id IN $paper_ids
        MATCH (p)-[r{rel_filter}]-(neighbor)
        RETURN p.id as source_paper_id,
               neighbor,
               type(r) as relationship_type,
               properties(r) as relationship_properties,
               labels(neighbor) as neighbor_labels
        """)

    _DB_GRAPH_TRAVERSAL_QUERY = staticmethod(lambda rel_filter, n_hops: f"""
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
        """)

    _DB_PAPERS_BY_AUTHOR = """
        MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
        WHERE a.fullname = $author_name
        RETURN p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic,
               a.fullname as author_name
        ORDER BY p.name
        """

    _DB_PAPERS_BY_AUTHOR_FUZZY = """
        MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
        WHERE toLower(a.fullname) CONTAINS toLower($author_name)
        RETURN p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic,
               a.fullname as author_name
        ORDER BY p.name
        """

    _DB_PAPERS_BY_TOPIC = """
        MATCH (p:Paper)-[:BELONGS_TO_TOPIC]->(t:Topic {name: $topic_name})
        RETURN p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic
        ORDER BY p.name
        """

    _DB_PAPERS_BY_TOPIC_AND_SUBTOPIC = """
        MATCH (t:Topic {name: $topic_name})
        OPTIONAL MATCH (subtopic:Topic)-[:SUBTOPIC_OF*]->(t)
        WITH t, collect(DISTINCT subtopic) + t as all_topics
        UNWIND all_topics as topic
        MATCH (p:Paper)-[:BELONGS_TO_TOPIC]->(topic)
        RETURN DISTINCT p.id as id,
               p.name as name,
               p.abstract as abstract,
               p.topic as topic
        ORDER BY p.name
        """

    _DB_SIMILAR_PAPER_SEARCH = """
        MATCH (p1:Paper {id: $paper_id})-[r:SIMILAR_TO]-(p2:Paper)
        WHERE r.similarity >= $min_similarity
        RETURN p2.id as id,
               p2.name as name,
               p2.abstract as abstract,
               p2.topic as topic,
               r.similarity as similarity
        ORDER BY r.similarity DESC
        LIMIT $top_k
        """

    def __init__(
            self,
            uri: str = "bolt://localhost:7687",
            username: str = "neo4j",
            password: str = "password"
    ):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    @staticmethod
    def embed_user_query(
        text: str,
        embedding_model: str = "ollama/nomic-embed-text",
        api_base: str = "http://localhost:11434"
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
        top_k: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on paper embeddings.

        Args:
            query_embedding: Query embedding vector (numpy array or list)
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

        with self.driver.session() as session:
            result = session.run(self._DB_SIMILARITY_SEARCH_QUERY, query_embedding=query_embedding, top_k=top_k)

            papers = []
            for record in result:
                paper = {
                    'id': record['id'],
                    'name': record['name'],
                    'abstract': record['abstract'],
                    'topic': record['topic'],
                    'similarity_score': record['score']
                }

                # Apply minimum similarity filter if specified
                if min_similarity is None or paper['similarity_score'] >= min_similarity:
                    papers.append(paper)

            return papers

    def neighborhood_search(
            self,
            paper_id: str,
            relationship_types: Optional[List[str]] = None,
            include_properties: bool = True
    ) -> Dict[str, Any]:
        """
        Find immediate neighbors of given paper nodes.

        Args:
            paper_id: Paper ID to find neighbors for
            relationship_types: Optional list of relationship types to filter
                               (e.g., ['SIMILAR_TO', 'AUTHORED_BY', 'BELONGS_TO_TOPIC'])
            include_properties: Whether to include relationship properties

        Returns:
            Dictionary with neighbors grouped by relationship type
        """
        with self.driver.session() as session:
            # Build relationship type filter
            if relationship_types:
                rel_filter = f":{':'.join(relationship_types)}"
            else:
                rel_filter = ""

            query = self._DB_NEIGHBORHOOD_SEARCH_QUERY(rel_filter)

            result = session.run(query, paper_ids=[paper_id])

            # Organize results by relationship type
            neighbors = {
                'similar_papers': [],
                'authors': [],
                'topics': [],
                'raw_results': []
            }

            for record in result:

                neighbor_node = dict(record['neighbor'])
                rel_type = record['relationship_type']
                rel_props = record['relationship_properties']
                labels = record['neighbor_labels']

                if "embedding" in neighbor_node.keys():
                    del neighbor_node["embedding"]

                neighbor_info = {
                    'source_paper_id': record['source_paper_id'],
                    'relationship_type': rel_type,
                    'neighbor': neighbor_node,
                    'labels': labels
                }

                if include_properties and rel_props:
                    neighbor_info['relationship_properties'] = rel_props

                # Categorize by type
                if 'Paper' in labels:
                    neighbors['similar_papers'].append(neighbor_info)
                elif 'Author' in labels:
                    neighbors['authors'].append(neighbor_info)
                elif 'Topic' in labels:
                    neighbors['topics'].append(neighbor_info)

                neighbors['raw_results'].append(neighbor_info)

            return neighbors

    def graph_traversal(
            self,
            start_paper_ids: List[str],
            n_hops: int = 2,
            relationship_types: Optional[List[str]] = None,
            max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Traverse the graph for n hops from starting paper nodes.

        Args:
            start_paper_ids: List of paper IDs to start traversal from
            n_hops: Number of hops to traverse (1-5 recommended)
            relationship_types: Optional list of relationship types to traverse
            max_results: Optional maximum number of results to return

        Returns:
            List of papers found through traversal with distance information
        """
        with self.driver.session() as session:
            # Build relationship type filter
            if relationship_types:
                rel_filter = f":{':'.join(relationship_types)}"
            else:
                rel_filter = ""

            query = self._DB_GRAPH_TRAVERSAL_QUERY(rel_filter=rel_filter, n_hops=n_hops)

            if max_results:
                query += f" LIMIT {max_results}"

            result = session.run(query, start_paper_ids=start_paper_ids)

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

    def combined_search_workflow(
            self,
            user_query: str,
            top_k: int = 5,
            n_hops: int = 2,
            relationship_types: Optional[List[str]] = None,
            include_neighborhood: bool = True
    ) -> Dict[str, Any]:
        """
        Combined workflow: similarity search -> neighborhood search -> graph traversal.

        Args:
            user_query: Query embedding vector
            top_k: Number of similar papers to find initially
            n_hops: Number of hops for graph traversal
            relationship_types: Optional relationship types to consider
            include_neighborhood: Whether to include immediate neighbors

        Returns:
            Dictionary containing all search results
        """
        # Step 1: Similarity search
        print(f"🔍 Finding {top_k} most similar papers...")
        similar_papers = self.similarity_search(user_query, top_k)

        if not similar_papers:
            return {
                'similar_papers': [],
                'neighborhood': {},
                'related_papers': [],
                'summary': {
                    'similar_count': 0,
                    'neighborhood_count': 0,
                    'related_count': 0
                }
            }

        paper_ids = [p['id'] for p in similar_papers]

        # Step 2: Neighborhood search (optional)
        neighborhood = {}
        if include_neighborhood:
            print(f"🔍 Finding immediate neighbors...")
            neighborhood = self.neighborhood_search(paper_ids, relationship_types)

        # Step 3: Graph traversal
        print(f"🔍 Traversing graph with {n_hops} hops...")
        related_papers = self.graph_traversal(
            paper_ids,
            n_hops,
            relationship_types
        )

        # Create summary
        summary = {
            'similar_count': len(similar_papers),
            'neighborhood_count': len(neighborhood.get('raw_results', [])),
            'related_count': len(related_papers),
            'unique_papers_found': len(set([p['id'] for p in similar_papers + related_papers]))
        }

        print(f"\n✅ Search complete: {summary}")

        return {
            'similar_papers': similar_papers,
            'neighborhood': neighborhood,
            'related_papers': related_papers,
            'summary': summary
        }

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
                    'author_name': record['author_name']
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

            result = session.run(query, topic_name=topic_name)

            papers = []
            for record in result:
                paper = {
                    'id': record['id'],
                    'name': record['name'],
                    'abstract': record['abstract'],
                    'topic': record['topic']
                }
                papers.append(paper)

            return papers

    def find_similar_papers_direct(
            self,
            paper_id: str,
            min_similarity: float = 0.7,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find papers directly connected via SIMILAR_TO relationship.

        Args:
            paper_id: Source paper ID
            min_similarity: Minimum similarity threshold
            top_k: Maximum number of results

        Returns:
            List of similar papers with similarity scores
        """
        with self.driver.session() as session:
            query = self._DB_SIMILAR_PAPER_SEARCH

            result = session.run(
                query,
                paper_id=paper_id,
                min_similarity=min_similarity,
                top_k=top_k
            )

            papers = []
            for record in result:
                paper = {
                    'id': record['id'],
                    'name': record['name'],
                    'abstract': record['abstract'],
                    'topic': record['topic'],
                    'similarity': record['similarity']
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
        uri="bolt://localhost:7687",
        username="neo4j",
        password="llm_agents"
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
            print(f"   Similarity: {paper['similarity_score']:.4f}")

        # Example 2: Neighborhood search
        if similar_papers:
            print("\n" + "=" * 60)
            print("Example 2: Neighborhood Search")
            print("=" * 60)
            paper_id = similar_papers[0]['id']
            neighbors = searcher.neighborhood_search([paper_id])
            print(f"\nNeighbors of: {similar_papers[0]['name']}")
            print(f"  - Similar papers: {len(neighbors['similar_papers'])}")
            print(f"  - Authors: {len(neighbors['authors'])}")
            print(f"  - Topics: {len(neighbors['topics'])}")

        # Example 3: Graph traversal
        print("\n" + "=" * 60)
        print("Example 3: Graph Traversal (2 hops)")
        print("=" * 60)
        if similar_papers:
            paper_ids = [p['id'] for p in similar_papers[:2]]
            related = searcher.graph_traversal(paper_ids, n_hops=2)
            print(f"\nFound {len(related)} related papers through traversal")
            for paper in related[:5]:  # Show first 5
                print(f"  - {paper['name']} (distance: {paper['distance']})")

        # Example 4: Combined workflow
        print("\n" + "=" * 60)
        print("Example 4: Combined Search Workflow")
        print("=" * 60)
        results = searcher.combined_search_workflow(
            user_query,
            top_k=3,
            n_hops=2
        )
        print(f"\nWorkflow Results:")
        print(f"  - Initial similar papers: {results['summary']['similar_count']}")
        print(f"  - Neighborhood connections: {results['summary']['neighborhood_count']}")
        print(f"  - Related papers (traversal): {results['summary']['related_count']}")
        print(f"  - Total unique papers: {results['summary']['unique_papers_found']}")

    finally:
        searcher.close()
