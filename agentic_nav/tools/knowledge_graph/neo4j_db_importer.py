"""
Neo4j importer for PaperKnowledgeGraph - OPTIMIZED VERSION
Imports NetworkX graph to Neo4j database with maximum performance
"""
import logging
import os

import click
import networkx as nx
from neo4j import GraphDatabase
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
from pathlib import Path

from agentic_nav.tools.knowledge_graph.file_handler import load_graph
from agentic_nav.utils.logger import setup_logging


# Setup logging
setup_logging(
    log_dir="logs",
    level=os.environ.get("AGENTIC_NAV_LOG_LEVEL", "INFO")
)
LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DB_URI = os.environ.get("NEO4J_DB_URI", "bolt://neo4j_db:7687")


class Neo4jImporter:
    """Import of PaperKnowledgeGraph to Neo4j database."""

    def __init__(
            self,
            uri: str = NEO4J_DB_URI,
            username: str = NEO4J_USERNAME,
            password: str = NEO4J_PASSWORD,
            connection_timeout: int = 30
    ):
        """
        Initialize Neo4j connection with optimized settings.

        Args:
            connection_timeout: Connection timeout in seconds (default: 30)
        """
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=connection_timeout,
            max_transaction_retry_time=30
        )
        self.driver.verify_connectivity()
        LOGGER.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def _iter_edges(self, kg: nx.Graph):
        """
        Iterate over edges, handling both Graph and MultiGraph.
        Yields (source, target, data) tuples.
        """
        if isinstance(kg, nx.MultiGraph) or isinstance(kg, nx.MultiDiGraph):
            for source, target, key, data in kg.edges(data=True, keys=True):
                yield source, target, data
        else:
            for source, target, data in kg.edges(data=True):
                yield source, target, data

    def clear_database(self, batch_size=100):
        """Clear database with optimized batch size."""
        with self.driver.session() as session:
            deleted_total = 0
            while True:
                result = session.run("""
                    CALL () {
                        MATCH (n)
                        WITH n LIMIT $batch_size
                        DETACH DELETE n
                        RETURN count(n) as deleted
                    }
                    RETURN deleted
                    """, batch_size=batch_size
                )

                deleted = result.single()["deleted"]
                deleted_total += deleted
                LOGGER.info(f"Deleted {deleted} nodes (total: {deleted_total})")

                if deleted == 0:
                    break

    def create_indexes(self, embedding_dimension: int = 768):
        """Create indexes for better query performance."""
        with self.driver.session() as session:
            indexes = [
                "CREATE INDEX paper_node_id IF NOT EXISTS FOR (p:Paper) ON (p.node_id)",
                "CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)",
                "CREATE INDEX paper_import_id IF NOT EXISTS FOR (p:Paper) ON (p.import_id)",
                "CREATE INDEX paper_presentation_category IF NOT EXISTS FOR (p:Paper) ON (p.presentation_category)",
                "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
                "CREATE INDEX author_id IF NOT EXISTS FOR (a:Author) ON (a.author_id)",
                "CREATE INDEX author_composite_id IF NOT EXISTS FOR (a:Author) ON (a.composite_id)",
                "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.fullname)",
            ]

            for idx in indexes:
                session.run(idx)

            LOGGER.info("Created standard indexes")

            # Vector index (requires Neo4j 5.11+)
            try:
                session.run("""
                    CREATE VECTOR INDEX paper_embeddings IF NOT EXISTS
                    FOR (p:Paper)
                    ON p.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: $dimension,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """, dimension=embedding_dimension)
                LOGGER.info(f"Created vector index for {embedding_dimension}-dimensional embeddings")
            except Exception as e:
                LOGGER.warning(f"Could not create vector index: {e}")

    def _import_paper_nodes(self, kg: nx.Graph, batch_size: int = 100):
        """
        Import paper nodes with optimized batch size.
        Uses UNWIND with CREATE instead of MERGE for better performance.
        """
        paper_nodes = [(n, d) for n, d in kg.nodes(data=True)
                       if d.get('node_type') == 'paper']

        LOGGER.info(f"\nImporting {len(paper_nodes)} paper nodes...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(paper_nodes), batch_size), desc="Paper nodes"):
                batch = paper_nodes[i:i + batch_size]
                papers_data = []

                for node_id, data in batch:
                    embedding = data.get('embedding', [])
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    paper_dict = {
                        "node_id": node_id,
                        "id": data.get('id', ''),
                        "name": data.get('name', ''),
                        "abstract": data.get('abstract', ''),
                        "topic": data.get('topic', ''),
                        "keywords": data.get('keywords', []),
                        "decisions": data.get('decisions', ''),
                        "session": data.get('session', ''),
                        "session_start_time": data.get('session_start_time', ''),
                        "session_end_time": data.get('session_end_time', ''),
                        "presentation_type": data.get('presentation_type', ''),
                        "presentation_category": data.get('presentation_category', ''),
                        "room_name": data.get('room_name', ''),
                        "project_url": data.get('project_url', ''),
                        "poster_position": data.get('poster_position', ''),
                        "paper_url": data.get("paper_url", ""),
                        "sourceid": data.get("sourceid", ""),
                        "virtualsite_url": data.get("virtualsite_url", ""),
                        "import_id": data.get("import_id", ""),
                        'embedding': embedding
                    }
                    papers_data.append(paper_dict)

                # Use CREATE instead of MERGE since we know nodes don't exist
                session.run("""
                    UNWIND $papers AS paper
                    CREATE (p:Paper {
                        node_id: paper.node_id,
                        id: paper.id,
                        name: paper.name,
                        abstract: paper.abstract,
                        topic: paper.topic,
                        keywords: paper.keywords,
                        decisions: paper.decisions,
                        session: paper.session,
                        session_start_time: paper.session_start_time,
                        session_end_time: paper.session_end_time,
                        presentation_type: paper.presentation_type,
                        presentation_category: paper.presentation_category,
                        room_name: paper.room_name,
                        project_url: paper.project_url,
                        poster_position: paper.poster_position,
                        paper_url: paper.paper_url,
                        sourceid: paper.sourceid,
                        virtualsite_url: paper.virtualsite_url,
                        import_id: paper.import_id,
                        embedding: paper.embedding
                    })
                """, papers=papers_data)

        LOGGER.info(f"Imported {len(paper_nodes)} paper nodes")

    def _import_oral_poster_relationships(self, kg: nx.Graph, batch_size: int = 100):
        """Import oral-poster relationships with optimized batch processing."""
        oral_poster_edges = [
            (source, target, data)
            for source, target, data in self._iter_edges(kg)
            if data.get('relationship') == 'oral_poster_pair'
        ]

        LOGGER.info(f"Importing {len(oral_poster_edges)} oral-poster pair relationships...")

        if len(oral_poster_edges) == 0:
            LOGGER.warning("No oral-poster pair relationships found!")
            return

        with self.driver.session() as session:
            for i in tqdm(range(0, len(oral_poster_edges), batch_size), desc="Oral-Poster relationships"):
                batch = oral_poster_edges[i:i + batch_size]

                edges_data = [{
                    'source': source,
                    'target': target,
                    'uid': data.get('uid', '')
                } for source, target, data in batch]

                # Use CREATE instead of MERGE - we know papers exist
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (oral:Paper {node_id: edge.source})
                    MATCH (poster:Paper {node_id: edge.target})
                    CREATE (oral)-[:ORAL_POSTER_PAIR {uid: edge.uid}]->(poster)
                """, edges=edges_data)

        LOGGER.info(f"Imported {len(oral_poster_edges)} oral-poster pair relationships")

    def _import_topic_hierarchy(self, kg: nx.Graph):
        """Import topic hierarchy with optimized queries."""
        topic_paths = set()
        for node_id, data in kg.nodes(data=True):
            if data.get('node_type') == 'paper':
                topic = data.get('topic', '')
                if topic:
                    topic_paths.add(topic)

        LOGGER.info(f"Processing {len(topic_paths)} unique topic paths...")

        all_topics = set()
        topic_relationships = []

        for path in topic_paths:
            parts = [p.strip() for p in path.split('->')]
            for part in parts:
                all_topics.add(part)
            for i in range(len(parts) - 1):
                topic_relationships.append({
                    'parent': parts[i],
                    'child': parts[i + 1]
                })

        LOGGER.info(f"Creating {len(all_topics)} topic nodes...")

        with self.driver.session() as session:
            # Batch create topics
            topics_batch = [{'name': topic} for topic in all_topics]

            # Process in chunks for very large topic sets
            chunk_size = 500
            for i in range(0, len(topics_batch), chunk_size):
                chunk = topics_batch[i:i + chunk_size]
                session.run("""
                    UNWIND $topics AS topic
                    MERGE (t:Topic {name: topic.name})
                """, topics=chunk)

            # Create relationships
            if topic_relationships:
                unique_rels = list({(r['parent'], r['child']): r for r in topic_relationships}.values())

                for i in range(0, len(unique_rels), chunk_size):
                    chunk = unique_rels[i:i + chunk_size]
                    session.run("""
                        UNWIND $rels AS rel
                        MATCH (parent:Topic {name: rel.parent})
                        MATCH (child:Topic {name: rel.child})
                        MERGE (child)-[:SUBTOPIC_OF]->(parent)
                    """, rels=chunk)

        LOGGER.info(f"Imported {len(all_topics)} topic nodes with hierarchy")

    def _connect_papers_to_topics(self, kg: nx.Graph, batch_size: int = 100):
        """Connect papers to topics with larger batches."""
        paper_topic_connections = []

        for node_id, data in kg.nodes(data=True):
            if data.get('node_type') == 'paper':
                topic = data.get('topic', '')
                if topic:
                    parts = [p.strip() for p in topic.split('->')]
                    leaf_topic = parts[-1]
                    paper_topic_connections.append({
                        'paper_node_id': node_id,
                        'topic_name': leaf_topic,
                        'full_path': topic
                    })

        LOGGER.info(f"Connecting {len(paper_topic_connections)} papers to topics...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(paper_topic_connections), batch_size),
                          desc="Paper-Topic connections"):
                batch = paper_topic_connections[i:i + batch_size]

                # Use CREATE for new relationships
                session.run("""
                    UNWIND $connections AS conn
                    MATCH (p:Paper {node_id: conn.paper_node_id})
                    MATCH (t:Topic {name: conn.topic_name})
                    CREATE (p)-[r:BELONGS_TO_TOPIC {full_path: conn.full_path}]->(t)
                """, connections=batch)

        LOGGER.info(f"Connected papers to leaf topics")

    def _import_similarity_relationships_optimized(self, kg: nx.Graph, batch_size: int = 500):
        """
        Import similarity relationships with maximum optimization.
        Uses very large batches and CREATE instead of MERGE.
        """
        similarity_edges = [
            (source, target, data)
            for source, target, data in self._iter_edges(kg)
            if data.get('relationship') == 'similar_to'
        ]

        LOGGER.info(f"Importing {len(similarity_edges)} similarity relationships...")

        if len(similarity_edges) == 0:
            LOGGER.info("No similarity relationships to import")
            return

        with self.driver.session() as session:
            for i in tqdm(range(0, len(similarity_edges), batch_size),
                          desc="Similarity relationships"):
                batch = similarity_edges[i:i + batch_size]

                edges_data = [{
                    'source': source,
                    'target': target,
                    'similarity': float(data.get('similarity', 0.0))
                } for source, target, data in batch]

                # Use CREATE - much faster than MERGE
                # Papers already exist, we just need to connect them
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (p1:Paper {node_id: edge.source})
                    MATCH (p2:Paper {node_id: edge.target})
                    CREATE (p1)-[:SIMILAR_TO {similarity: edge.similarity}]->(p2)
                """, edges=edges_data)

        LOGGER.info(f"Imported {len(similarity_edges)} similarity relationships")

    def _import_authors_and_relationships(self, kg: nx.Graph, batch_size: int = 100):
        """Import authors and relationships with optimized batch sizes."""
        author_nodes = [
            (node_id, data)
            for node_id, data in kg.nodes(data=True)
            if data.get('node_type') == 'author'
        ]

        LOGGER.info(f"Found {len(author_nodes)} author nodes in graph...")

        all_authors = []
        for node_id, data in author_nodes:
            parts = node_id.split(' - ', 1)
            author_id = parts[0].strip() if len(parts) > 0 else ""

            author_dict = {
                'composite_id': node_id,
                'author_id': author_id,
                'fullname': data.get('fullname', ''),
                'institution': data.get('institution', ''),
                'url': data.get('url', '')
            }
            all_authors.append(author_dict)

        LOGGER.info(f"Importing {len(all_authors)} unique authors...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(all_authors), batch_size), desc="Author nodes"):
                batch = all_authors[i:i + batch_size]

                session.run("""
                    UNWIND $authors AS author
                    MERGE (a:Author {composite_id: author.composite_id})
                    ON CREATE SET
                        a.author_id = author.author_id,
                        a.fullname = author.fullname,
                        a.institution = author.institution,
                        a.url = author.url
                    ON MATCH SET
                        a.author_id = author.author_id,
                        a.fullname = author.fullname,
                        a.institution = author.institution,
                        a.url = author.url
                """, authors=batch)

        LOGGER.info(f"Imported {len(all_authors)} author nodes")

        # Import relationships
        author_paper_edges = [
            (source, target, data)
            for source, target, data in self._iter_edges(kg)
            if data.get('relationship') == 'is_author_of'
        ]

        LOGGER.info(f"Found {len(author_paper_edges)} IS_AUTHOR_OF edges in graph")

        if len(author_paper_edges) > 0:
            with self.driver.session() as session:
                for i in tqdm(range(0, len(author_paper_edges), batch_size), desc="Author-Paper relationships"):
                    batch = author_paper_edges[i:i + batch_size]

                    edges_data = [{
                        'author_id': source,
                        'paper_node_id': target,
                        'author_order': data.get('author_order', 0)
                    } for source, target, data in batch]

                    # Use CREATE instead of MERGE
                    session.run("""
                        UNWIND $edges AS edge
                        MATCH (a:Author {composite_id: edge.author_id})
                        MATCH (p:Paper {node_id: edge.paper_node_id})
                        CREATE (a)-[r:IS_AUTHOR_OF {author_order: edge.author_order}]->(p)
                    """, edges=edges_data)

            LOGGER.info(f"Created {len(author_paper_edges)} author-paper relationships")

    def import_graph(
            self,
            kg_path: str,
            batch_size: int = 100,
            similarity_batch_size: int = 500,
            embedding_dimension: int = 768
    ):
        """
        Import the entire knowledge graph to Neo4j with optimized settings.

        Args:
            kg_path: Path to graph pickle file
            batch_size: Batch size for most operations (default: 1000)
            similarity_batch_size: Larger batch for similarity edges (default: 10000)
            embedding_dimension: Embedding vector dimension
        """
        LOGGER.info(f"Loading graph from path {kg_path}")
        kg = load_graph(kg_path)

        LOGGER.info("Starting optimized Neo4j import...")

        # Clear and prepare database
        self.clear_database()
        self.create_indexes(embedding_dimension)

        # Import in optimized order
        self._import_paper_nodes(kg, batch_size)
        self._import_authors_and_relationships(kg, batch_size)
        self._import_topic_hierarchy(kg)
        self._connect_papers_to_topics(kg, batch_size * 2)  # 2x batch for simpler queries
        self._import_oral_poster_relationships(kg, batch_size)
        self._import_similarity_relationships_optimized(kg, similarity_batch_size)

        LOGGER.info("Import completed successfully!")

    def verify_import(self) -> Dict[str, Any]:
        """Verify the import by checking node and relationship counts."""
        with self.driver.session() as session:
            result = session.run("MATCH (p:Paper) RETURN count(p) as count")
            paper_count = result.single()['count']

            result = session.run("MATCH (p:Paper) WHERE p.presentation_category = 'oral' RETURN count(p) as count")
            oral_count = result.single()['count']

            result = session.run("MATCH (p:Paper) WHERE p.presentation_category = 'poster' RETURN count(p) as count")
            poster_count = result.single()['count']

            result = session.run("MATCH (t:Topic) RETURN count(t) as count")
            topic_count = result.single()['count']

            result = session.run("MATCH (a:Author) RETURN count(a) as count")
            author_count = result.single()['count']

            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']

            result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count")
            similarity_count = result.single()['count']

            result = session.run("MATCH ()-[r:SUBTOPIC_OF]->() RETURN count(r) as count")
            subtopic_count = result.single()['count']

            result = session.run("MATCH ()-[r:IS_AUTHOR_OF]->() RETURN count(r) as count")
            is_author_of_count = result.single()['count']

            result = session.run("MATCH ()-[r:ORAL_POSTER_PAIR]->() RETURN count(r) as count")
            oral_poster_count = result.single()['count']

            stats = {
                'papers': paper_count,
                'orals': oral_count,
                'posters': poster_count,
                'topics': topic_count,
                'authors': author_count,
                'total_relationships': rel_count,
                'similarity_relationships': similarity_count,
                'subtopic_relationships': subtopic_count,
                'is_author_of_relationships': is_author_of_count,
                'oral_poster_pair_relationships': oral_poster_count
            }

            LOGGER.info("Neo4j Database Statistics:")
            for key, value in stats.items():
                LOGGER.info(f"   {key}: {value}")

            return stats


@click.command()
@click.option("-g", "--graph-path", help="Path to the knowledge graph file (pickle).", default=f"{PROJECT_ROOT}/graphs/knowledge_graph.pkl")
@click.option("-l", "--neo4j-uri", help="Database URI", default="bolt://localhost:7687")
@click.option("-u", "--neo4j-username", help="Database user", default=NEO4J_USERNAME)
@click.option("-p", "--neo4j-password", help="Database password", default=NEO4J_PASSWORD)
@click.option("-b", "--batch-size", help="Batch size for node insertion", default=100)
@click.option("-s", "--similarity-batch-size", help="Batch size for similarity edges", default=5000)
@click.option("-e", "--embedding-dimension", help="Vector embedding dimensions", default=768)
def main(
    graph_path: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    batch_size: int = 100,
    similarity_batch_size: int = 5000,
    embedding_dimension: int = 768
):
    importer = Neo4jImporter(
        neo4j_uri,
        neo4j_username,
        neo4j_password
    )
    try:
        importer.import_graph(
            graph_path,
            batch_size,
            similarity_batch_size,
            embedding_dimension
        )
        importer.verify_import()
    finally:
        importer.close()


if __name__ == "__main__":
    main()
