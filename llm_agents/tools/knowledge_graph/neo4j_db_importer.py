"""
Neo4j exporter for PaperKnowledgeGraph
Exports NetworkX graph to Neo4j database with proper handling of embeddings and relationships
"""
import logging
import os

import click
import networkx as nx
from neo4j import GraphDatabase
from typing import Dict, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path

from llm_agents.tools.knowledge_graph.file_handler import load_graph
from llm_agents.utils.logging import setup_logging


# Setup logging
setup_logging(
    log_dir="logs",
    level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "INFO")
)
LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DB_URI = os.environ.get("NEO4J_DB_URI", "bolt://neo4j_db:7687")


class Neo4jImporter:
    """Import PaperKnowledgeGraph to Neo4j database."""

    def __init__(
            self,
            uri: str = NEO4J_DB_URI,
            username: str = NEO4J_USERNAME,
            password: str = NEO4J_PASSWORD
    ):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        LOGGER.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            LOGGER.info("Cleared existing database")

    def create_indexes(self, embedding_dimension: int = 768):
        """Create indexes for better query performance, including vector index."""
        with self.driver.session() as session:
            # Create index on paper IDs
            session.run("CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)")

            # Create index on topic names
            session.run("CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)")

            # Create index on author IDs
            session.run("CREATE INDEX author_id IF NOT EXISTS FOR (a:Author) ON (a.author_id)")

            # Create index on author names (useful for searching)
            session.run("CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.fullname)")

            # Create vector index for embeddings (Neo4j 5.11+)
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
                LOGGER.warning(f"Warning: Could not create vector index: {e}")
                LOGGER.warning("Vector indexes require Neo4j 5.11+ or Enterprise Edition")

            LOGGER.info("Created standard indexes")

    def _export_paper_nodes(self, kg: nx.Graph, batch_size: int):
        """Export paper nodes to Neo4j with all attributes."""
        paper_nodes = [(n, d) for n, d in kg.nodes(data=True)
                       if d.get('node_type') == 'paper']

        LOGGER.info(f"\nExporting {len(paper_nodes)} paper nodes...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(paper_nodes), batch_size), desc="Paper nodes"):
                batch = paper_nodes[i:i + batch_size]
                papers_data = []

                for node_id, data in batch:
                    # Convert embedding to list if it's numpy array
                    embedding = data.get('embedding', [])
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    paper_dict = {
                        'id': node_id,
                        'name': data.get('name', ''),
                        'abstract': data.get('abstract', ''),
                        'topic': data.get('topic', ''),
                        'keywords': data.get('keywords', []),
                        'decision': data.get('decision', ''),
                        'session': data.get('session', ''),
                        'session_start_time': data.get('session_start_time', ''),
                        'session_end_time': data.get('session_end_time', ''),
                        'presentation_type': data.get('presentation_type', ''),
                        'room_name': data.get('room_name', ''),
                        'project_url': data.get('project_url', ''),
                        'poster_position': data.get('poster_position', ''),
                        'embedding': embedding
                    }
                    papers_data.append(paper_dict)

                # Batch create paper nodes
                session.run("""
                    UNWIND $papers AS paper
                    CREATE (p:Paper {
                        id: paper.id,
                        name: paper.name,
                        abstract: paper.abstract,
                        topic: paper.topic,
                        keywords: paper.keywords,
                        decision: paper.decision,
                        session: paper.session,
                        session_start_time: paper.session_start_time,
                        session_end_time: paper.session_end_time,
                        presentation_type: paper.presentation_type,
                        room_name: paper.room_name,
                        project_url: paper.project_url,
                        poster_position: paper.poster_position,
                        embedding: paper.embedding
                    })
                """, papers=papers_data)

        LOGGER.info(f"Exported {len(paper_nodes)} paper nodes")

    def _export_topic_hierarchy(self, kg: nx.Graph):
        """
        Export topic nodes with hierarchical structure to Neo4j.
        Splits topics like "Deep Learning->Theory" into separate nodes with parent-child relationships.
        """
        # Collect all unique topic paths from paper nodes
        topic_paths = set()
        for node_id, data in kg.nodes(data=True):
            if data.get('node_type') == 'paper':
                topic = data.get('topic', '')
                if topic:
                    topic_paths.add(topic)

        LOGGER.info(f"Processing {len(topic_paths)} unique topic paths...")

        # Parse topic paths and create hierarchy
        all_topics = set()
        topic_relationships = []

        for path in topic_paths:
            parts = [p.strip() for p in path.split('->')]

            # Add all topic parts
            for part in parts:
                all_topics.add(part)

            # Create parent-child relationships
            for i in range(len(parts) - 1):
                topic_relationships.append({
                    'parent': parts[i],
                    'child': parts[i + 1]
                })

        LOGGER.info(
            f"Creating {len(all_topics)} topic nodes with {len(set(tuple(r.items()) for r in topic_relationships))} "
            f"hierarchical relationships..."
        )

        with self.driver.session() as session:
            # Create all topic nodes (using MERGE to avoid duplicates)
            topics_data = [{'name': topic} for topic in all_topics]
            session.run("""
                UNWIND $topics AS topic
                MERGE (t:Topic {name: topic.name})
            """, topics=topics_data)

            # Create hierarchical relationships between topics (deduplicate first)
            if topic_relationships:
                # Remove duplicates
                unique_rels = list({(r['parent'], r['child']): r for r in topic_relationships}.values())
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (parent:Topic {name: rel.parent})
                    MATCH (child:Topic {name: rel.child})
                    MERGE (child)-[:SUBTOPIC_OF]->(parent)
                """, rels=unique_rels)

        LOGGER.info(f"Exported {len(all_topics)} topic nodes with hierarchy")

    def _connect_papers_to_topics(self, kg: nx.Graph, batch_size: int):
        """
        Connect papers to their leaf topic nodes.
        For "Deep Learning->Theory", connects paper to "Theory" node.
        """
        paper_topic_connections = []

        for node_id, data in kg.nodes(data=True):
            if data.get('node_type') == 'paper':
                topic = data.get('topic', '')
                if topic:
                    # Get the leaf topic (last part after splitting)
                    parts = [p.strip() for p in topic.split('->')]
                    leaf_topic = parts[-1]

                    paper_topic_connections.append({
                        'paper_id': node_id,
                        'topic_name': leaf_topic,
                        'full_path': topic  # Store full path as property
                    })

        LOGGER.info(f"Connecting {len(paper_topic_connections)} papers to topics...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(paper_topic_connections), batch_size),
                          desc="Paper-Topic connections"):
                batch = paper_topic_connections[i:i + batch_size]

                session.run("""
                    UNWIND $connections AS conn
                    MATCH (p:Paper {id: conn.paper_id})
                    MATCH (t:Topic {name: conn.topic_name})
                    MERGE (p)-[r:BELONGS_TO_TOPIC]->(t)
                    SET r.full_path = conn.full_path
                """, connections=batch)

        LOGGER.info(f"Connected papers to leaf topics")

    def _export_similarity_relationships(self, kg: nx.Graph, batch_size: int):
        """Export similarity relationships between papers to Neo4j."""
        # Filter only similarity edges
        similarity_edges = [
            (source, target, data)
            for source, target, data in kg.edges(data=True)
            if data.get('relationship') == 'similar_to'
        ]

        LOGGER.info(f"Exporting {len(similarity_edges)} similarity relationships...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(similarity_edges), batch_size),
                          desc="Similarity relationships"):
                batch = similarity_edges[i:i + batch_size]

                edges_data = [{
                    'source': source,
                    'target': target,
                    'similarity': data.get('similarity', 0.0)
                } for source, target, data in batch]

                session.run("""
                    UNWIND $edges AS edge
                    MATCH (p1:Paper {id: edge.source})
                    MATCH (p2:Paper {id: edge.target})
                    MERGE (p1)-[:SIMILAR_TO {similarity: edge.similarity}]->(p2)
                """, edges=edges_data)

        LOGGER.info(f"Exported {len(similarity_edges)} similarity relationships")

    def _export_authors_and_relationships(self, kg: nx.Graph, batch_size: int):
        """
        Export author nodes from NetworkX graph (where they already exist as separate nodes)
        and create IS_AUTHOR_OF relationships between authors and papers.

        Author nodes in NetworkX have composite IDs like "12345 - John Doe"
        """
        # Collect author nodes from the graph
        author_nodes = [
            (node_id, data)
            for node_id, data in kg.nodes(data=True)
            if data.get('node_type') != 'paper' and data.get('node_type') != 'topic'
        ]

        LOGGER.info(f"Found {len(author_nodes)} author nodes in graph...")

        # Extract author data
        all_authors = []
        for node_id, data in author_nodes:
            # Parse composite ID "12345 - John Doe"
            parts = node_id.split(' - ', 1)
            author_id = parts[0].strip() if len(parts) > 0 else ""

            author_dict = {
                'composite_id': node_id,  # Store the full composite ID
                'author_id': author_id,
                'fullname': data.get('fullname', ''),
                'institution': data.get('institution', ''),
                'url': data.get('url', '')
            }
            all_authors.append(author_dict)

        LOGGER.info(f"Exporting {len(all_authors)} unique authors...")

        with self.driver.session() as session:
            # Create author nodes in batches
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

        LOGGER.info(f"Exported {len(all_authors)} author nodes")

        # Method 1: Try to collect author-paper relationships from graph edges
        author_paper_edges = [
            (source, target, data)
            for source, target, data in kg.edges(data=True)
            if data.get('relationship') == 'is_author_of'
        ]

        LOGGER.info(f"Found {len(author_paper_edges)} IS_AUTHOR_OF edges in graph")

        # Method 2: If no edges found, extract from paper node 'authors' attribute
        if len(author_paper_edges) == 0:
            LOGGER.warning("No IS_AUTHOR_OF edges found in graph. Extracting from paper 'authors' attribute...")

            paper_author_relationships = []
            for node_id, data in kg.nodes(data=True):
                if data.get('node_type') == 'paper':
                    authors = data.get('authors', [])

                    if authors and isinstance(authors, list) and len(authors) > 0:
                        # Check if authors are stored as dicts
                        if isinstance(authors[0], dict):
                            for author in authors:
                                author_id = str(author.get('id', ''))
                                fullname = author.get('fullname', '')
                                if author_id and fullname:
                                    composite_id = f"{author_id} - {fullname}"
                                    paper_author_relationships.append({
                                        'author_id': composite_id,
                                        'paper_id': node_id
                                    })

            LOGGER.info(f"Extracted {len(paper_author_relationships)} relationships from paper attributes")

            # Create relationships from extracted data
            with self.driver.session() as session:
                for i in tqdm(range(0, len(paper_author_relationships), batch_size),
                              desc="Author-Paper relationships"):
                    batch = paper_author_relationships[i:i + batch_size]

                    session.run("""
                        UNWIND $edges AS edge
                        MATCH (a:Author {composite_id: edge.author_id})
                        MATCH (p:Paper {id: edge.paper_id})
                        MERGE (a)-[:IS_AUTHOR_OF]->(p)
                    """, edges=batch)

            LOGGER.info(f"Created {len(paper_author_relationships)} author-paper relationships")
        else:
            # Create relationships from graph edges
            with self.driver.session() as session:
                for i in tqdm(range(0, len(author_paper_edges), batch_size),
                              desc="Author-Paper relationships"):
                    batch = author_paper_edges[i:i + batch_size]

                    edges_data = [{
                        'author_id': source,  # composite ID like "12345 - John Doe"
                        'paper_id': target
                    } for source, target, data in batch]

                    session.run("""
                        UNWIND $edges AS edge
                        MATCH (a:Author {composite_id: edge.author_id})
                        MATCH (p:Paper {id: edge.paper_id})
                        MERGE (a)-[:IS_AUTHOR_OF]->(p)
                    """, edges=edges_data)

            LOGGER.info(f"Created {len(author_paper_edges)} author-paper relationships")

    def export_graph(self, kg_path: str, batch_size: int = 100, embedding_dimension: int = 768):
        """Export the entire knowledge graph to Neo4j."""
        LOGGER.info(f"Loading graph from path {kg_path}")
        kg = load_graph(kg_path)

        LOGGER.info("Starting Neo4j export...")

        # Clear and prepare database
        self.clear_database()
        self.create_indexes(embedding_dimension)

        # Export paper nodes
        self._export_paper_nodes(kg, batch_size)

        # Export authors and author-paper relationships
        self._export_authors_and_relationships(kg, batch_size)

        # Export topic hierarchy
        self._export_topic_hierarchy(kg)

        # Connect papers to topics
        self._connect_papers_to_topics(kg, batch_size)

        # Export similarity relationships
        self._export_similarity_relationships(kg, batch_size)

        LOGGER.info("Export completed successfully!")

    def verify_export(self) -> Dict[str, Any]:
        """Verify the export by checking node and relationship counts."""
        with self.driver.session() as session:
            # Count papers
            result = session.run("MATCH (p:Paper) RETURN count(p) as count")
            paper_count = result.single()['count']

            # Count topics
            result = session.run("MATCH (t:Topic) RETURN count(t) as count")
            topic_count = result.single()['count']

            # Count authors
            result = session.run("MATCH (a:Author) RETURN count(a) as count")
            author_count = result.single()['count']

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']

            # Count similarity relationships
            result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count")
            similarity_count = result.single()['count']

            # Count topic hierarchy relationships
            result = session.run("MATCH ()-[r:SUBTOPIC_OF]->() RETURN count(r) as count")
            subtopic_count = result.single()['count']

            # Count author relationships (updated relationship name)
            result = session.run("MATCH ()-[r:IS_AUTHOR_OF]->() RETURN count(r) as count")
            is_author_of_count = result.single()['count']

            stats = {
                'papers': paper_count,
                'topics': topic_count,
                'authors': author_count,
                'total_relationships': rel_count,
                'similarity_relationships': similarity_count,
                'subtopic_relationships': subtopic_count,
                'is_author_of_relationships': is_author_of_count
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
@click.option("-e", "--embedding-dimension", help="Vector embedding dimensions", default=768)
def main(
    graph_path: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    batch_size: int = 100,
    embedding_dimension: int = 768
):
    """
    Convenience function to export a knowledge graph to Neo4j.

    Args:
        graph_path: PaperKnowledgeGraph instance
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        batch_size: Batch size for processing
        embedding_dimension: Dimension of embedding vectors (default: 768)
    """
    importer = Neo4jImporter(neo4j_uri, neo4j_username, neo4j_password)
    try:
        importer.export_graph(
            graph_path,
            batch_size,
            embedding_dimension
        )
        importer.verify_export()
    finally:
        importer.close()


if __name__ == "__main__":
    main()
