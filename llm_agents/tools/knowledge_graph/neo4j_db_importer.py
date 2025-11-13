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

try:
    from .file_handler import load_graph
except ImportError:
    from file_handler import load_graph

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")


class Neo4jImporter:
    """Import PaperKnowledgeGraph to Neo4j database."""

    def __init__(
            self,
            uri: str = "bolt://localhost:7687",
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
            session.run("CREATE INDEX author_id IF NOT EXISTS FOR (a:Author) ON (a.id)")

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
        """Export paper nodes to Neo4j."""
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
        Export author nodes and create AUTHORED_BY relationships between papers and authors.
        """
        # Collect all unique authors and paper-author relationships
        all_authors = {}  # Use dict to deduplicate by author ID
        paper_author_relationships = []

        for node_id, data in kg.nodes(data=True):
            if data.get('node_type') == 'paper':
                authors = data.get('authors', [])

                if authors and isinstance(authors[0], dict):
                    for author in authors:
                        author_id = str(author.get('id', ''))
                        if author_id and author_id != '':
                            # Store author info (will deduplicate automatically)
                            all_authors[author_id] = {
                                'id': author_id,
                                'fullname': author.get('fullname', ''),
                                'institution': author.get('institution', ''),
                                'url': author.get('url', '')
                            }

                            # Store paper-author relationship
                            paper_author_relationships.append({
                                'paper_id': node_id,
                                'author_id': author_id
                            })

        LOGGER.info(f"Exporting {len(all_authors)} unique authors...")

        with self.driver.session() as session:
            # Create author nodes in batches
            authors_list = list(all_authors.values())
            for i in tqdm(range(0, len(authors_list), batch_size), desc="Author nodes"):
                batch = authors_list[i:i + batch_size]

                session.run("""
                    UNWIND $authors AS author
                    MERGE (a:Author {id: author.id})
                    ON CREATE SET
                        a.fullname = author.fullname,
                        a.institution = author.institution,
                        a.url = author.url
                    ON MATCH SET
                        a.fullname = author.fullname,
                        a.institution = author.institution,
                        a.url = author.url
                """, authors=batch)

        LOGGER.info(f"Exported {len(all_authors)} author nodes")

        # Create paper-author relationships in batches
        LOGGER.info(f"Creating {len(paper_author_relationships)} paper-author relationships...")

        with self.driver.session() as session:
            for i in tqdm(range(0, len(paper_author_relationships), batch_size),
                          desc="Paper-Author relationships"):
                batch = paper_author_relationships[i:i + batch_size]

                session.run("""
                    UNWIND $rels AS rel
                    MATCH (p:Paper {id: rel.paper_id})
                    MATCH (a:Author {id: rel.author_id})
                    MERGE (p)-[:AUTHORED_BY]->(a)
                """, rels=batch)

        LOGGER.info(f"Created {len(paper_author_relationships)} paper-author relationships")

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

        # Export authors and paper-author relationships
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

            # Count author relationships
            result = session.run("MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) as count")
            authored_by_count = result.single()['count']

            stats = {
                'papers': paper_count,
                'topics': topic_count,
                'authors': author_count,
                'total_relationships': rel_count,
                'similarity_relationships': similarity_count,
                'subtopic_relationships': subtopic_count,
                'authored_by_relationships': authored_by_count
            }

            LOGGER.info("Neo4j Database Statistics:")
            for key, value in stats.items():
                LOGGER.info(f"   {key}: {value}")

            return stats


@click.command()
@click.option("-g", "--graph-path", help="Path to the knowledge graph file (pickle).", default=f"{PROJECT_ROOT}/graphs/knowledge_graph.pkl")
@click.option("-l", "--neo4j-uri", help="Database URI", default="bolt://localhost:7687")
@click.option("-u", "--neo4j-username", help="Database user", default="neo4j")
@click.option("-p", "--neo4j-password", help="Database password")
@click.option("-b", "--batch-size", help="Batch size for node insertion", default=100)
@click.option("-e", "--embedding-dimension", help="Vector embedding dimensions", default=768)
def export_to_neo4j(
    graph_path: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: str = "neo4j",
    neo4j_password: str = "password",
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
    export_to_neo4j()