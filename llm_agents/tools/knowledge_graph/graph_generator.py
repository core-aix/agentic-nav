import json
import logging
import os

import click
import networkx as nx
import numpy as np
import litellm
from typing import List, Dict, Any, Union
from litellm import embedding
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pathlib import Path

from llm_agents.utils.embedding_generator import batch_embed_documents
from llm_agents.utils.logging import setup_logging
from llm_agents.tools.knowledge_graph.file_handler import save_graph


# Setup logging
setup_logging(
    log_dir="logs",
    level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "INFO")
)
LOGGER = logging.getLogger(__name__)
litellm._logging._disable_debugging()
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "ollama/nomic-embed-text")
EMBEDDING_MODEL_API_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE", "http://localhost:11435")


class PaperKnowledgeGraph:
    """
    A knowledge graph builder for academic papers focusing on:
    - Paper names (nodes)
    - Topics (nodes)
    - Abstract embeddings (stored as node attributes)
    Uses litellm with ollama for local embedding generation with parallel processing.
    """
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        ollama_base_url: str = EMBEDDING_MODEL_API_BASE,
        embedding_gen_batch_size: int = 32,
        max_parallel_workers: int = 8,
        limit_num_papers: Union[int, None] = None
    ):
        """
        Initialize the knowledge graph builder.

        Args:
            embedding_model: Name of the ollama embedding model (e.g., 'nomic-embed-text')
            ollama_base_url: Base URL for the ollama server
            embedding_gen_batch_size: Batch size for generating text embeddings
            max_parallel_workers: Number of parallel workers for embedding generation
        """
        self.graph = nx.Graph()
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.batch_size = embedding_gen_batch_size
        self.max_workers = max_parallel_workers
        self.papers_data = []
        self.limit_num_papers = limit_num_papers

        # Test connection
        LOGGER.info(f"Initializing with model: {embedding_model}")
        LOGGER.info(f"Ollama server: {ollama_base_url}")
        self._test_embedding_connection()

    def _test_embedding_connection(self):
        """Test connection to ollama server."""
        try:
            response = embedding(
                model=self.embedding_model,
                input=["test connection"],
                api_base=self.ollama_base_url
            )
            LOGGER.info(f"Successfully connected to ollama server")
            LOGGER.info(f"Embedding dimension: {len(response.data[0]['embedding'])}")
        except Exception as e:
            LOGGER.error(f"❌ Error connecting to ollama server: {e}")
            LOGGER.error(f"Please ensure ollama is running and the model '{self.embedding_model}' is available")
            LOGGER.error(f"Run: ollama pull nomic-embed-text")
            raise

    def load_papers_from_json(self, json_file_path: str, paper_dict_key: str = "results"):
        """
        Load papers from a JSON file or JSONL file.

        Args:
            json_file_path: Path to the JSON/JSONL file
        """
        self.papers_data = []

        with open(json_file_path, 'r') as f:
            # Try to parse as regular JSON first
            try:
                content = f.read()
                # Try parsing as a single JSON object
                try:
                    data = json.loads(content)
                    if isinstance(data[paper_dict_key], list):
                        self.papers_data = data[paper_dict_key]
                    else:
                        raise TypeError("File importer expects a list of papers.")
                except json.JSONDecodeError:
                    # Try parsing as JSONL (one JSON object per line)
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            self.papers_data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Error parsing JSON file: {e}")

        if self.limit_num_papers is not None and self.limit_num_papers > 0:
            LOGGER.warning(f"WARNING: Number of papers limited to {self.limit_num_papers} items. Set to 'None' for all papers")
            self.papers_data = self.papers_data[:self.limit_num_papers]

        LOGGER.info(f"Loaded {len(self.papers_data)} papers from {json_file_path}")

    def build_graph(self):
        """
        Build the knowledge graph from loaded papers.
        Creates nodes for papers and topics, and edges between them.
        Computes embeddings for abstracts in parallel.
        """
        topic_nodes = set()
        author_nodes = set()

        LOGGER.info(f"\nPreparing to process {len(self.papers_data)} papers...")

        # Extract all abstracts and paper info
        paper_info = []
        abstracts = []

        for paper in self.papers_data:
            paper_id = paper.get('uid', paper.get('id'))
            paper_name = paper.get('name', 'Unnamed Paper')
            abstract = paper.get('abstract', '')
            topic = paper.get('topic', 'Unknown')
            authors = paper.get('authors', [])
            keywords = paper.get("keywords", [])
            decision = paper.get("decision", "")
            session = paper.get("session", "")
            session_start_time = paper.get("starttime", "")
            session_end_time = paper.get("endtime", "")
            presentation_type = paper.get("eventtype", "")
            room_name = paper.get("room_name", "")
            project_url = paper.get("url", "")
            poster_position = paper.get("poster_position", "")
            paper_url = paper.get("paper_url", "")
            sourceid = paper.get("sourceid", "")
            virtualsite_url = paper.get("virtualsite_url", "")

            paper_info.append({
                "id": paper_id,
                "name": paper_name,
                "abstract": abstract,
                "topic": topic,
                "authors": authors,
                "keywords": keywords,
                "decisions": decision,
                "session": session,
                "session_start_time": session_start_time,
                "session_end_time": session_end_time,
                "presentation_type": presentation_type,
                "room_name": room_name,
                "project_url": project_url,
                "poster_position": poster_position,
                "paper_url": paper_url,
                "sourceid": sourceid,
                "virtualsite_url": virtualsite_url

            })
            abstracts.append(abstract)

        # Generate all embeddings in parallel
        LOGGER.info(f"\nGenerating embeddings with batch size {self.batch_size}...")
        embeddings = batch_embed_documents(
            abstracts,
            batch_size=self.batch_size,
            embedding_model=self.embedding_model,
            api_base=self.ollama_base_url
        )

        # Convert to list so that embeddings can be mapped to samples properly
        embeddings = embeddings.tolist()

        # Add nodes to graph
        LOGGER.info("\nBuilding graph structure...")
        with tqdm(total=len(paper_info), desc="Adding nodes") as pbar:
            for info, embedding in zip(paper_info, embeddings):

                # Extract author information (store as list of dicts)
                author_list = []
                if info['authors']:
                    for author in info['authors']:
                        author_info = {
                            'id': author.get('id'),
                            'fullname': author.get('fullname', ''),
                            'institution': author.get('institution', ''),
                            'url': author.get('url', '')
                        }

                        author_uid = f"{author_info['id']} - {author_info['fullname']}"
                        if author_uid not in author_nodes:
                            self.graph.add_node(
                                author_uid,
                                **author_info
                            )
                            author_nodes.add(author_uid)

                        author_list.append(author_info)

                # Add paper node with attributes
                paper_attrs = info.copy()
                del paper_attrs["authors"]

                self.graph.add_node(
                    info["id"],
                    **paper_attrs,
                    embedding=embedding,
                    authors=author_list,
                    node_type="paper"
                )

                for author in author_list:
                    self.graph.add_edge(f"{author['id']} - {author['fullname']}", info["id"], relationship="is_author_of")

                # Add topic node if it doesn't exist
                if info['topic'] and info['topic'] not in topic_nodes:
                    self.graph.add_node(
                        info['topic'],
                        node_type='topic',
                        name=info['topic']
                    )
                    topic_nodes.add(info['topic'])

                # Add edge between paper and topic
                if info['topic']:
                    self.graph.add_edge(info['id'], info['topic'], relationship='belongs_to_topic')

                pbar.update(1)

        LOGGER.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        LOGGER.info(f"   Papers: {len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'paper'])}")
        LOGGER.info(f"   Topics: {len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'topic'])}")

    def connect_similar_papers(self, similarity_threshold: float = 0.7):
        """
        Connect papers based on abstract embedding similarity using parallel processing.
        Args:
            similarity_threshold: Minimum cosine similarity to create an edge (0-1)
        """
        paper_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'paper']
        LOGGER.info(f"\nComputing similarities for {len(paper_nodes)} papers...")

        # Create pairs to compare (fast!)
        pairs = [(i, j) for i in range(len(paper_nodes)) for j in range(i + 1, len(paper_nodes))]
        LOGGER.info(f"Created {len(pairs)} pairs to compare")

        connections_added = 0

        def compute_similarity(pair_idx):
            """Compute similarity for a pair of papers."""
            i, j = pair_idx
            node1, data1 = paper_nodes[i]
            node2, data2 = paper_nodes[j]
            emb1 = data1['embedding']
            emb2 = data2['embedding']
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            if similarity >= similarity_threshold:
                return (node1, node2, float(similarity))
            return None

        # Compute similarities in parallel
        edges_to_add = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(compute_similarity, pair): pair for pair in pairs}
            with tqdm(total=len(pairs), desc="Computing similarities", unit="pair") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        edges_to_add.append(result)
                    pbar.update(1)

        # Add edges to graph
        for node1, node2, similarity in edges_to_add:
            self.graph.add_edge(
                node1,
                node2,
                relationship='similar_to',
                similarity=similarity
            )
            connections_added += 1

        LOGGER.info(f"Added {connections_added} similarity edges with threshold {similarity_threshold}")

    def get_papers_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get all papers belonging to a specific topic.

        Args:
            topic: Topic name

        Returns:
            List of paper information dictionaries
        """
        if topic not in self.graph:
            return []

        papers = []
        for neighbor in self.graph.neighbors(topic):
            node_data = self.graph.nodes[neighbor]
            if node_data.get('node_type') == 'paper':
                papers.append({
                    'id': neighbor,
                    'name': node_data.get('name'),
                    'abstract': node_data.get('abstract'),
                    'embedding': node_data.get('embedding')
                })
        return papers

    def find_similar_papers(self, paper_id: str, top_k: int = 5) -> List[tuple]:
        """
        Find the most similar papers to a given paper.

        Args:
            paper_id: ID of the paper
            top_k: Number of similar papers to return

        Returns:
            List of (paper_id, similarity_score) tuples
        """
        if paper_id not in self.graph:
            return []

        paper_data = self.graph.nodes[paper_id]
        if paper_data.get('node_type') != 'paper':
            return []

        target_embedding = paper_data['embedding']
        similarities = []

        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'paper' and node != paper_id:
                similarity = np.dot(target_embedding, data['embedding']) / \
                             (np.linalg.norm(target_embedding) * np.linalg.norm(data['embedding']))
                similarities.append((node, float(similarity), data.get('name')))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        paper_nodes = [n for n, d in self.graph.nodes(data=True)
                       if d.get('node_type') == 'paper']
        topic_nodes = [n for n, d in self.graph.nodes(data=True)
                       if d.get('node_type') == 'topic']

        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'paper_nodes': len(paper_nodes),
            'topic_nodes': len(topic_nodes),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
        }

        if nx.is_connected(self.graph):
            stats['diameter'] = nx.diameter(self.graph)
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph)

        return stats


@click.command()
@click.option("-m", "--embedding-model", default="nomic-embed-text")
@click.option("-l", "--ollama-server-url", default="http://localhost:11434")
@click.option("-b", "--embedding-gen-batch-size", default=32)
@click.option("-w", "--max-parallel-workers", default=16)
@click.option("-p", "--limit-num-papers", default=None, type=int)
@click.option("-f", "--input-json-file", default=f"{PROJECT_ROOT}/data/neurips-2025-orals-posters.json")
@click.option("-o", "--output-file", default=f"{PROJECT_ROOT}/graphs/knowledge_graph.pkl")
@click.option("-s", "--similarity-threshold", default=0.8)
def main(
    embedding_model: str,
    ollama_server_url: str,
    embedding_gen_batch_size: int,
    max_parallel_workers: int,
    limit_num_papers: int,
    input_json_file: str,
    output_file: str,
    similarity_threshold: float
):

    kg = PaperKnowledgeGraph(
        embedding_model=f"ollama/{embedding_model}",
        ollama_base_url=ollama_server_url,
        embedding_gen_batch_size=embedding_gen_batch_size,
        max_parallel_workers=max_parallel_workers,
        limit_num_papers=limit_num_papers
    )

    # Load papers from JSON file
    kg.load_papers_from_json(input_json_file)

    # Build the graph (parallel embedding generation)
    kg.build_graph()

    # Optionally connect similar papers based on embeddings (parallel)
    kg.connect_similar_papers(similarity_threshold=similarity_threshold)

    # Save the graph to disk
    save_graph(
        graph=kg.graph,
        output_path=output_file
    )

    # Print statistics
    stats = kg.get_graph_statistics()
    LOGGER.info("\nGraph Statistics:")
    for key, value in stats.items():
        LOGGER.info(f"  {key}: {value}")

    # Test run: Find similar papers
    if kg.papers_data:
        first_paper_id = kg.papers_data[0].get('uid', kg.papers_data[0].get('id'))
        LOGGER.debug(f"\nPapers similar to '{kg.graph.nodes[first_paper_id]['name']}':")
        similar = kg.find_similar_papers(first_paper_id, top_k=3)
        for pid, sim, name in similar:
            LOGGER.debug(f"  - {name} (similarity: {sim:.3f})")


# Run
if __name__ == "__main__":
    main()
