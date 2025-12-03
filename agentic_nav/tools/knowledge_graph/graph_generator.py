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

from agentic_nav.utils.embedding_generator import batch_embed_documents
from agentic_nav.utils.logger import setup_logging
from agentic_nav.tools.knowledge_graph.file_handler import save_graph, load_graph


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "ollama/nomic-embed-text")
EMBEDDING_MODEL_API_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE", "http://localhost:11435")

# Setup logging

setup_logging(
    log_dir=f"{PROJECT_ROOT}/logs",
    level=os.environ.get("AGENTIC_NAV_LOG_LEVEL", "INFO"),
    console_level="INFO"
)
LOGGER = logging.getLogger(__name__)
litellm._logging._disable_debugging()
litellm.suppress_debug_info = True
litellm.set_verbose = False


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
        self.graph = nx.MultiGraph()
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

        LOGGER.info(f"Preparing to process {len(self.papers_data)} papers...")

        # Extract all abstracts and paper info
        paper_info = []
        abstracts_for_embeddings = []

        # Debug counters
        oral_count = 0
        poster_count = 0
        unknown_count = 0
        eventtype_samples = {}  # Track unique eventtypes we see

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
            import_id = paper.get("import_id", "")

            # Create unique node ID based on presentation type and uid
            # Use eventtype to determine if it's an oral or poster presentation
            presentation_type_lower = presentation_type.lower() if presentation_type else ""

            # Track unique eventtypes we encounter
            if presentation_type and presentation_type not in eventtype_samples:
                eventtype_samples[presentation_type] = 0
            if presentation_type:
                eventtype_samples[presentation_type] += 1

            if "oral" in presentation_type_lower:
                unique_node_id = f"oral_{paper_id}"
                presentation_category = "oral"
                oral_count += 1
            elif "poster" in presentation_type_lower:
                unique_node_id = f"poster_{paper_id}"
                presentation_category = "poster"
                poster_count += 1
            else:
                # Unknown presentation type - use original paper_id
                unique_node_id = f"paper_{paper_id}"
                presentation_category = "unknown"
                unknown_count += 1
                if unknown_count <= 5:  # Only log first 5 unknown types
                    LOGGER.debug(f"Paper {paper_id}: unknown eventtype='{presentation_type}'")

            paper_info.append({
                "id": paper_id,  # Keep original ID (uid)
                "node_id": unique_node_id,  # New unique node identifier
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
                "presentation_category": presentation_category,
                "room_name": room_name,
                "project_url": project_url,
                "poster_position": poster_position,
                "paper_url": paper_url,
                "sourceid": sourceid,
                "virtualsite_url": virtualsite_url,
                "import_id": import_id
            })

            # We generate the embeddings over name, abstract, and decision to get unique embeddings for every event.
            abstracts_for_embeddings.append(f"{paper_name}. {abstract} - {decision}")

        # Debug output
        LOGGER.info(f"Found {oral_count} orals, {poster_count} posters, and {unknown_count} unknown presentation types")
        LOGGER.info(f"Unique eventtypes found: {eventtype_samples}")

        # Generate all embeddings in parallel
        LOGGER.info(f"Generating embeddings with batch size {self.batch_size}...")
        embeddings = batch_embed_documents(
            abstracts_for_embeddings,
            batch_size=self.batch_size,
            embedding_model=self.embedding_model,
            api_base=self.ollama_base_url
        )

        # Convert to list so that embeddings can be mapped to samples properly
        embeddings = embeddings.tolist()

        # Track oral-poster pairs by uid for creating edges
        oral_poster_pairs = {}  # key: uid, value: {'oral': node_id, 'poster': node_id}

        # Add nodes to graph
        LOGGER.info("Building graph structure...")
        with tqdm(total=len(paper_info), desc="Adding nodes") as pbar:
            for info, embedding in zip(paper_info, embeddings):

                # Extract author information (store as list of dicts)
                author_list = []
                if info['authors']:
                    for idx, author in enumerate(info['authors']):
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
                                node_type="author",
                                **author_info
                            )
                            author_nodes.add(author_uid)

                        author_list.append(author_info)

                # Add paper node with attributes using unique node_id
                paper_attrs = info.copy()
                del paper_attrs["authors"]
                del paper_attrs["node_id"]  # Don't duplicate this in attributes

                self.graph.add_node(
                    info["node_id"],  # Use unique node_id instead of id
                    **paper_attrs,
                    embedding=embedding,
                    authors=author_list,
                    node_type="paper"
                )

                # Track oral-poster pairs by uid
                uid = info['id']  # This is the original uid

                if uid not in oral_poster_pairs:
                    oral_poster_pairs[uid] = {}

                if info['presentation_category'] == 'oral':
                    oral_poster_pairs[uid]['oral'] = info["node_id"]
                    LOGGER.debug(f"Tracked oral: {info['node_id']} for uid {uid}")
                elif info['presentation_category'] == 'poster':
                    oral_poster_pairs[uid]['poster'] = info["node_id"]
                    LOGGER.debug(f"Tracked poster: {info['node_id']} for uid {uid}")
                elif info['presentation_category'] == 'unknown':
                    # Track unknown categories too for debugging
                    if 'unknown' not in oral_poster_pairs[uid]:
                        oral_poster_pairs[uid]['unknown'] = []
                    oral_poster_pairs[uid]['unknown'].append(info["node_id"])

                # Add edges to authors
                for idx, author in enumerate(author_list):
                    self.graph.add_edge(
                        f"{author['id']} - {author['fullname']}", info["node_id"],
                        relationship="is_author_of",
                        author_order=idx
                    )

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
                    self.graph.add_edge(info['node_id'], info['topic'], relationship='belongs_to_topic')

                pbar.update(1)

        # Debug oral-poster pairs before adding edges
        LOGGER.info(f"\nOral-poster pairs tracked: {len(oral_poster_pairs)}")
        complete_pairs = [k for k, v in oral_poster_pairs.items() if 'oral' in v and 'poster' in v]
        incomplete_pairs = [k for k, v in oral_poster_pairs.items() if 'oral' not in v or 'poster' not in v]

        LOGGER.info(f"Complete pairs (both oral and poster): {len(complete_pairs)}")
        LOGGER.info(f"Incomplete pairs: {len(incomplete_pairs)}")

        if complete_pairs:
            LOGGER.info(f"Sample complete pairs (first 3):")
            for uid in complete_pairs[:3]:
                LOGGER.info(f"  uid {uid}: oral={oral_poster_pairs[uid].get('oral')}, poster={oral_poster_pairs[uid].get('poster')}")

        if incomplete_pairs:
            LOGGER.warning(f"Sample incomplete pairs:")
            for uid in incomplete_pairs[:5]:
                LOGGER.warning(f"  uid {uid}: {oral_poster_pairs[uid]}")

        # Add edges between oral-poster pairs
        oral_poster_edges_added = 0
        for uid, pair in oral_poster_pairs.items():
            if 'oral' in pair and 'poster' in pair:
                self.graph.add_edge(
                    pair['oral'],
                    pair['poster'],
                    relationship='oral_poster_pair',
                    uid=uid
                )
                oral_poster_edges_added += 1
                LOGGER.debug(f"Added edge: {pair['oral']} <-> {pair['poster']}")

        LOGGER.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        LOGGER.info(f"   Papers: {len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'paper'])}")
        LOGGER.info(f"   Topics: {len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'topic'])}")
        LOGGER.info(f"   Authors: {len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'author'])}")
        LOGGER.info(f"   Oral-Poster pairs connected: {oral_poster_edges_added}")

    def connect_similar_papers(self, similarity_threshold: float = 0.7):
        """
        Connect papers based on abstract embedding similarity using parallel processing.
        Args:
            similarity_threshold: Minimum cosine similarity to create an edge (0-1)
        """
        paper_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'paper']
        LOGGER.info(f"Computing similarities for {len(paper_nodes)} papers...")

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

        # if nx.is_connected(self.graph):
        # stats['diameter'] = nx.diameter(self.graph)
        # stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph)

        return stats

    def get_poster_oral_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about poster-oral pairs in the knowledge graph.
        Pairs are matched by 'uid' attribute where both oral and poster versions
        share the same uid.

        Returns:
            Dictionary with poster-oral pair statistics
        """
        paper_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'paper']

        # Count by presentation category and track by uid
        orals = {}  # key: uid, value: node_id
        posters = {}  # key: uid, value: node_id
        other_papers = 0

        for node_id, data in paper_nodes:
            presentation_category = data.get('presentation_category', 'unknown')
            uid = data.get('id', '')  # This is the original uid

            if presentation_category == 'oral':
                orals[uid] = node_id
            elif presentation_category == 'poster':
                posters[uid] = node_id
            else:
                other_papers += 1

        # Find matched pairs (can also check edges)
        matched_pairs = []
        for uid in orals.keys():
            if uid in posters:
                oral_id = orals[uid]
                poster_id = posters[uid]

                # Verify edge exists
                has_edge = self.graph.has_edge(oral_id, poster_id)

                matched_pairs.append({
                    'uid': uid,
                    'oral_id': oral_id,
                    'poster_id': poster_id,
                    'oral_name': self.graph.nodes[oral_id].get('name', ''),
                    'poster_name': self.graph.nodes[poster_id].get('name', ''),
                    'has_edge': has_edge
                })

        # Find orals without corresponding posters
        orals_without_posters = [
            uid for uid in orals.keys()
            if uid not in posters
        ]

        # Find posters without corresponding orals
        posters_without_orals = [
            uid for uid in posters.keys()
            if uid not in orals
        ]

        # Check edges
        edge_tracker = 0
        for pair in matched_pairs:
            if pair['has_edge']:
                edge_tracker += 1

        stats = {
            'total_papers': len(paper_nodes),
            'total_orals': len(orals),
            'total_posters': len(posters),
            'other_papers': other_papers,
            'matched_pairs': len(matched_pairs),
            'pairs_with_edges': edge_tracker,
            'orals_without_posters': len(orals_without_posters),
            'posters_without_orals': len(posters_without_orals),
            'pair_details': matched_pairs
        }

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
@click.option("--stats-only", is_flag=True)
def main(
        embedding_model: str,
        ollama_server_url: str,
        embedding_gen_batch_size: int,
        max_parallel_workers: int,
        limit_num_papers: int,
        input_json_file: str,
        output_file: str,
        similarity_threshold: float,
        stats_only: bool = False
):

    kg = PaperKnowledgeGraph(
        embedding_model=embedding_model,
        ollama_base_url=ollama_server_url,
        embedding_gen_batch_size=embedding_gen_batch_size,
        max_parallel_workers=max_parallel_workers,
        limit_num_papers=limit_num_papers
    )

    if not stats_only:
        # Load papers from JSON file
        kg.load_papers_from_json(input_json_file)

        # Build the graph (parallel embedding generation)
        kg.build_graph()

        # Optionally connect similar papers based on embeddings (parallel)
        kg.connect_similar_papers(similarity_threshold=similarity_threshold)

        # DEBUG: Check edges before saving
        LOGGER.info("\n=== BEFORE SAVING ===")
        oral_poster_edges_before = [
            (s, t, d) for s, t, d in kg.graph.edges(data=True)
            if d.get('relationship') == 'oral_poster_pair'
        ]
        LOGGER.info(f"Oral-poster edges before save: {len(oral_poster_edges_before)}")
        if oral_poster_edges_before:
            sample = oral_poster_edges_before[0]
            LOGGER.info(f"Sample edge: {sample[0]} -> {sample[1]}, data: {sample[2]}")

        # Save the graph to disk
        save_graph(
            graph=kg.graph,
            output_path=output_file
        )

        # DEBUG: Load it back and check
        LOGGER.info("\n=== AFTER LOADING BACK ===")
        test_load = load_graph(output_file)
        oral_poster_edges_after = [
            (s, t, d) for s, t, d in test_load.edges(data=True)
            if d.get('relationship') == 'oral_poster_pair'
        ]
        LOGGER.info(f"Oral-poster edges after load: {len(oral_poster_edges_after)}")
        if oral_poster_edges_after:
            sample = oral_poster_edges_after[0]
            LOGGER.info(f"Sample edge: {sample[0]} -> {sample[1]}, data: {sample[2]}")
        else:
            LOGGER.warning("Edges lost during save/load!")
            # Check what edges DO exist
            LOGGER.info("Sample of edges that exist after load:")
            for i, (s, t, d) in enumerate(test_load.edges(data=True)):
                if i >= 3:
                    break
                LOGGER.info(f"  {s} -> {t}, data keys: {list(d.keys())}, relationship: {d.get('relationship')}")
    else:
        kg.graph = load_graph(output_file)

    # Get Poster Oral Pairs
    po_stats = kg.get_poster_oral_statistics()
    LOGGER.info("Paper statistics: ")
    for key, value in po_stats.items():
        if type(value) is int:
            LOGGER.info(f"  {key}: {value}")

    # Print statistics
    stats = kg.get_graph_statistics()
    LOGGER.info("Graph Statistics:")
    for key, value in stats.items():
        LOGGER.info(f"  {key}: {value}")

    # Test run: Find similar papers
    if kg.papers_data:
        first_paper = kg.papers_data[0]
        first_paper_uid = first_paper.get('uid', first_paper.get('id'))
        first_paper_eventtype = first_paper.get('eventtype', '').lower()

        # Construct the correct node_id based on eventtype
        if "oral" in first_paper_eventtype:
            first_paper_node_id = f"oral_{first_paper_uid}"
        elif "poster" in first_paper_eventtype:
            first_paper_node_id = f"poster_{first_paper_uid}"
        else:
            first_paper_node_id = f"paper_{first_paper_uid}"

        if first_paper_node_id in kg.graph:
            LOGGER.debug(f"Papers similar to '{kg.graph.nodes[first_paper_node_id]['name']}':")
            similar = kg.find_similar_papers(first_paper_node_id, top_k=3)
            for pid, sim, name in similar:
                LOGGER.debug(f"  - {name} (similarity: {sim:.3f})")
        else:
            LOGGER.warning(f"First paper node '{first_paper_node_id}' not found in graph")


# Run
if __name__ == "__main__":
    main()
