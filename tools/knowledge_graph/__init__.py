"""
TBD
"""
import json
import pickle

import networkx as nx
import pickle

from datetime import datetime

try:
    from .schema import DynamicSchemaGenerator
except ImportError:
    from schema import DynamicSchemaGenerator

try:
    from .extractor import HybridKGBuilder
except ImportError:
    from extractor import HybridKGBuilder

try:
    from .hashing import hash_multigraph, parse_graph_filename
except ImportError:
    from hashing import hash_multigraph, parse_graph_filename

try:
    from .visualizer import GraphVisualizer
except ImportError:
    from visualizer import GraphVisualizer

try:
    from .query import KnowledgeGraphQueryTools, setup_kg_tool_manager
except ImportError:
    from query import KnowledgeGraphQueryTools, setup_kg_tool_manager

from pathlib import Path
from typing import List, Dict, Optional, Union, Any

PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent


def hash_graph(graph: nx.Graph, file_path: Path):
    """
    Creates a unique hash.
    """
    if graph.is_multigraph():
        hash_value = hash_multigraph(graph)
    else:
        hash_value = nx.weisfeiler_lehman_graph_hash(graph)

    # Check if hash already exists
    existing_file_hashes = []
    for link in file_path.iterdir():
        if link.is_file():
            print(link)
            file_metadata = parse_graph_filename(filename=link)
            existing_file_hashes.append(file_metadata["hash"])

    if hash_value in existing_file_hashes:
        print(f"There is another graph with the same hash as created for the query at hand: {hash_value}")

    return hash_value


def get_or_build_dynamic_knowledge_graph(
    prompt: str,
    paper: Dict[str, List],
    max_papers: int = 50,
    llm_name: str = "ollama_chat/gpt-oss:20b",
    api_base: str = "http://localhost:11434",
    llm_default_params={}
):
    graph_builder = HybridKGBuilder(
        paper=paper,
        llm_name=llm_name,
        api_base=api_base,
        llm_default_params=llm_default_params
    )

    graph, schema = graph_builder.build_kg_for_query(
        user_query=prompt,
        max_papers=max_papers
    )

    stats = graph_builder.query_graph(
        query=prompt,
        schema=schema
    )

    # store graph on disk
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = Path(f"{PROJECT_ROOT_PATH}/graphs/")

    try:
        hsh = hash_graph(graph, file_path=file_path)
    except Exception as e:
        print(f"Caught Exception - {e}")
        hsh = "nohash"

    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)

    with open(f"{file_path}/{timestamp}_graph_{hsh}.pkl", "wb") as f:
        pickle.dump(graph, f)
        f.close()

    stats["path_on_disk"] = f"{file_path}/{timestamp}_graph_{hsh}.pkl"

    return graph, schema, stats


def search_knowledge_graph(
    query: str,
    top_k: int,
    path_to_graph_pkl: str,
    entity_types: Optional[List[str]] = None,
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Search for entities in the knowledge graph based on text query.

    Args:
        query: Natural language search query
        top_k: Number of top results to return
        entity_types: Optional list of entity types to filter (e.g., ['Author', 'Concept'])

    Returns:
        List of matching entities with their properties
    """
    tool_manager = setup_kg_tool_manager(
        graph_path=path_to_graph_pkl,
        embedding_model=embedding_model
    )

    entities = tool_manager.search_entities(
        query=query,
        top_k=top_k,
        entity_types=entity_types
    )

    return entities


def get_knowledge_graph_relationships(
    entity_id: Union[str, int],
    path_to_graph_pkl: str,
    relationship_types: Optional[List[str]] = None,
    max_hops: int = None
):
    """
    Get all relationships for a specific entity.

    Args:
        entity_id: The node ID of the entity
        relationship_types: Optional list of relationship types to filter
        max_hops: Maximum number of hops from the entity (1 = direct neighbors only)

    Returns:
        Dictionary containing entity info and its relationships
    """
    tool_manager = setup_kg_tool_manager(
        graph_path=path_to_graph_pkl,
        embedding_model=""
    )

    relationships = tool_manager.get_entity_relationships(
        entity_id=entity_id,
        relationship_types=relationship_types,
        max_hops=max_hops
    )

    return relationships


def find_path_between_nodes_in_knowledge_graph(
    source_entity: str,
    target_entity: str,
    path_to_graph_pkl: str,
    max_path_length: int = 10
):
    """
    Find connection path between two entities.

    Args:
        source_entity: Source entity ID
        target_entity: Target entity ID
        max_path_length: Maximum path length to search

    Returns:
        Path information including intermediate nodes and relationships
    """
    tool_manager = setup_kg_tool_manager(
        graph_path=path_to_graph_pkl,
        embedding_model=None
    )

    path = tool_manager.find_path_between_entities(
        source_entity=source_entity,
        target_entity=target_entity,
        max_path_length=max_path_length
    )

    return path


def get_entity_neighborhood_in_knowledge_graph(
    entity_id: str,
    path_to_graph_pkl: str,
    radius: int = 2,
    max_nodes: int = 20
):
    """
    Get the neighborhood subgraph around an entity.

    Args:
        entity_id: Central entity ID
        radius: How many hops to include
        max_nodes: Maximum number of nodes to return

    Returns:
        Subgraph information including nodes and edges
    """
    tool_manager = setup_kg_tool_manager(
        graph_path=path_to_graph_pkl,
        embedding_model=None
    )

    neighborhood = tool_manager.get_entity_neighborhood(
        entity_id=entity_id,
        radius=radius,
        max_nodes=max_nodes
    )

    return neighborhood


def query_knowledge_graph_for_relationship_patterns(
    pattern: Dict[str, Any],
    path_to_graph_pkl: str,
    limit: int = 10
):
    """
    Query graph using relationship patterns.

    Args:
        pattern: Pattern dict, e.g.:
            {
                'source_type': 'Author',
                'relationship': 'wrote',
                'target_type': 'Paper'
            }
        limit: Maximum results to return

    Returns:
        List of matching patterns with entity details
    """
    tool_manager = setup_kg_tool_manager(
        graph_path=path_to_graph_pkl,
        embedding_model=None
    )

    entities = tool_manager.query_by_relationship_pattern(
        pattern=pattern,
        limit=limit
    )

    return entities


def visualize_graph(path_to_graph_pkl: str, engine: str = "pyvis"):
    vis = GraphVisualizer()
    vis(path_to_graph=path_to_graph_pkl, engine=engine)


if __name__ == "__main__":
    project_root_path = Path(__file__).parent.parent.parent
    graph_path = f"{project_root_path}/graphs/20251103_151145_graph_5b3829ee573499084ba676a7f6823562d68694124f7d46a30cc4ce55ab5441e3.pkl"

    # Test the knowledge graph search functionality
    entities = search_knowledge_graph(
        query="lora",
        top_k=1,
        path_to_graph_pkl=graph_path,
    )

    # Get direct relationships & neighborhood
    relationships = {}
    neighborhood = {}
    for entity in entities:

        relationships[entity["node_id"]] = get_knowledge_graph_relationships(
            entity_id=entity["node_id"],
            path_to_graph_pkl=graph_path,
            max_hops=1
        )

        neighborhood[entity["node_id"]] = get_entity_neighborhood_in_knowledge_graph(
            entity_id=entity["node_id"],
            path_to_graph_pkl=graph_path,
            max_nodes=20,
            radius=10
        )

    print(neighborhood)


    # with open(f"{project_root_path}/data/neurips-2025-orals-posters.json", "r") as f:
    #     papers = json.load(fp=f)
    #     f.close()
    #
    # graph, schema, stats = get_or_build_dynamic_knowledge_graph(
    #     prompt="What deep learning methods are used for dimension reduction?",
    #     paper=papers,
    #     max_papers=100
    # )
    #
    # visualize_graph(path_to_graph_pkl=stats["path_on_disk"])
    #
    # print(f"SCHEMA: {schema}")
    # print(f"GRAPH STATS: {stats['nodes']} - {stats['edges']} - {dict(stats['node_types'])}")
