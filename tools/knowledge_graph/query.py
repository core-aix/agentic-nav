from typing import List, Dict, Any, Optional, Union
import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np


class KnowledgeGraphQueryTools:
    """Query tools for LLM to interact with knowledge graph"""

    def __init__(self, graph_path: str, embedding_model: Union[str, None] = "all-MiniLM-L6-v2"):
        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

        if embedding_model is not None and len(embedding_model) > 0:
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None

    def search_entities(
            self,
            query: str,
            top_k: int = 5,
            entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entities in the knowledge graph based on text query.

        Args:
            query: Natural language search query
            top_k: Number of top results to return
            entity_types: Optional list of entity types to filter (e.g., ['Author', 'Concept'])

        Returns:
            List of matching entities with their properties
        """
        print(f"Retrieving from {self.graph}")
        # Embed the query
        query_embedding = self.embedding_model.encode(query)

        # Collect nodes
        candidates = []
        for node, data in self.graph.nodes(data=True):
            # Filter by type if specified
            if entity_types and data.get('type') not in entity_types:
                continue

            # Get text representation
            text = data.get('text', data.get('label', data.get('name', str(node))))
            candidates.append({
                'node_id': node,
                'text': text,
                'data': data
            })

        if not candidates:
            return []

        # Embed candidates
        candidate_texts = [c['text'] for c in candidates]
        candidate_embeddings = self.embedding_model.encode(candidate_texts)

        # Calculate similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            candidate = candidates[idx]
            results.append({
                'node_id': candidate['node_id'],
                'text': candidate['text'],
                'type': candidate['data'].get('type', 'Unknown'),
                'properties': candidate['data'],
                'relevance_score': float(similarities[idx])
            })

        return results

    def get_entity_relationships(
            self,
            entity_id: str,
            relationship_types: Optional[List[str]] = None,
            max_hops: int = 1
    ) -> Dict[str, Any]:
        """
        Get all relationships for a specific entity.

        Args:
            entity_id: The node ID of the entity
            relationship_types: Optional list of relationship types to filter
            max_hops: Maximum number of hops from the entity (1 = direct neighbors only)

        Returns:
            Dictionary containing entity info and its relationships
        """
        if entity_id not in self.graph:
            return {'error': f'Entity {entity_id} not found in graph'}

        entity_data = self.graph.nodes[entity_id]

        relationships = {
            'entity_id': entity_id,
            'entity_data': entity_data,
            'outgoing': [],
            'incoming': []
        }

        # Get direct neighbors
        if max_hops >= 1:
            # Outgoing edges
            for neighbor in self.graph.successors(entity_id):
                edge_data = self.graph[entity_id][neighbor]
                rel_type = edge_data.get('type', edge_data.get('relation', 'related_to'))

                if relationship_types and rel_type not in relationship_types:
                    continue

                relationships['outgoing'].append({
                    'target_id': neighbor,
                    'target_data': self.graph.nodes[neighbor],
                    'relationship_type': rel_type,
                    'relationship_properties': edge_data
                })

            # Incoming edges (if directed graph)
            if self.graph.is_directed():
                for predecessor in self.graph.predecessors(entity_id):
                    edge_data = self.graph[predecessor][entity_id]
                    rel_type = edge_data.get('type', edge_data.get('relation', 'related_to'))

                    if relationship_types and rel_type not in relationship_types:
                        continue

                    relationships['incoming'].append({
                        'source_id': predecessor,
                        'source_data': self.graph.nodes[predecessor],
                        'relationship_type': rel_type,
                        'relationship_properties': edge_data
                    })

        return relationships

    def find_path_between_entities(
            self,
            source_entity: str,
            target_entity: str,
            max_path_length: int = 5
    ) -> Dict[str, Any]:
        """
        Find connection path between two entities.

        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            max_path_length: Maximum path length to search

        Returns:
            Path information including intermediate nodes and relationships
        """
        if source_entity not in self.graph:
            return {'error': f'Source entity {source_entity} not found'}
        if target_entity not in self.graph:
            return {'error': f'Target entity {target_entity} not found'}

        try:
            # Find shortest path
            path = nx.shortest_path(
                self.graph,
                source=source_entity,
                target=target_entity
            )

            if len(path) > max_path_length + 1:
                return {
                    'found': False,
                    'reason': f'Path too long (>{max_path_length} hops)'
                }

            # Build detailed path
            path_details = []
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]
                edge_data = self.graph[current][next_node]

                path_details.append({
                    'from': current,
                    'from_data': self.graph.nodes[current],
                    'to': next_node,
                    'to_data': self.graph.nodes[next_node],
                    'relationship': edge_data.get('type', 'related_to'),
                    'relationship_properties': edge_data
                })

            return {
                'found': True,
                'path_length': len(path) - 1,
                'path': path,
                'path_details': path_details
            }

        except nx.NetworkXNoPath:
            return {
                'found': False,
                'reason': 'No path exists between entities'
            }

    def get_entity_neighborhood(
            self,
            entity_id: str,
            radius: int = 2,
            max_nodes: int = 20
    ) -> Dict[str, Any]:
        """
        Get the neighborhood subgraph around an entity.

        Args:
            entity_id: Central entity ID
            radius: How many hops to include
            max_nodes: Maximum number of nodes to return

        Returns:
            Subgraph information including nodes and edges
        """
        if entity_id not in self.graph:
            return {'error': f'Entity {entity_id} not found'}

        # Get ego graph (subgraph centered on entity)
        ego_graph = nx.ego_graph(self.graph, entity_id, radius=radius)

        # Limit size if too large
        if len(ego_graph.nodes) > max_nodes:
            # Keep only most connected nodes
            degree_dict = dict(ego_graph.degree())
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in top_nodes[:max_nodes]]
            ego_graph = ego_graph.subgraph(top_nodes)

        # Extract nodes and edges
        nodes = []
        for node, data in ego_graph.nodes(data=True):
            nodes.append({
                'node_id': node,
                'properties': data,
                'is_center': node == entity_id
            })

        edges = []
        for source, target, data in ego_graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'relationship': data.get('type', 'related_to'),
                'properties': data
            })

        return {
            'center_entity': entity_id,
            'radius': radius,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'nodes': nodes,
            'edges': edges
        }

    def query_by_relationship_pattern(
            self,
            pattern: Dict[str, Any],
            limit: int = 10
    ) -> List[Dict[str, Any]]:
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
        results = []

        for source, target, edge_data in self.graph.edges(data=True):
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]

            # Check if pattern matches
            match = True

            if 'source_type' in pattern:
                if source_data.get('type') != pattern['source_type']:
                    match = False

            if 'target_type' in pattern:
                if target_data.get('type') != pattern['target_type']:
                    match = False

            if 'relationship' in pattern:
                rel_type = edge_data.get('type', edge_data.get('relation', ''))
                if rel_type != pattern['relationship']:
                    match = False

            if match:
                results.append({
                    'source_id': source,
                    'source_data': source_data,
                    'relationship': edge_data.get('type', 'related_to'),
                    'relationship_properties': edge_data,
                    'target_id': target,
                    'target_data': target_data
                })

                if len(results) >= limit:
                    break

        return results


def setup_kg_tool_manager(graph_path: str, embedding_model: Union[str, None] = "all-MiniLM-L6-v2"):
    return KnowledgeGraphQueryTools(graph_path=graph_path, embedding_model=embedding_model)
