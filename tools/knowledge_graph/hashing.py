import hashlib
import json
import re
import networkx as nx


def hash_multigraph(graph: nx.Graph):
    """Hash a multigraph including all parallel edges"""
    # Get all edges with their keys (for parallel edges)
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = sorted([
            (str(u), str(v), str(k), str(data))
            for u, v, k, data in graph.edges(keys=True, data=True)
        ])
    else:
        edges = sorted([
            (str(u), str(v), str(data))
            for u, v, data in graph.edges(data=True)
        ])

    nodes = sorted([
        (str(n), str(data))
        for n, data in graph.nodes(data=True)
    ])

    graph_repr = json.dumps({
        'nodes': nodes,
        'edges': edges,
        'directed': graph.is_directed()
    }, sort_keys=True)

    return hashlib.sha256(graph_repr.encode()).hexdigest()


def parse_graph_filename(filename):
    """Extract components from graph filename"""
    pattern = r'(\d{8}_\d{6})_(\w+)_([a-f0-9]{64}|nohash)\.(\w+)'
    filename = str(filename)

    match = re.search(pattern, filename)

    if match:
        hash_val = match.group(3)
        return {
            'timestamp': match.group(1),
            'name': match.group(2),
            'hash': hash_val if hash_val != 'nohash' else None,
            'extension': match.group(4),
            'has_hash': hash_val != 'nohash'
        }
    return None