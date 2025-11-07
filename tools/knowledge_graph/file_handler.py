import pickle
import networkx as nx


def save_graph(graph: nx.Graph, output_path: str):
    """
    Save the graph to a file using pickle.

    Args:
        output_path: Path to save the graph
    """
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
        f.close()
    print(f"Graph saved to {output_path}")


def load_graph(input_path: str) -> nx.Graph:
    """
    Load a graph from a pickle file.

    Args:
        input_path: Path to the saved graph
    """
    with open(input_path, 'rb') as f:
        graph = pickle.load(f)
        f.close()
    print(f"Graph loaded from {input_path}")
    return graph