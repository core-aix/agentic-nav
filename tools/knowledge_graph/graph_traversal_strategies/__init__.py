from enum import Enum

from .breadth_first_random import _graph_traversal_bfs_random
from .depth_first_random import _graph_traversal_dfs_random
from .neo4j_builtin import _graph_traversal_cypher


class TraversalStrategy(Enum):
    """Traversal strategy options"""
    BFS = "breadth_first"
    DFS = "depth_first"
    BFS_RANDOM = "breadth_first_random"
    DFS_RANDOM = "depth_first_random"


