from enum import Enum

from llm_agents.tools.knowledge_graph.graph_traversal_strategies.breadth_first_random import _graph_traversal_bfs_random
from llm_agents.tools.knowledge_graph.graph_traversal_strategies.depth_first_random import _graph_traversal_dfs_random
from llm_agents.tools.knowledge_graph.graph_traversal_strategies.neo4j_builtin import _graph_traversal_cypher


class TraversalStrategy(Enum):
    """Traversal strategy options"""
    BFS = "breadth_first"
    DFS = "depth_first"
    BFS_RANDOM = "breadth_first_random"
    DFS_RANDOM = "depth_first_random"


