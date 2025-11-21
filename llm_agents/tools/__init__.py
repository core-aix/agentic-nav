from llm_agents.tools.knowledge_graph import search_similar_papers, find_neighboring_papers, traverse_graph
from llm_agents.tools.session_routing import build_visit_schedule


__all__ = [
    'search_similar_papers',
    'find_neighboring_papers',
    'traverse_graph',
    'build_visit_schedule',
]


def get_all_tools():
    """Get all tools as a dictionary."""
    return [globals()[name] for name in __all__]