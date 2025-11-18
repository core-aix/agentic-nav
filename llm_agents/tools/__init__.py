from llm_agents.tools.knowledge_graph import search_similar_papers, find_neighboring_papers, traverse_graph


__all__ = [
    'search_similar_papers',
    'find_neighboring_papers',
    'traverse_graph',
]


def get_all_tools():
    """Get all tools as a dictionary."""
    return [globals()[name] for name in __all__]