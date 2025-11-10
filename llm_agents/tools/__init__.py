import inspect

from llm_agents.tools.rag_search import search_papers
from llm_agents.tools.knowledge_graph import search_similar_papers, find_neighboring_papers, traverse_graph


def get_all_tools():
    current_module = inspect.getmodule(inspect.currentframe())
    tools = []

    for name, obj in vars(current_module).items():
        # Check if it's callable and not a built-in/private function
        if callable(obj) and not name.startswith('_') and name != 'get_all_tools':
            # Filter to only imported functions (not classes, modules, etc.)
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                tools.append(obj)

    return tools


if __name__ == "__main__":
    print(get_all_tools())