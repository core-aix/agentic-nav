import os

from dataclasses import dataclass
from llm_agents.agents.base import LLMAgent
from llm_agents.tools import search_similar_papers, find_neighboring_papers, traverse_graph  # <- the tools we expose

DEFAULT_NEURIPS2025_AGENT_ARGS = {
    "model": os.environ.get("OLLAMA_LOCAL_AGENT_MODEL_NAME", "gpt-oss:120b-cloud"),
    "api_base": os.environ.get("OLLAMA_LOCAL_AGENT_MODEL_API_BASE", "https://ollama.com"),
    "api_key": os.environ.get("OLLAMA_API_KEY"),
    "llm_args": {"temperature": 0.2, "max_tokens": 6000, "num_ctx": 131072},
    "global_tool_args": {"max_num_papers": 10}
}


system = {
    "role": "system",
    "content": "You are an assistant who can help browsing NeurIPS 2025 papers. "
               "You are provided with a search tool that can search all accepted papers of NeurIPS 2025. "
               "However, note that the search tool only takes paper titles and abstracts as input keywords; "
               "it cannot take anything else as the input keywords. "
               "However, the output of the search includes various metadata fields such as authors, affiliations, "
               "and session times. "
               "If your answer includes a list of papers, cite titles, abstracts, and OpenReview URLs in your answers."
}


@dataclass
class NeurIPS2025Agent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = [{
            "role": "system",
            "content": "You are an assistant who can help browsing NeurIPS 2025 papers. You are provided with a search tool that can search all accepted papers of NeurIPS 2025. However, note that the search tool only takes paper titles and abstracts as input keywords; it cannot take anything else as the input keywords. However, the output of the search includes various metadata fields such as authors, affiliations, and session times. If your answer includes a list of papers, cite titles, abstracts, and OpenReview URLs in your answers."
        }]
        self.tools = [search_similar_papers, find_neighboring_papers, traverse_graph]
