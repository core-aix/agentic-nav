from dataclasses import dataclass

from llm_agents.agents.base import LLMAgent


from llm_agents.tools import search_similar_papers, find_neighboring_papers, traverse_graph  # <- the tools we expose


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
    messages = [{
    "role": "system",
    "content": "You are an assistant who can help browsing NeurIPS 2025 papers. You are provided with a search tool that can search all accepted papers of NeurIPS 2025. However, note that the search tool only takes paper titles and abstracts as input keywords; it cannot take anything else as the input keywords. However, the output of the search includes various metadata fields such as authors, affiliations, and session times. If your answer includes a list of papers, cite titles, abstracts, and OpenReview URLs in your answers. Every response must include a brief joke."
    }]
    tools = [search_similar_papers, find_neighboring_papers, traverse_graph]
